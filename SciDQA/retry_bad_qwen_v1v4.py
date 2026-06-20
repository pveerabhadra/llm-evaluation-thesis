"""
Retry bad Qwen 3.5 records from batches v1–v4.

"Bad" records are those where the model exhausted its 4096-token budget during
thinking and never wrote a real answer — identified by response_text == reasoning
(the in-place patch applied earlier).  This script re-runs only those
question+condition pairs with the correct 8192-token limit.

Results are written to: scidqa_qwen3.5_retries_v1v4.jsonl

Run AFTER the main Qwen run is fully complete:
  python3 retry_bad_qwen_v1v4.py
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import sys
import time
import threading
import concurrent.futures

import pandas as pd
from openai import OpenAI

import nltk
nltk.download("punkt",      quiet=True)
nltk.download("punkt_tab",  quiet=True)
from nltk.tokenize import sent_tokenize

import numpy as np
from rouge_score import rouge_scorer as _rouge_scorer_module
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "https://litellm.uni-osnabrueck.de/v1")
LITELLM_API_KEY  = os.getenv("LITELLM_API_KEY", "")
MODEL_NAME          = "Qwen/Qwen3.5-122B-A10B-FP8"
RATE_LIMIT          = 100           # conservative for retries
MAX_WORKERS         = 10
CONTEXT_CHARS       = 140_000
SENTENCES_PER_CHUNK = 10
CHUNK_OVERLAP       = 1
DENSE_EMBED_MODEL   = "all-MiniLM-L6-v2"
CONDITIONS          = ["no_retrieval", "rag_top3", "rag_top5", "rag_dense", "long_context"]

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "scidqa_qwen3.5_retries_v1v4.jsonl")
VERSION_MAP = {"Initial": "initial", "Revised": "final"}

client = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_NO_RETRIEVAL = (
    "You are a knowledgeable research assistant. "
    "Answer the question accurately based on your knowledge of scientific literature. "
    "If you are uncertain, provide your best answer based on the general field."
)
SYSTEM_RAG = (
    "You are a research assistant. "
    "You are given relevant excerpts retrieved from a research paper. "
    "Answer the question by grounding your response in the provided excerpts. "
    "Be as complete and accurate as possible. "
    "If the excerpts do not contain enough information, state what is missing."
)
SYSTEM_LONG_CONTEXT = (
    "You are a research assistant. "
    "Answer the question based on the research paper provided. "
    "Ground your answer in the paper's content and be as complete and accurate as possible."
)

def build_prompt_no_retrieval(q): return f"Question: {q}"
def build_prompt_rag(q, chunks):
    block = "\n\n---\n\n".join(f"[Excerpt {i+1}]\n{c}" for i, c in enumerate(chunks))
    return f"Retrieved Paper Excerpts:\n\n{block}\n\n---\n\nQuestion: {q}"
def build_prompt_long_context(q, text):
    return f"Paper:\n\n{text[:CONTEXT_CHARS]}\n\n---\n\nQuestion: {q}"

# ── Chunking & retrieval ───────────────────────────────────────────────────────
def chunk_text(text):
    chunks, stride = [], max(1, SENTENCES_PER_CHUNK - CHUNK_OVERLAP)
    for para in text.split("\n"):
        para = para.strip()
        if not para: continue
        sents = sent_tokenize(para)
        for i in range(0, len(sents), stride):
            c = " ".join(sents[i:i+SENTENCES_PER_CHUNK]).strip()
            if c: chunks.append(c)
    return chunks

def retrieve_chunks(q, text, k=3):
    chunks = chunk_text(text)
    if not chunks: return [], []
    bm25   = BM25Okapi([c.lower().split() for c in chunks])
    scores = bm25.get_scores(q.lower().split())
    idx    = sorted(sorted(range(len(scores)), key=lambda i: -scores[i])[:k])
    return [chunks[i] for i in idx], idx

_embed_model = None
_embed_lock  = threading.Lock()
def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(DENSE_EMBED_MODEL)
    return _embed_model

def retrieve_chunks_dense(q, text, k=3):
    chunks = chunk_text(text)
    if not chunks: return [], []
    m = get_embed_model()
    with _embed_lock:
        ce = m.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        qe = m.encode([q],    normalize_embeddings=True, show_progress_bar=False)[0]
    scores = np.dot(ce, qe)
    idx    = sorted(sorted(range(len(scores)), key=lambda i: -scores[i])[:k])
    return [chunks[i] for i in idx], idx

# ── Metrics ────────────────────────────────────────────────────────────────────
_rouge = _rouge_scorer_module.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
def compute_rouge(pred, ref):
    if not pred or not ref:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0, "rouge_avg": 0.0}
    s  = _rouge.score(ref, pred)
    r1, r2, rl = s["rouge1"].fmeasure, s["rouge2"].fmeasure, s["rougeL"].fmeasure
    return {"rouge_1": round(r1,4), "rouge_2": round(r2,4),
            "rouge_l": round(rl,4), "rouge_avg": round((r1+r2+rl)/3, 4)}

_NO_ANS = ["i don't know","i do not know","cannot answer","not mentioned","not provided",
           "not discussed","i cannot","no information","insufficient information",
           "not enough context","cannot find","does not contain","not present in"]
def detect_no_answer(t):
    if not t: return True
    return any(p in t.lower() for p in _NO_ANS)

def ngram_score(resp, src, n=4):
    if not resp or not src: return None
    def ng(txt):
        w = txt.lower().split()
        return {tuple(w[i:i+n]) for i in range(max(0, len(w)-n+1))}
    r, s = ng(resp), ng(src)
    return round(len(r & s) / len(r), 4) if r else None

# ── Rate limiter ───────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, rpm):
        self._iv, self._lock, self._last = 60/rpm, threading.Lock(), 0.0
    def acquire(self):
        with self._lock:
            w = self._iv - (time.time() - self._last)
            if w > 0: time.sleep(w)
            self._last = time.time()

rate_limiter = RateLimiter(RATE_LIMIT)
write_lock   = threading.Lock()

# ── API call ───────────────────────────────────────────────────────────────────
def call_model(sys_p, usr_p, retries=3):
    for attempt in range(retries):
        rate_limiter.acquire()
        try:
            t0  = time.time()
            res = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": sys_p},
                          {"role": "user",   "content": usr_p}],
                temperature=0,
                max_tokens=8192,
            )
            latency = round(time.time() - t0, 3)
            msg     = res.choices[0].message
            content = msg.content
            reasoning = (getattr(msg, "reasoning_content", None)
                         or (msg.model_extra or {}).get("reasoning_content"))
            usage  = res.usage
            tokens = {"prompt_tokens":     usage.prompt_tokens     if usage else None,
                      "completion_tokens": usage.completion_tokens if usage else None,
                      "total_tokens":      usage.total_tokens      if usage else None}
            if content is None:
                if reasoning:
                    return {"text": reasoning.strip(), "reasoning": reasoning,
                            "latency": latency, "tokens": tokens, "error": None}
                return {"text": None, "reasoning": None, "latency": latency,
                        "tokens": tokens, "error": "null content from model"}
            return {"text": content.strip(), "reasoning": reasoning,
                    "latency": latency, "tokens": tokens, "error": None}
        except Exception as e:
            err = str(e)
            if "401" in err: print("\n[FATAL] 401 — check API key."); sys.exit(1)
            if "404" in err: print("\n[FATAL] 404 — model not found."); sys.exit(1)
            if attempt < retries - 1: time.sleep(2**attempt * 5)
            else: return {"text": None, "reasoning": None, "latency": 0,
                          "tokens": {}, "error": f"failed: {err[:120]}"}
    return {"text": None, "reasoning": None, "latency": 0, "tokens": {}, "error": "unknown"}

# ── Per-task retry ─────────────────────────────────────────────────────────────
def retry_one(task):
    row, paper_text, condition = task
    gold_answer    = str(row["ans"])
    paper_available = bool(paper_text)
    grounding_src, chunks_used, rag_chars_given, paper_chars_given = None, [], 0, 0

    if condition == "no_retrieval":
        sys_p = SYSTEM_NO_RETRIEVAL
        usr_p = build_prompt_no_retrieval(row["que"])
    elif condition in ("rag_top3", "rag_top5", "rag_dense"):
        if not paper_text: return None
        if condition == "rag_dense":
            rc, chunks_used = retrieve_chunks_dense(row["que"], paper_text)
        else:
            k = 3 if condition == "rag_top3" else 5
            rc, chunks_used = retrieve_chunks(row["que"], paper_text, k)
        sys_p = SYSTEM_RAG
        usr_p = build_prompt_rag(row["que"], rc)
        rag_chars_given = sum(len(c) for c in rc)
        grounding_src   = "\n\n".join(rc)
    else:  # long_context
        if not paper_text: return None
        sys_p = SYSTEM_LONG_CONTEXT
        usr_p = build_prompt_long_context(row["que"], paper_text)
        paper_chars_given = min(len(paper_text), CONTEXT_CHARS)
        grounding_src     = paper_text[:CONTEXT_CHARS]

    response  = call_model(sys_p, usr_p)
    rt        = response["text"] or ""
    reasoning = response.get("reasoning") or ""
    tokens    = response.get("tokens") or {}
    rouge     = compute_rouge(rt, gold_answer)
    ng        = ngram_score(rt, grounding_src) if grounding_src else None

    record = {
        "id": int(row.name), "model": MODEL_NAME, "condition": condition,
        "pid": row["pid"], "venue": row["venue"], "year": int(row["year"]),
        "version": row["version"], "question": row["que"], "gold_answer": gold_answer,
        "response_text": rt or None, "reasoning": reasoning or None,
        "paper_available": paper_available, "paper_chars_given": paper_chars_given,
        "rag_chars_given": rag_chars_given, "chunks_used": chunks_used,
        "response_length_chars": len(rt), "reasoning_length_chars": len(reasoning),
        "no_answer_signal": detect_no_answer(rt),
        "latency_s": response["latency"],
        "prompt_tokens":     tokens.get("prompt_tokens"),
        "completion_tokens": tokens.get("completion_tokens"),
        "total_tokens":      tokens.get("total_tokens"),
        "rouge_1": rouge["rouge_1"], "rouge_2": rouge["rouge_2"],
        "rouge_l": rouge["rouge_l"], "rouge_avg": rouge["rouge_avg"],
        "ngram_grounding_score": ng, "error": response["error"],
        "is_retry": True,
    }

    with write_lock:
        with open(OUTPUT_FILE, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    return record

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Find bad records in v1–v4 only
    batch_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, "archive", "qwen_4k_archive", "scidqa_qwen3.5_v[1234].jsonl")))
    if not batch_files:
        print("No Qwen v1–v4 files found.")
        sys.exit(0)

    bad_records = []
    for fp in batch_files:
        with open(fp) as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                r = json.loads(line)
                # Bad records: response_text equals reasoning (thinking monologue patch)
                if (r.get("reasoning")
                        and r.get("response_text")
                        and r["response_text"] == r["reasoning"]):
                    bad_records.append(r)

    if not bad_records:
        print("No bad records found in v1–v4. Nothing to retry.")
        sys.exit(0)

    from collections import Counter
    by_file = Counter(os.path.basename(fp) for fp in batch_files
                      for _ in range(sum(1 for r in bad_records
                                         if r.get("response_text") == r.get("reasoning"))))
    print(f"Found {len(bad_records)} bad records across v1–v4.")
    by_batch = Counter()
    for fp in batch_files:
        with open(fp) as fh:
            cnt = sum(1 for l in fh if l.strip() and
                      (r := json.loads(l)) and
                      r.get("reasoning") and r.get("response_text") == r.get("reasoning"))
            by_batch[os.path.basename(fp)] = cnt
    for fname, cnt in sorted(by_batch.items()):
        print(f"  {fname}: {cnt} bad records")

    print("\nLoading dataset and paper texts...")
    df = pd.read_excel(os.path.join(DATA_DIR, "SciDQADataset.xlsx"))
    df = df.set_index("id")
    with open(os.path.join(DATA_DIR, "papers_fulltext_nougat.pkl"), "rb") as f:
        fulltext = pickle.load(f)

    tasks = []
    for r in bad_records:
        qid = r["id"]
        condition = r["condition"]
        try:
            row = df.loc[qid]
        except KeyError:
            print(f"  Warning: id={qid} not found in dataset, skipping.")
            continue
        vkey  = VERSION_MAP.get(row["version"], "initial")
        ptext = fulltext.get(vkey, {}).get(row["pid"], "")
        tasks.append((row, ptext, condition))

    print(f"\nRetrying {len(tasks)} records at {RATE_LIMIT} req/min (max_tokens=8192)...")

    success, failed = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(retry_one, t): t for t in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                rec = future.result()
                if rec and not rec.get("error"):
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"\n  Worker error: {e}")
                failed += 1

    print(f"\nDone.  {success} succeeded, {failed} still failed.")
    print(f"Results saved to: {os.path.basename(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()
