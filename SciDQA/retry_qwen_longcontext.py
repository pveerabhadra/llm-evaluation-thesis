"""
retry_qwen_longcontext.py
--------------------------
Retries the 29 Qwen 3.5 long_context records where the model exhausted its
8192-token budget during thinking and never produced a real answer
(identified by response_text == reasoning).

These records HAVE the full paper as context — more tokens should yield a
genuine answer. This is different from no_retrieval thinking-only records,
where the model genuinely lacks information and more tokens won't help.

Results saved to: scidqa_qwen3.5_longcontext_retry.jsonl

Usage:
    python3 retry_qwen_longcontext.py
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
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

import numpy as np
from rouge_score import rouge_scorer as _rouge_scorer_module
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
LITELLM_BASE_URL = "https://litellm.uni-osnabrueck.de/v1"
LITELLM_API_KEY  = "sk-12_MUp73XhwdRhvdxJZy3w"
MODEL_NAME       = "Qwen/Qwen3.5-122B-A10B-FP8"
RATE_LIMIT       = 200          # req/min
MAX_WORKERS      = 20
MAX_TOKENS       = 16384        # increased to handle long_context thinking budget
CONTEXT_CHARS    = 140_000

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "scidqa_qwen3.5_longcontext_retry.jsonl")

VERSION_MAP = {"Initial": "initial", "Revised": "final"}

client = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_LONG_CONTEXT = (
    "You are a research assistant. "
    "Answer the question based on the research paper provided. "
    "Ground your answer in the paper's content and be as complete and accurate as possible."
)

def build_prompt_long_context(q, text):
    return f"Paper:\n\n{text[:CONTEXT_CHARS]}\n\n---\n\nQuestion: {q}"

# ── Metrics ────────────────────────────────────────────────────────────────────
_rouge = _rouge_scorer_module.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def compute_rouge(pred, ref):
    if not pred or not ref:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0, "rouge_avg": 0.0}
    s  = _rouge.score(ref, pred)
    r1, r2, rl = s["rouge1"].fmeasure, s["rouge2"].fmeasure, s["rougeL"].fmeasure
    return {"rouge_1": round(r1, 4), "rouge_2": round(r2, 4),
            "rouge_l": round(rl, 4), "rouge_avg": round((r1+r2+rl)/3, 4)}

_NO_ANS = ["i don't know", "i do not know", "cannot answer", "not mentioned",
           "not provided", "not discussed", "i cannot", "no information",
           "insufficient information", "not enough context", "cannot find",
           "does not contain", "not present in"]

def detect_no_answer(t):
    if not t:
        return True
    return any(p in t.lower() for p in _NO_ANS)

def ngram_score(resp, src, n=4):
    if not resp or not src:
        return None
    def ng(txt):
        w = txt.lower().split()
        return {tuple(w[i:i+n]) for i in range(max(0, len(w)-n+1))}
    r, s = ng(resp), ng(src)
    return round(len(r & s) / len(r), 4) if r else None

# ── Rate limiter ───────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, rpm):
        self._iv   = 60 / rpm
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self):
        with self._lock:
            w = self._iv - (time.time() - self._last)
            if w > 0:
                time.sleep(w)
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
                max_tokens=MAX_TOKENS,
            )
            latency = round(time.time() - t0, 3)
            msg     = res.choices[0].message
            content = msg.content
            reasoning = (getattr(msg, "reasoning_content", None)
                         or (msg.model_extra or {}).get("reasoning_content"))
            usage  = res.usage
            tokens = {
                "prompt_tokens":     usage.prompt_tokens     if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "total_tokens":      usage.total_tokens      if usage else None,
            }
            if content is None:
                if reasoning:
                    # Still thinking-only — record it honestly, don't overwrite with garbage
                    return {"text": None, "reasoning": reasoning,
                            "latency": latency, "tokens": tokens,
                            "error": "still thinking-only at 16384 tokens"}
                return {"text": None, "reasoning": None,
                        "latency": latency, "tokens": tokens,
                        "error": "null content from model"}
            return {"text": content.strip(), "reasoning": reasoning,
                    "latency": latency, "tokens": tokens, "error": None}
        except Exception as e:
            err = str(e)
            if "401" in err:
                print("\n[FATAL] 401 — check API key.")
                sys.exit(1)
            if "404" in err:
                print("\n[FATAL] 404 — model not found.")
                sys.exit(1)
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 5)
            else:
                return {"text": None, "reasoning": None, "latency": 0,
                        "tokens": {}, "error": f"failed after retries: {err[:120]}"}
    return {"text": None, "reasoning": None, "latency": 0, "tokens": {}, "error": "unknown"}

# ── Per-task retry ─────────────────────────────────────────────────────────────
def retry_one(task):
    row, paper_text, qid = task

    if not paper_text:
        return None  # no paper = cannot do long_context

    gold_answer       = str(row["ans"])
    paper_chars_given = min(len(paper_text), CONTEXT_CHARS)
    grounding_src     = paper_text[:CONTEXT_CHARS]

    sys_p = SYSTEM_LONG_CONTEXT
    usr_p = build_prompt_long_context(row["que"], paper_text)

    response  = call_model(sys_p, usr_p)
    rt        = response["text"] or ""
    reasoning = response.get("reasoning") or ""
    tokens    = response.get("tokens") or {}
    rouge     = compute_rouge(rt, gold_answer)
    ng        = ngram_score(rt, grounding_src)

    record = {
        "id":                    int(qid),
        "model":                 MODEL_NAME,
        "condition":             "long_context",
        "pid":                   row["pid"],
        "venue":                 row["venue"],
        "year":                  int(row["year"]),
        "version":               row["version"],
        "question":              row["que"],
        "gold_answer":           gold_answer,
        "response_text":         rt or None,
        "reasoning":             reasoning or None,
        "paper_available":       True,
        "paper_chars_given":     paper_chars_given,
        "rag_chars_given":       0,
        "chunks_used":           [],
        "response_length_chars": len(rt),
        "reasoning_length_chars":len(reasoning),
        "no_answer_signal":      detect_no_answer(rt),
        "latency_s":             response["latency"],
        "prompt_tokens":         tokens.get("prompt_tokens"),
        "completion_tokens":     tokens.get("completion_tokens"),
        "total_tokens":          tokens.get("total_tokens"),
        "rouge_1":               rouge["rouge_1"],
        "rouge_2":               rouge["rouge_2"],
        "rouge_l":               rouge["rouge_l"],
        "rouge_avg":             rouge["rouge_avg"],
        "ngram_grounding_score": ng,
        "error":                 response["error"],
        "max_tokens_used":       MAX_TOKENS,
        "is_retry":              True,
    }

    with write_lock:
        with open(OUTPUT_FILE, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    return record

# ── Find thinking-only long_context records across all Qwen result files ───────
def find_thinking_only_longcontext() -> list[dict]:
    """
    Load all Qwen result files in merge order (same as combine_scidqa.py),
    dedup by (id, condition), then return only long_context thinking-only records.
    """
    all_files = (
        sorted(glob.glob(os.path.join(SCRIPT_DIR, "archive", "qwen_4k_archive", "scidqa_qwen3.5_v*.jsonl"))) +
        [os.path.join(SCRIPT_DIR, "scidqa_qwen3.5_retries_v1v4.jsonl")] +
        sorted(glob.glob(os.path.join(SCRIPT_DIR, "scidqa_qwen3.5_v[56].jsonl")))
    )
    seen: dict[tuple, dict] = {}
    for fp in all_files:
        if not os.path.exists(fp):
            continue
        with open(fp) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    seen[(r["id"], r["condition"])] = r

    thinking_only_lc = [
        r for r in seen.values()
        if r["condition"] == "long_context"
        and r.get("response_text")
        and r.get("reasoning")
        and r["response_text"] == r["reasoning"]
    ]
    return thinking_only_lc

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    if not LITELLM_API_KEY:
        print("[ERROR] LITELLM_API_KEY env var not set.")
        sys.exit(1)

    print(f"[DEBUG] Base URL : {LITELLM_BASE_URL}")
    print(f"[DEBUG] API key  : {LITELLM_API_KEY[:12]}...")

    print(f"\n{'─'*60}")
    print(f"  Qwen 3.5 — long_context thinking-only retry")
    print(f"  max_tokens={MAX_TOKENS}  rate={RATE_LIMIT} req/min  workers={MAX_WORKERS}")
    print(f"  Output: {os.path.basename(OUTPUT_FILE)}")
    print(f"{'─'*60}\n")

    bad_records = find_thinking_only_longcontext()
    if not bad_records:
        print("No long_context thinking-only records found. Nothing to do.")
        sys.exit(0)

    print(f"Found {len(bad_records)} long_context thinking-only records to retry.\n")

    # Skip already-retried records if output file exists
    already_done: set[int] = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    already_done.add(r["id"])
        if already_done:
            print(f"  Resuming — {len(already_done)} already saved, skipping.")
            bad_records = [r for r in bad_records if r["id"] not in already_done]
            if not bad_records:
                print("  All records already retried.")
                sys.exit(0)

    print("Loading dataset and paper full-texts...")
    df = pd.read_excel(os.path.join(DATA_DIR, "SciDQADataset.xlsx"))
    df = df.set_index("id")
    with open(os.path.join(DATA_DIR, "papers_fulltext_nougat.pkl"), "rb") as f:
        fulltext = pickle.load(f)

    tasks = []
    for r in bad_records:
        qid = r["id"]
        try:
            row = df.loc[qid]
        except KeyError:
            print(f"  Warning: id={qid} not in dataset, skipping.")
            continue
        vkey  = VERSION_MAP.get(row["version"], "initial")
        ptext = fulltext.get(vkey, {}).get(row["pid"], "")
        if not ptext:
            print(f"  Warning: no paper text for id={qid} (pid={row['pid']}), skipping.")
            continue
        tasks.append((row, ptext, qid))

    print(f"Retrying {len(tasks)} records at {RATE_LIMIT} req/min, max_tokens={MAX_TOKENS}...\n")

    success = failed = still_thinking = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(retry_one, t): t for t in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                           desc="long_context retry"):
            try:
                rec = future.result()
                if rec is None:
                    failed += 1
                elif rec.get("error") and "still thinking-only" in str(rec.get("error", "")):
                    still_thinking += 1
                elif rec.get("error"):
                    failed += 1
                else:
                    success += 1
            except Exception as e:
                print(f"\n  Worker error: {e}")
                failed += 1

    print(f"\n{'─'*60}")
    print(f"  Done.")
    print(f"  Succeeded       : {success}")
    print(f"  Still thinking  : {still_thinking}  (even 16384 tokens not enough)")
    print(f"  Failed (errors) : {failed}")
    print(f"  Results saved to: {os.path.basename(OUTPUT_FILE)}")
    print(f"{'─'*60}\n")
    print("To include in combined analysis, add this file to MODEL_FILES in combine_scidqa.py")
    print("It will automatically overwrite the thinking-only records for these 29 IDs.")

if __name__ == "__main__":
    main()
