from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time
import threading
import concurrent.futures
from typing import Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

import numpy as np

try:
    from rouge_score import rouge_scorer as _rouge_scorer_module
except ImportError:
    sys.exit("Missing dependency: pip3 install rouge-score")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    sys.exit("Missing dependency: pip3 install rank-bm25")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Missing dependency: pip3 install sentence-transformers")


LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
LITELLM_API_KEY  = os.getenv("LITELLM_API_KEY", "")
MODEL_NAME       = "openai/gpt-oss-120b"

client = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)

MAX_WORKERS        = 40
RATE_LIMIT         = 200         # requests per minute
CONTEXT_CHARS      = 140_000     # long_context: max paper chars sent (~35 k tokens)
SENTENCES_PER_CHUNK = 10         # SciDQA paper Algorithm 1
CHUNK_OVERLAP       = 1
DENSE_EMBED_MODEL  = "all-MiniLM-L6-v2"

CONDITIONS = ["no_retrieval", "rag_top3", "rag_top5", "rag_dense", "long_context"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
VERSION_MAP = {"Initial": "initial", "Revised": "final"}

parser = argparse.ArgumentParser()
parser.add_argument("--offset", type=int, default=0,   help="Row offset into dataset")
parser.add_argument("--n",      type=int, default=500, help="Number of questions to process")
args = parser.parse_args()

OFFSET      = args.offset
N_QUESTIONS = args.n

_BASE = "scidqa_gptoss"

def _next_version(base: str) -> str:
    existing = glob.glob(os.path.join(SCRIPT_DIR, f"{base}_v*.jsonl"))
    nums = []
    for f in existing:
        m = re.search(rf"{re.escape(base)}_v(\d+)\.jsonl$", os.path.basename(f))
        if m:
            nums.append(int(m.group(1)))
    return f"{base}_v{max(nums) + 1}" if nums else f"{base}_v1"

_RUN_NAME    = _next_version(_BASE)
OUTPUT_FILE  = os.path.join(SCRIPT_DIR, f"{_RUN_NAME}.jsonl")
SUMMARY_FILE = os.path.join(SCRIPT_DIR, f"{_RUN_NAME}_summary.txt")

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

def build_prompt_no_retrieval(question: str) -> str:
    return f"Question: {question}"

def build_prompt_rag(question: str, chunks: list[str]) -> str:
    chunk_block = "\n\n---\n\n".join(
        f"[Excerpt {i + 1}]\n{c}" for i, c in enumerate(chunks)
    )
    return (
        f"Retrieved Paper Excerpts:\n\n{chunk_block}\n\n"
        f"---\n\n"
        f"Question: {question}"
    )

def build_prompt_long_context(question: str, paper_text: str) -> str:
    return (
        f"Paper:\n\n{paper_text[:CONTEXT_CHARS]}\n\n"
        f"---\n\n"
        f"Question: {question}"
    )

def chunk_text(text: str,
               sentences_per_chunk: int = SENTENCES_PER_CHUNK,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks: list[str] = []
    stride = max(1, sentences_per_chunk - overlap)
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        sents = sent_tokenize(para)
        if not sents:
            continue
        for i in range(0, len(sents), stride):
            window = sents[i : i + sentences_per_chunk]
            chunk = " ".join(window).strip()
            if chunk:
                chunks.append(chunk)
    return chunks

def retrieve_chunks(question: str, paper_text: str, top_k: int = 3) -> tuple[list[str], list[int]]:
    chunks = chunk_text(paper_text)
    if not chunks:
        return [], []
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(question.lower().split())
    top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    top_idx_ordered = sorted(top_idx)
    return [chunks[i] for i in top_idx_ordered], top_idx_ordered

_embed_model: Optional[SentenceTransformer] = None
_embed_lock = threading.Lock()

def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(DENSE_EMBED_MODEL)
    return _embed_model

def retrieve_chunks_dense(question: str, paper_text: str, top_k: int = 3) -> tuple[list[str], list[int]]:
    chunks = chunk_text(paper_text)
    if not chunks:
        return [], []
    model = get_embed_model()
    with _embed_lock:
        chunk_embs = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        query_emb  = model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = np.dot(chunk_embs, query_emb)
    top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    top_idx_ordered = sorted(top_idx)
    return [chunks[i] for i in top_idx_ordered], top_idx_ordered

_rouge = _rouge_scorer_module.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def compute_rouge(prediction: str, reference: str) -> dict:
    if not prediction or not reference:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0, "rouge_avg": 0.0}
    scores = _rouge.score(reference, prediction)
    r1 = scores["rouge1"].fmeasure
    r2 = scores["rouge2"].fmeasure
    rl = scores["rougeL"].fmeasure
    return {
        "rouge_1"  : round(r1, 4),
        "rouge_2"  : round(r2, 4),
        "rouge_l"  : round(rl, 4),
        "rouge_avg": round((r1 + r2 + rl) / 3, 4),
    }

def ngram_grounding_score(response: str, source: str, n: int = 4) -> Optional[float]:
    if not response or not source:
        return None
    def get_ngrams(text: str) -> set:
        words = text.lower().split()
        return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}
    resp_ng = get_ngrams(response)
    if not resp_ng:
        return None
    src_ng = get_ngrams(source)
    return round(len(resp_ng & src_ng) / len(resp_ng), 4)

_NO_ANSWER_PHRASES = [
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "cannot answer", "can't answer", "unable to answer",
    "not mentioned", "not provided", "not discussed", "not stated",
    "i cannot", "no information", "insufficient information",
    "not enough context", "cannot find", "not available in",
    "does not contain", "not present in", "no relevant",
]

def detect_no_answer(text: str) -> bool:
    if not text:
        return True
    tl = text.lower()
    return any(p in tl for p in _NO_ANSWER_PHRASES)

class RateLimiter:
    def __init__(self, max_per_minute: int):
        self._interval  = 60.0 / max_per_minute
        self._lock      = threading.Lock()
        self._last_call = 0.0

    def acquire(self):
        with self._lock:
            now  = time.time()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()

rate_limiter = RateLimiter(RATE_LIMIT)
write_lock   = threading.Lock()

def call_model(system_prompt: str, user_prompt: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        rate_limiter.acquire()
        try:
            t0       = time.time()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0,
                max_tokens=8192,
            )
            latency = round(time.time() - t0, 3)
            msg     = response.choices[0].message
            content = msg.content
            reasoning = (
                getattr(msg, "reasoning_content", None)
                or (msg.model_extra or {}).get("reasoning_content")
            )
            usage  = response.usage
            tokens = {
                "prompt_tokens"    : usage.prompt_tokens     if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "total_tokens"     : usage.total_tokens      if usage else None,
            }
            if content is None:
                return {"text": None, "reasoning": reasoning, "latency": latency,
                        "tokens": tokens, "error": "null content from model"}
            return {"text": content.strip(), "reasoning": reasoning,
                    "latency": latency, "tokens": tokens, "error": None}

        except Exception as e:
            err_str = str(e)
            # Critical errors — stop immediately, do not retry
            if "401" in err_str:
                print(f"\n[FATAL] Unauthorised (401) — check API key. Stopping.")
                sys.exit(1)
            if "404" in err_str:
                print(f"\n[FATAL] Model not found (404) — check model name. Stopping.")
                sys.exit(1)
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 5)
            else:
                # Non-critical repeated failure — log but continue
                return {"text": None, "reasoning": None, "latency": 0,
                        "tokens": {}, "error": f"failed after {retries} attempts: {err_str[:120]}"}

    return {"text": None, "reasoning": None, "latency": 0, "tokens": {}, "error": "unknown"}

def evaluate_question(args: tuple) -> list[dict]:
    i, row, paper_text = args
    records: list[dict] = []
    paper_available = bool(paper_text)
    gold_answer = str(row["ans"])

    for condition in CONDITIONS:
        grounding_src: Optional[str] = None
        chunks_used: list[int] = []
        rag_chars_given = 0
        paper_chars_given = 0
        retrieved_chunks: list[str] = []

        if condition == "no_retrieval":
            sys_prompt  = SYSTEM_NO_RETRIEVAL
            user_prompt = build_prompt_no_retrieval(row["que"])

        elif condition in ("rag_top3", "rag_top5", "rag_dense"):
            if not paper_text:
                records.append({
                    "id": int(row["id"]), "condition": condition,
                    "paper_available": False,
                    "error": "no paper text available", "response_text": None,
                })
                continue
            if condition == "rag_dense":
                retrieved_chunks, chunks_used = retrieve_chunks_dense(row["que"], paper_text, top_k=3)
            else:
                top_k = 3 if condition == "rag_top3" else 5
                retrieved_chunks, chunks_used = retrieve_chunks(row["que"], paper_text, top_k=top_k)
            sys_prompt      = SYSTEM_RAG
            user_prompt     = build_prompt_rag(row["que"], retrieved_chunks)
            rag_chars_given = sum(len(c) for c in retrieved_chunks)
            grounding_src   = "\n\n".join(retrieved_chunks)

        else:  # long_context
            if not paper_text:
                records.append({
                    "id": int(row["id"]), "condition": condition,
                    "paper_available": False,
                    "error": "no paper text available", "response_text": None,
                })
                continue
            sys_prompt        = SYSTEM_LONG_CONTEXT
            user_prompt       = build_prompt_long_context(row["que"], paper_text)
            paper_chars_given = min(len(paper_text), CONTEXT_CHARS)
            grounding_src     = paper_text[:CONTEXT_CHARS]

        response = call_model(sys_prompt, user_prompt)

        response_text = response["text"] or ""
        reasoning     = response.get("reasoning") or ""
        tokens        = response.get("tokens") or {}

        rouge         = compute_rouge(response_text, gold_answer)
        ngram_score   = ngram_grounding_score(response_text, grounding_src) if grounding_src else None
        no_ans_signal = detect_no_answer(response_text)

        record = {
            "id"                    : int(row["id"]),
            "model"                 : MODEL_NAME,
            "condition"             : condition,
            "pid"                   : row["pid"],
            "venue"                 : row["venue"],
            "year"                  : int(row["year"]),
            "version"               : row["version"],
            "question"              : row["que"],
            "gold_answer"           : gold_answer,
            "response_text"         : response_text or None,
            "reasoning"             : reasoning or None,
            "paper_available"       : paper_available,
            "paper_chars_given"     : paper_chars_given,
            "rag_chars_given"       : rag_chars_given,
            "chunks_used"           : chunks_used,
            "response_length_chars" : len(response_text),
            "reasoning_length_chars": len(reasoning),
            "no_answer_signal"      : no_ans_signal,
            "latency_s"             : response["latency"],
            "prompt_tokens"         : tokens.get("prompt_tokens"),
            "completion_tokens"     : tokens.get("completion_tokens"),
            "total_tokens"          : tokens.get("total_tokens"),
            "rouge_1"               : rouge["rouge_1"],
            "rouge_2"               : rouge["rouge_2"],
            "rouge_l"               : rouge["rouge_l"],
            "rouge_avg"             : rouge["rouge_avg"],
            "ngram_grounding_score" : ngram_score,
            "error"                 : response["error"],
        }
        records.append(record)

        with write_lock:
            with open(OUTPUT_FILE, "a") as fh:
                fh.write(json.dumps(record) + "\n")

    return records

def load_data(offset: int, n: int) -> list[tuple]:
    df = pd.read_excel(os.path.join(DATA_DIR, "SciDQADataset.xlsx"))
    df = df.iloc[offset : offset + n].reset_index(drop=True)

    print("  Loading paper full texts...")
    with open(os.path.join(DATA_DIR, "papers_fulltext_nougat.pkl"), "rb") as f:
        fulltext = pickle.load(f)

    items = []
    missing = 0
    for _, row in df.iterrows():
        vkey  = VERSION_MAP.get(row["version"], "initial")
        ptext = fulltext.get(vkey, {}).get(row["pid"], "")
        if not ptext:
            missing += 1
        items.append((int(row["id"]), row, ptext))

    if missing:
        print(f"  Warning: {missing}/{len(items)} questions have no paper text")
    return items

def _avg(vals: list) -> float:
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else 0.0

def build_summary(all_records: list[dict]) -> str:
    lines = [
        "=" * 68,
        "SCIDQA EVALUATION SUMMARY — GPT-OSS",
        "=" * 68,
        f"Model      : {MODEL_NAME}",
        f"Questions  : {N_QUESTIONS}  (offset {OFFSET})",
        f"Conditions : {CONDITIONS}",
        f"Rate limit : {RATE_LIMIT} req/min  |  Workers: {MAX_WORKERS}",
        "",
    ]

    cond_stats: dict[str, dict] = {}

    for cond in CONDITIONS:
        recs     = [r for r in all_records if r.get("condition") == cond]
        errors   = [r for r in recs if r.get("error")]
        answered = [r for r in recs if not r.get("error") and r.get("response_text")]
        n_ans    = len(answered)
        n_tot    = len(recs)

        avg_r1   = _avg([r.get("rouge_1")   for r in answered])
        avg_r2   = _avg([r.get("rouge_2")   for r in answered])
        avg_rl   = _avg([r.get("rouge_l")   for r in answered])
        avg_ravg = _avg([r.get("rouge_avg") for r in answered])
        avg_lat  = _avg([r.get("latency_s") for r in answered])
        avg_ptok = _avg([r.get("prompt_tokens") for r in answered])
        avg_ctok = _avg([r.get("completion_tokens") for r in answered])
        no_ans_n = sum(1 for r in answered if r.get("no_answer_signal"))

        cond_stats[cond] = {"avg_ravg": avg_ravg}

        if cond in ("rag_top3", "rag_top5", "rag_dense"):
            avg_ctx_chars = _avg([r.get("rag_chars_given") for r in answered])
            avg_ng = _avg([r.get("ngram_grounding_score") for r in answered])
            ctx_label = "Avg RAG chars sent"
        elif cond == "long_context":
            avg_ctx_chars = _avg([r.get("paper_chars_given") for r in answered])
            avg_ng = _avg([r.get("ngram_grounding_score") for r in answered])
            ctx_label = "Avg paper chars sent"
        else:
            avg_ctx_chars = 0
            avg_ng = None
            ctx_label = ""

        lines += [f"── Condition: {cond} ──",
                  f"  Answered          : {n_ans}/{n_tot}",
                  f"  Errors            : {len(errors)}",
                  f"  No-answer signals : {no_ans_n}/{n_ans}"]
        if avg_ctx_chars:
            lines.append(f"  {ctx_label:<22}: {avg_ctx_chars:,.0f}")
        lines += [
            f"  Avg ROUGE-1       : {avg_r1:.4f}",
            f"  Avg ROUGE-2       : {avg_r2:.4f}",
            f"  Avg ROUGE-L       : {avg_rl:.4f}",
            f"  Avg ROUGE-avg     : {avg_ravg:.4f}",
        ]
        if avg_ng is not None:
            lines.append(f"  Avg n-gram grounding: {avg_ng:.4f}")
        lines += [
            f"  Avg latency       : {avg_lat:.1f}s",
            f"  Avg prompt tokens : {avg_ptok:.0f}",
            f"  Avg completion tok: {avg_ctok:.0f}",
            "",
        ]

    compare_conds = [c for c in ["rag_top3", "rag_top5", "rag_dense", "long_context"]
                     if c in cond_stats]
    if "no_retrieval" in cond_stats and compare_conds:
        lines += ["── Cross-condition ROUGE-avg Δ (vs no_retrieval) ──"]
        base = cond_stats["no_retrieval"]["avg_ravg"]
        for cond in compare_conds:
            delta = cond_stats[cond]["avg_ravg"] - base
            sign  = "+" if delta >= 0 else ""
            lines.append(
                f"  {cond:<15} ROUGE-avg: {cond_stats[cond]['avg_ravg']:.4f}  "
                f"(Δ {sign}{delta:.4f} vs no_retrieval)"
            )
        lines.append("")

    lines += [f"Output : {OUTPUT_FILE}", f"Summary: {SUMMARY_FILE}"]
    return "\n".join(lines)

def run_evaluation():
    print("=" * 68)
    print("SciDQA Evaluation  —  GPT-OSS 120B")
    print(f"Model      : {MODEL_NAME}")
    print(f"Questions  : {N_QUESTIONS}  (offset {OFFSET})")
    print(f"Conditions : {CONDITIONS}")
    print(f"Rate limit : {RATE_LIMIT} req/min  |  Workers: {MAX_WORKERS}")
    print(f"Output     : {os.path.basename(OUTPUT_FILE)}")
    print("=" * 68)

    print("\nLoading data...")
    items = load_data(OFFSET, N_QUESTIONS)
    if not items:
        print("No questions to process. Exiting.")
        sys.exit(0)

    total_calls = len(items) * len(CONDITIONS)
    est_min = total_calls / RATE_LIMIT
    print(f"  {len(items)} questions × {len(CONDITIONS)} conditions = {total_calls} API calls")
    print(f"  Estimated min time at {RATE_LIMIT} req/min: ~{est_min:.0f} min\n")

    all_records: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(evaluate_question, item): item for item in items}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures), desc="Questions"):
            try:
                all_records.extend(future.result())
            except SystemExit:
                raise
            except Exception as e:
                print(f"\n  Worker error: {e}")

    summary_text = build_summary(all_records)
    print("\n" + summary_text)
    with open(SUMMARY_FILE, "w") as sf:
        sf.write(summary_text + "\n")

    print(f"\nDone. Results → {os.path.basename(OUTPUT_FILE)}")


if __name__ == "__main__":
    run_evaluation()
