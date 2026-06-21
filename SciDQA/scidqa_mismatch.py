from __future__ import annotations

import argparse
import json
import os
import pickle
import random
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


LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
LITELLM_API_KEY  = os.getenv("LITELLM_API_KEY")

MODEL_REGISTRY = {
    "gemma4"  : {"name": "RedHatAI/gemma-4-31B-it-FP8-Dynamic",  "rate": 300, "workers": 40, "max_tokens": 8192},
    "gptoss"  : {"name": "openai/gpt-oss-120b",                   "rate": 200, "workers": 30, "max_tokens": 8192},
    "qwen3.5" : {"name": "Qwen/Qwen3.5-122B-A10B-FP8",           "rate": 200, "workers": 20, "max_tokens": 16384},
}

parser = argparse.ArgumentParser(description="SciDQA mismatch (hallucination) evaluation")
parser.add_argument("--model",  required=True, choices=list(MODEL_REGISTRY),
                    help="Which model to evaluate (gemma4 | gptoss | qwen3.5)")
parser.add_argument("--offset", type=int, default=0,    help="Row offset into dataset")
parser.add_argument("--n",      type=int, default=2937, help="Number of questions to process")
args = parser.parse_args()

MODEL_KEY   = args.model
MODEL_CFG   = MODEL_REGISTRY[MODEL_KEY]
MODEL_NAME  = MODEL_CFG["name"]
RATE_LIMIT  = MODEL_CFG["rate"]
MAX_WORKERS = MODEL_CFG["workers"]
MAX_TOKENS  = MODEL_CFG["max_tokens"]
OFFSET      = args.offset
N_QUESTIONS = args.n

CONTEXT_CHARS       = 140_000
SENTENCES_PER_CHUNK = 10
CHUNK_OVERLAP       = 1
TOP_K               = 3         # same as rag_top3 for a fair comparison

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
VERSION_MAP = {"Initial": "initial", "Revised": "final"}

OUTPUT_FILE  = os.path.join(SCRIPT_DIR,
    f"scidqa_mismatch_{MODEL_KEY}_offset{OFFSET}_n{N_QUESTIONS}.jsonl")
SUMMARY_FILE = os.path.join(SCRIPT_DIR,
    f"scidqa_mismatch_{MODEL_KEY}_offset{OFFSET}_n{N_QUESTIONS}_summary.txt")

client = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)

SYSTEM_RAG = (
    "You are a research assistant. "
    "You are given excerpts retrieved from a research paper. "
    "Answer the question as accurately as possible. "
    "After your answer, on a new line write exactly one of the following lines "
    "(choose the one that best describes where your answer came from):\n"
    "SOURCE: EXCERPT\n"
    "SOURCE: TRAINING MEMORY\n"
    "SOURCE: NOT FOUND\n\n"
    "Use SOURCE: EXCERPT if you based your answer on the provided excerpts.\n"
    "Use SOURCE: TRAINING MEMORY if you answered from your own knowledge because "
    "the excerpts did not contain the relevant information.\n"
    "Use SOURCE: NOT FOUND if neither the excerpts nor your knowledge contain "
    "a clear answer."
)

def build_prompt_rag(question: str, chunks: list[str]) -> str:
    chunk_block = "\n\n---\n\n".join(
        f"[Excerpt {i + 1}]\n{c}" for i, c in enumerate(chunks)
    )
    return (
        f"Retrieved Paper Excerpts:\n\n{chunk_block}\n\n"
        f"---\n\n"
        f"Question: {question}"
    )

def parse_source_attribution(response_text: str) -> str:
    """
    Extracts the SOURCE: label from the model's response.
    Returns one of: 'EXCERPT', 'TRAINING MEMORY', 'NOT FOUND', 'UNKNOWN'.
    """
    if not response_text:
        return "UNKNOWN"
    for line in reversed(response_text.splitlines()):
        line = line.strip().upper()
        if line.startswith("SOURCE:"):
            label = line.replace("SOURCE:", "").strip()
            if "EXCERPT" in label:
                return "EXCERPT"
            if "TRAINING" in label or "MEMORY" in label:
                return "TRAINING MEMORY"
            if "NOT FOUND" in label or "NOT_FOUND" in label:
                return "NOT FOUND"
    return "UNKNOWN"

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
            chunk  = " ".join(window).strip()
            if chunk:
                chunks.append(chunk)
    return chunks

def retrieve_chunks(question: str, paper_text: str, top_k: int = TOP_K) -> list[str]:
    chunks = chunk_text(paper_text)
    if not chunks:
        return []
    tokenized = [c.lower().split() for c in chunks]
    bm25      = BM25Okapi(tokenized)
    scores    = bm25.get_scores(question.lower().split())
    top_idx   = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [chunks[i] for i in sorted(top_idx)]

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
    return any(p in text.lower() for p in _NO_ANSWER_PHRASES)

class RateLimiter:
    def __init__(self, max_per_minute: int):
        self._interval  = 60.0 / max_per_minute
        self._lock      = threading.Lock()
        self._last_call = 0.0

    def acquire(self):
        with self._lock:
            wait = self._interval - (time.time() - self._last_call)
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
                max_tokens=MAX_TOKENS,
            )
            latency   = round(time.time() - t0, 3)
            msg       = response.choices[0].message
            content   = msg.content
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
                if MODEL_KEY == "qwen3.5" and reasoning:
                    return {"text": reasoning.strip(), "reasoning": reasoning,
                            "latency": latency, "tokens": tokens, "error": "thinking-only"}
                return {"text": None, "reasoning": reasoning, "latency": latency,
                        "tokens": tokens, "error": "null content from model"}
            return {"text": content.strip(), "reasoning": reasoning,
                    "latency": latency, "tokens": tokens, "error": None}

        except Exception as e:
            err = str(e)
            if "401" in err:
                print("\n[FATAL] Unauthorised (401) — check API key. Stopping.")
                sys.exit(1)
            if "404" in err:
                print("\n[FATAL] Model not found (404). Stopping.")
                sys.exit(1)
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 5)
            else:
                return {"text": None, "reasoning": None, "latency": 0,
                        "tokens": {}, "error": f"failed after {retries} attempts: {err[:120]}"}

    return {"text": None, "reasoning": None, "latency": 0, "tokens": {}, "error": "unknown"}

def pick_wrong_paper(correct_pid: str,
                     all_pids: list[str],
                     fulltext: dict,
                     question_id: int) -> tuple[str, str]:
    """
    Returns (wrong_pid, wrong_paper_text).

    Selection is deterministic per question_id (reproducible) and guarantees:
      - different pid from the correct paper
      - the wrong paper has fulltext available
    """
    rng = random.Random(question_id)
    candidates = [p for p in all_pids if p != correct_pid]
    rng.shuffle(candidates)
    for pid in candidates:
        # Try both version keys
        text = fulltext.get("initial", {}).get(pid, "") or \
               fulltext.get("final",   {}).get(pid, "")
        if text:
            return pid, text
    return "", ""

def evaluate_question(task: tuple) -> Optional[dict]:
    qid, row, correct_text, wrong_pid, wrong_text, pid_meta = task

    gold_answer  = str(row["ans"])
    wrong_venue  = pid_meta.get(wrong_pid, {}).get("venue", "")
    wrong_year   = pid_meta.get(wrong_pid, {}).get("year", "")

    if not wrong_text:
        record = {
            "id": int(qid), "model": MODEL_NAME, "condition": "rag_mismatch",
            "error": "no wrong paper text available", "response_text": None,
        }
        with write_lock:
            with open(OUTPUT_FILE, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        return record

    retrieved_chunks = retrieve_chunks(row["que"], wrong_text)
    rag_chars_given  = sum(len(c) for c in retrieved_chunks)
    grounding_src    = "\n\n".join(retrieved_chunks)

    response      = call_model(SYSTEM_RAG, build_prompt_rag(row["que"], retrieved_chunks))
    response_text = response["text"] or ""
    reasoning     = response.get("reasoning") or ""
    tokens        = response.get("tokens") or {}

    rouge               = compute_rouge(response_text, gold_answer)
    ngram_score         = ngram_grounding_score(response_text, grounding_src) if grounding_src else None
    no_ans              = detect_no_answer(response_text)
    source_attribution  = parse_source_attribution(response_text)

    record = {
        "id"                    : int(qid),
        "model"                 : MODEL_NAME,
        "condition"             : "rag_mismatch",
        "pid"                   : row["pid"],
        "venue"                 : row["venue"],
        "year"                  : int(row["year"]),
        "version"               : row["version"],
        "question"              : row["que"],
        "gold_answer"           : gold_answer,
        "mismatch_pid"          : wrong_pid,
        "mismatch_venue"        : wrong_venue,
        "mismatch_year"         : wrong_year,
        "response_text"         : response_text or None,
        "reasoning"             : reasoning or None,
        "paper_available"       : bool(correct_text),
        "rag_chars_given"       : rag_chars_given,
        "chunks_used"           : TOP_K,
        "response_length_chars" : len(response_text),
        "reasoning_length_chars": len(reasoning),
        "no_answer_signal"      : no_ans,
        "source_attribution"    : source_attribution,
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

    with write_lock:
        with open(OUTPUT_FILE, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    return record

def load_data() -> tuple[list[tuple], dict]:
    df = pd.read_excel(os.path.join(DATA_DIR, "SciDQADataset.xlsx"))
    df_slice = df.iloc[OFFSET : OFFSET + N_QUESTIONS].reset_index(drop=True)

    print("  Loading paper full texts...")
    with open(os.path.join(DATA_DIR, "papers_fulltext_nougat.pkl"), "rb") as f:
        fulltext = pickle.load(f)

    # Build list of all available pids + a pid→meta lookup for wrong-paper fields
    all_pids: set[str] = set()
    for version_dict in fulltext.values():
        all_pids.update(version_dict.keys())
    all_pids_list = sorted(all_pids)

    pid_meta: dict[str, dict] = {}
    for _, row in df.iterrows():
        pid_meta[row["pid"]] = {"venue": row["venue"], "year": int(row["year"])}

    # Resume logic: skip successfully answered records, retry errors
    already_done: set[int] = set()
    if os.path.exists(OUTPUT_FILE):
        good_records: list[str] = []
        error_ids:    set[int]  = set()
        with open(OUTPUT_FILE) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("error"):
                    error_ids.add(r["id"])
                else:
                    already_done.add(r["id"])
                    good_records.append(line)

        # Rewrite file keeping only clean records — errors will be re-run and appended
        if error_ids:
            with open(OUTPUT_FILE, "w") as fh:
                for line in good_records:
                    fh.write(line + "\n")
            print(f"  Resuming — {len(already_done)} clean, {len(error_ids)} errors will be retried.")
        elif already_done:
            print(f"  Resuming — {len(already_done)} already saved, skipping.")

    tasks: list[tuple] = []
    missing_correct = 0
    for _, row in df_slice.iterrows():
        qid = int(row["id"])
        if qid in already_done:
            continue
        vkey         = VERSION_MAP.get(row["version"], "initial")
        correct_text = fulltext.get(vkey, {}).get(row["pid"], "")
        if not correct_text:
            missing_correct += 1

        wrong_pid, wrong_text = pick_wrong_paper(
            row["pid"], all_pids_list, fulltext, qid
        )
        tasks.append((qid, row, correct_text, wrong_pid, wrong_text, pid_meta))

    if missing_correct:
        print(f"  Warning: {missing_correct} questions have no correct-paper text (wrong paper still used)")

    return tasks, pid_meta

def _avg(vals: list) -> float:
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else 0.0

def build_summary(records: list[dict]) -> str:
    answered = [r for r in records if not r.get("error") and r.get("response_text")]
    errors   = [r for r in records if r.get("error")]
    no_ans   = [r for r in answered if r.get("no_answer_signal")]

    from collections import Counter
    src_counts = Counter(r.get("source_attribution", "UNKNOWN") for r in answered)
    n_ans_total = max(len(answered), 1)

    lines = [
        "=" * 68,
        "SCIDQA MISMATCH EVALUATION SUMMARY",
        "=" * 68,
        f"Model        : {MODEL_NAME}",
        f"Questions    : {N_QUESTIONS}  (offset {OFFSET})",
        f"Condition    : rag_mismatch (wrong paper's BM25 top-{TOP_K} chunks)",
        f"Rate limit   : {RATE_LIMIT} req/min  |  Workers: {MAX_WORKERS}",
        "",
        f"Total records : {len(records)}",
        f"Answered      : {len(answered)}",
        f"Errors        : {len(errors)}",
        f"No-answer     : {len(no_ans)} ({100*len(no_ans)/n_ans_total:.1f}%)",
        "",
        "── Source Attribution Breakdown ──────────────────────────────────",
        f"  EXCERPT        : {src_counts.get('EXCERPT', 0):4d}  ({100*src_counts.get('EXCERPT',0)/n_ans_total:.1f}%)  ← used wrong paper (hallucination risk)",
        f"  TRAINING MEMORY: {src_counts.get('TRAINING MEMORY', 0):4d}  ({100*src_counts.get('TRAINING MEMORY',0)/n_ans_total:.1f}%)  ← ignored context, used memory",
        f"  NOT FOUND      : {src_counts.get('NOT FOUND', 0):4d}  ({100*src_counts.get('NOT FOUND',0)/n_ans_total:.1f}%)  ← faithful refusal (best behaviour)",
        f"  UNKNOWN        : {src_counts.get('UNKNOWN', 0):4d}  ({100*src_counts.get('UNKNOWN',0)/n_ans_total:.1f}%)  ← did not follow format",
        "",
        "── ROUGE Scores ──────────────────────────────────────────────────",
        f"  Avg ROUGE-1   : {_avg([r.get('rouge_1')   for r in answered]):.4f}",
        f"  Avg ROUGE-2   : {_avg([r.get('rouge_2')   for r in answered]):.4f}",
        f"  Avg ROUGE-L   : {_avg([r.get('rouge_l')   for r in answered]):.4f}",
        f"  Avg ROUGE-avg : {_avg([r.get('rouge_avg') for r in answered]):.4f}",
        f"  Avg n-gram grounding: {_avg([r.get('ngram_grounding_score') for r in answered]):.4f}",
        f"  Avg latency   : {_avg([r.get('latency_s') for r in answered]):.1f}s",
        f"  Avg RAG chars : {_avg([r.get('rag_chars_given') for r in answered]):,.0f}",
        "",
        "── ROUGE by Source ───────────────────────────────────────────────",
    ]
    for src_label in ["EXCERPT", "TRAINING MEMORY", "NOT FOUND"]:
        src_recs = [r for r in answered if r.get("source_attribution") == src_label]
        if src_recs:
            lines.append(
                f"  {src_label:<16}: n={len(src_recs):4d}  "
                f"ROUGE-avg={_avg([r.get('rouge_avg') for r in src_recs]):.4f}"
            )
    lines += [
        "",
        "── Interpretation guide ──────────────────────────────────────────",
        "  Compare Avg ROUGE-avg here vs rag_top3 in main evaluation:",
        "  Large drop  → model used the context  (RAG is grounding the model)",
        "  Small drop  → model used training memory (context is being ignored)",
        "  No-answer ↑ → model is faithfully refusing on irrelevant chunks",
        "",
        f"Output : {OUTPUT_FILE}",
        f"Summary: {SUMMARY_FILE}",
    ]
    return "\n".join(lines)

def main():
    print("=" * 68)
    print("SciDQA Mismatch Evaluation — Hallucination Test")
    print(f"Model      : {MODEL_NAME}")
    print(f"Condition  : rag_mismatch (wrong paper BM25 top-{TOP_K})")
    print(f"Questions  : {N_QUESTIONS}  (offset {OFFSET})")
    print(f"Rate limit : {RATE_LIMIT} req/min  |  Workers: {MAX_WORKERS}")
    print(f"Output     : {os.path.basename(OUTPUT_FILE)}")
    print("=" * 68)

    tasks, _ = load_data()
    if not tasks:
        print("No questions to process (all already done or none loaded). Exiting.")
        sys.exit(0)

    est_min = len(tasks) / RATE_LIMIT
    print(f"\n  {len(tasks)} questions × 1 condition = {len(tasks)} API calls")
    print(f"  Estimated min time at {RATE_LIMIT} req/min: ~{est_min:.0f} min\n")

    all_records: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(evaluate_question, t): t for t in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures), desc="Mismatch"):
            try:
                rec = future.result()
                if rec:
                    all_records.append(rec)
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
    main()
