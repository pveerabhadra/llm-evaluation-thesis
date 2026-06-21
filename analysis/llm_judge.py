from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import threading
import concurrent.futures
import statistics
from collections import defaultdict
from typing import Optional

from openai import OpenAI
from tqdm import tqdm


LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
LITELLM_API_KEY  = os.getenv("LITELLM_API_KEY")

MODELS = {
    "gemma4"  : {"name": "RedHatAI/gemma-4-31B-it-FP8-Dynamic", "rate": 180, "workers": 30, "max_tokens": 1024},
    "gptoss"  : {"name": "openai/gpt-oss-120b",                  "rate": 180, "workers": 25, "max_tokens": 1024},
    # Qwen 3.5: thinking mode disabled via /no_think prefix; 8192 is a safety net for long answers
    "qwen3.5" : {"name": "Qwen/Qwen3.5-122B-A10B-FP8",          "rate": 180, "workers": 15, "max_tokens": 8192},
}

# Cross-reference map: answer_model_key → [judge_model_keys]
JUDGE_MAP = {
    "gemma4"  : ["gptoss", "qwen3.5"],
    "gptoss"  : ["gemma4", "qwen3.5"],
    "qwen3.5" : ["gemma4", "gptoss"],
}

# Combined file label → answer_model_key
LABEL_TO_KEY = {
    "gemma-4-31B"  : "gemma4",
    "gpt-oss-120b" : "gptoss",
    "Qwen3.5-122B" : "qwen3.5",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COMBINED_FILES = {
    "gemma-4-31B"  : os.path.join(SCRIPT_DIR, "scidqa_gemma4_combined.jsonl"),
    "gpt-oss-120b" : os.path.join(SCRIPT_DIR, "scidqa_gptoss_combined.jsonl"),
    "Qwen3.5-122B" : os.path.join(SCRIPT_DIR, "scidqa_qwen3.5_combined.jsonl"),
}

ALL_CONDITIONS = ["no_retrieval", "rag_top3", "rag_top5", "rag_dense", "long_context"]

client = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)

# The paper uses a single unified prompt (not split into system/user).
# We use an empty system message and place the full prompt in the user turn
# to stay as close as possible to the original.
SYSTEM_JUDGE = ""   # paper uses a single-message prompt

def build_judge_prompt(question: str, model_answer: str, gold_answer: str) -> str:
    return (
        "You are an expert evaluator tasked with assessing the quality of a "
        "model-generated answer compared to a gold standard correct answer in a "
        "long-form question-answering context. "
        "Your goal is to provide a quantified evaluation across multiple dimensions. "
        "Please follow these steps: "
        "Carefully read the original question, the model-generated answer, and the gold correct answer. "
        "Evaluate the model-generated answer on the following dimensions, "
        "providing a score from 1-10 for each (where 1 is poor and 10 is excellent): "
        "a) Relevance (1-10): How well does the answer address the specific question asked? "
        "b) Accuracy (1-10): To what extent is the information provided correct and aligned with the gold answer? "
        "c) Completeness (1-10): How thoroughly does the answer cover all aspects of the question compared to the gold answer? "
        "d) Conciseness (1-10): Does the answer provide information efficiently without unnecessary details? "
        "Calculate an overall quality score by taking the average of the four dimension scores. "
        "In your answer for each dimension, provide a justification why not a higher score and why not a lower score. "
        "Structure your response as follows:\n"
        "Evaluation:\n"
        "1. Relevance: [Score] - [Explanation]\n"
        "2. Accuracy: [Score] - [Explanation]\n"
        "3. Completeness: [Score] - [Explanation]\n"
        "4. Conciseness: [Score] - [Explanation]\n"
        "Overall Quality Score: [Average of the four above scores]\n\n"
        f"Question: {question}\n\n"
        f"Gold Answer: {gold_answer}\n\n"
        f"Model Answer: {model_answer}"
    )

def parse_scores(text: str) -> dict:
    """Extract dimension scores and overall from judge response."""
    if not text:
        return {}

    def extract(pattern: str) -> Optional[float]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                v = float(m.group(1))
                return round(v, 2) if 1.0 <= v <= 10.0 else None
            except ValueError:
                return None
        return None

    relevance    = extract(r"relevance[:\s]*(\d+(?:\.\d+)?)")
    accuracy     = extract(r"accuracy[:\s]*(\d+(?:\.\d+)?)")
    completeness = extract(r"completeness[:\s]*(\d+(?:\.\d+)?)")
    conciseness  = extract(r"conciseness[:\s]*(\d+(?:\.\d+)?)")
    overall      = extract(r"overall(?:\s+quality)?(?:\s+score)?[:\s]*(\d+(?:\.\d+)?)")

    dims = [v for v in [relevance, accuracy, completeness, conciseness] if v is not None]
    if dims and overall is None:
        overall = round(sum(dims) / len(dims), 2)

    return {
        "relevance"    : relevance,
        "accuracy"     : accuracy,
        "completeness" : completeness,
        "conciseness"  : conciseness,
        "overall"      : overall,
        "dims_parsed"  : len(dims),
    }

_rate_limiters: dict[str, object] = {}

class RateLimiter:
    def __init__(self, rpm: int):
        self._iv   = 60.0 / rpm
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self):
        # Sleep OUTSIDE the lock so other threads aren't blocked while waiting
        while True:
            with self._lock:
                now  = time.time()
                wait = self._iv - (now - self._last)
                if wait <= 0:
                    self._last = now
                    return
            time.sleep(wait)

for key, cfg in MODELS.items():
    _rate_limiters[key] = RateLimiter(cfg["rate"])

write_lock = threading.Lock()


def task_key(record: dict) -> tuple:
    return (
        record["id"],
        record["condition"],
        record["answer_model_key"],
        record["judge_key"],
    )


def is_successful_judgment(record: dict) -> bool:
    """True when the judge call produced usable dimension scores."""
    if record.get("error"):
        return False
    if record.get("overall") is not None:
        return True
    return record.get("dims_parsed", 0) >= 4


def record_quality(record: dict) -> tuple:
    """Higher = better when deduplicating duplicate keys."""
    return (
        1 if is_successful_judgment(record) else 0,
        record.get("dims_parsed", 0),
        1 if record.get("judge_response") else 0,
    )


def load_output_records(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def dedupe_records(records: list[dict]) -> tuple[list[dict], int]:
    """Keep the best record per task key. Returns (deduped, n_removed)."""
    best: dict[tuple, dict] = {}
    order: list[tuple] = []
    for record in records:
        key = task_key(record)
        if key not in best:
            order.append(key)
            best[key] = record
            continue
        if record_quality(record) > record_quality(best[key]):
            best[key] = record
    deduped = [best[key] for key in order]
    return deduped, len(records) - len(deduped)


def write_jsonl_atomic(path: str, records: list[dict]) -> None:
    """Rewrite JSONL via temp file so a crash cannot truncate the output."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def append_record(path: str, record: dict) -> None:
    line = json.dumps(record) + "\n"
    with write_lock:
        with open(path, "a") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

def call_judge(judge_key: str, question: str,
               model_answer: str, gold_answer: str,
               retries: int = 3) -> dict:
    cfg        = MODELS[judge_key]
    model_name = cfg["name"]
    limiter    = _rate_limiters[judge_key]
    user_p     = build_judge_prompt(question, model_answer, gold_answer)

    # Qwen3.5 thinking mode produces reasoning traces so long that the actual
    # structured evaluation scores are cut off by the token limit.
    # /no_think disables thinking mode so the model responds directly.
    if judge_key == "qwen3.5":
        user_p = "/no_think\n\n" + user_p

    for attempt in range(retries):
        limiter.acquire()
        try:
            t0  = time.time()
            # Paper uses a single unified prompt — no separate system message
            messages = [{"role": "user", "content": user_p}]
            res = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                max_tokens=cfg["max_tokens"],
            )
            latency = round(time.time() - t0, 3)
            msg     = res.choices[0].message
            content = msg.content
            reasoning = (
                getattr(msg, "reasoning_content", None)
                or (msg.model_extra or {}).get("reasoning_content")
            )
            # Qwen thinking-mode fallback
            if content is None and reasoning:
                content = reasoning
            if not content:
                return {"text": None, "latency": latency, "error": "null content"}
            return {"text": content.strip(), "latency": latency, "error": None}

        except Exception as e:
            err = str(e)
            if "401" in err:
                print("\n[FATAL] 401 — check API key.")
                sys.exit(1)
            if "404" in err:
                print(f"\n[FATAL] 404 — {model_name} not found.")
                sys.exit(1)
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 5)
            else:
                return {"text": None, "latency": 0,
                        "error": f"failed: {err[:120]}"}

    return {"text": None, "latency": 0, "error": "unknown"}

def judge_one(task: dict, output_jsonl: str) -> Optional[dict]:
    response = call_judge(
        task["judge_key"],
        task["question"],
        task["model_answer"],
        task["gold_answer"],
    )

    scores = parse_scores(response["text"]) if response["text"] else {}

    record = {
        "id"               : task["id"],
        "condition"        : task["condition"],
        "answer_model_key" : task["answer_model_key"],
        "judge_key"        : task["judge_key"],
        "judge_model"      : MODELS[task["judge_key"]]["name"],
        "relevance"        : scores.get("relevance"),
        "accuracy"         : scores.get("accuracy"),
        "completeness"     : scores.get("completeness"),
        "conciseness"      : scores.get("conciseness"),
        "overall"          : scores.get("overall"),
        "dims_parsed"      : scores.get("dims_parsed", 0),
        "latency_s"        : response["latency"],
        "error"            : response["error"],
        "judge_response"   : response["text"],
    }

    append_record(output_jsonl, record)
    return record

def load_answers(conditions: list[str]) -> list[dict]:
    answers = []
    for model_label, fpath in COMBINED_FILES.items():
        if not os.path.exists(fpath):
            print(f"  [{model_label}] File not found — skipping")
            continue
        answer_key = LABEL_TO_KEY[model_label]
        with open(fpath) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("condition") not in conditions:
                    continue
                if r.get("error") or not r.get("response_text") or not r.get("gold_answer"):
                    continue
                answers.append({
                    "id"               : r["id"],
                    "condition"        : r["condition"],
                    "answer_model_key" : answer_key,
                    "question"         : r["question"],
                    "model_answer"     : r["response_text"],
                    "gold_answer"      : r["gold_answer"],
                })
    return answers

def main():
    if not LITELLM_BASE_URL or not LITELLM_API_KEY:
        print("[ERROR] LITELLM_BASE_URL and LITELLM_API_KEY must be set as environment variables.")
        print("  Copy .env.example to .env and fill in your values.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="LLM-as-judge for SciDQA")
    parser.add_argument("--judge",      choices=list(MODELS), default=None,
                        help="Only use this model as judge (default: all)")
    parser.add_argument("--condition",  choices=ALL_CONDITIONS, default=None,
                        help="Score one condition only (default: all)")
    parser.add_argument("--self-judge", action="store_true",
                        help="Each model judges its OWN answers (self-bias check). "
                             "Writes to separate files scidqa_llm_judge_self.*")
    parser.add_argument("--dedupe", action="store_true",
                        help="Rewrite output JSONL keeping the best record per "
                             "(id, condition, answer_model, judge) and exit.")
    args = parser.parse_args()

    judge_filter     = args.judge
    conditions       = [args.condition] if args.condition else ALL_CONDITIONS
    self_judge_mode  = args.self_judge

    if self_judge_mode:
        output_jsonl        = os.path.join(SCRIPT_DIR, "scidqa_llm_judge_self.jsonl")
        output_report       = os.path.join(SCRIPT_DIR, "scidqa_llm_judge_self_report.txt")
        output_report_per_model = {
            "gemma4"  : os.path.join(SCRIPT_DIR, "scidqa_llm_judge_self_gemma4.txt"),
            "gptoss"  : os.path.join(SCRIPT_DIR, "scidqa_llm_judge_self_gptoss.txt"),
            "qwen3.5" : os.path.join(SCRIPT_DIR, "scidqa_llm_judge_self_qwen3.5.txt"),
        }
        judge_map_effective = {
            "gemma4"  : ["gemma4"],
            "gptoss"  : ["gptoss"],
            "qwen3.5" : ["qwen3.5"],
        }
    else:
        output_jsonl        = os.path.join(SCRIPT_DIR, "scidqa_llm_judge.jsonl")
        output_report       = os.path.join(SCRIPT_DIR, "scidqa_llm_judge_report.txt")
        output_report_per_model = {
            "gemma4"  : os.path.join(SCRIPT_DIR, "scidqa_llm_judge_gemma4.txt"),
            "gptoss"  : os.path.join(SCRIPT_DIR, "scidqa_llm_judge_gptoss.txt"),
            "qwen3.5" : os.path.join(SCRIPT_DIR, "scidqa_llm_judge_qwen3.5.txt"),
        }
        judge_map_effective = JUDGE_MAP

    if args.dedupe:
        records = load_output_records(output_jsonl)
        if not records:
            print(f"  No records in {os.path.basename(output_jsonl)}")
            return
        deduped, removed = dedupe_records(records)
        write_jsonl_atomic(output_jsonl, deduped)
        print(f"  Deduped {os.path.basename(output_jsonl)}")
        print(f"    Before : {len(records):,} lines")
        print(f"    After  : {len(deduped):,} lines")
        print(f"    Removed: {removed:,} stale duplicate/failed lines")
        return

    mode_label = "Self-Judge (bias check)" if self_judge_mode else "Cross-Reference"
    print(f"\n{'═'*65}")
    print(f"  SciDQA — LLM-as-Judge  [{mode_label}]")
    print(f"  Conditions  : {conditions}")
    print(f"  Judge filter: {judge_filter or 'all'}")
    print(f"  Output file : {output_jsonl}")
    if self_judge_mode:
        print(f"  Mode: each model judges its OWN answers")
    print(f"{'═'*65}\n")

    existing_records = load_output_records(output_jsonl)
    already_done: set[tuple] = set()
    for record in existing_records:
        if is_successful_judgment(record):
            already_done.add(task_key(record))

    if existing_records:
        n_parse_fail = sum(
            1 for r in existing_records
            if not r.get("error") and not is_successful_judgment(r)
        )
        n_api_errors = sum(1 for r in existing_records if r.get("error"))
        print(f"  Resuming from {os.path.basename(output_jsonl)}")
        print(f"    Already done (successful)  : {len(already_done):,}")
        print(f"    API errors (will retry)    : {n_api_errors:,}")
        print(f"    Parse failures (will retry): {n_parse_fail:,}")
        print(f"    Total lines in file        : {len(existing_records):,}\n")

    print("  Loading answers from combined JSONL files...")
    answers = load_answers(conditions)
    print(f"  Loaded {len(answers):,} answers\n")

    tasks: list[dict] = []
    for ans in answers:
        judge_keys = judge_map_effective[ans["answer_model_key"]]
        for jk in judge_keys:
            if judge_filter and jk != judge_filter:
                continue
            key = (ans["id"], ans["condition"], ans["answer_model_key"], jk)
            if key in already_done:
                continue
            tasks.append({**ans, "judge_key": jk})

    if not tasks:
        print("  All tasks already done.")
    else:
        total_calls = len(tasks)
        est_min     = total_calls / 180
        print(f"  Tasks to run : {total_calls:,}")
        print(f"  Est. time    : ~{est_min:.0f} min at 180 req/min avg\n")

        all_records: list[dict] = []
        max_workers = 40
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(judge_one, t, output_jsonl): t for t in tasks}
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures), desc="Judging"):
                try:
                    rec = future.result()
                    if rec:
                        all_records.append(rec)
                except SystemExit:
                    raise
                except Exception as e:
                    print(f"\n  Worker error: {e}")

    print("\n  Building report from output file...")
    all_records_raw = load_output_records(output_jsonl)
    all_records, n_dupes = dedupe_records(all_records_raw)
    if n_dupes:
        print(f"  Note: {n_dupes:,} duplicate/stale lines ignored in report "
              f"(run --dedupe to compact the file).")

    good_records = [r for r in all_records if is_successful_judgment(r)]

    report_lines: list[str] = []

    def p(s: str = "") -> None:
        report_lines.append(s)
        print(s)

    answer_keys = ["gemma4", "gptoss", "qwen3.5"]
    labels      = {"gemma4": "gemma-4-31B", "gptoss": "gpt-oss-120b", "qwen3.5": "Qwen3.5-122B"}
    DIMS        = ["relevance", "accuracy", "completeness", "conciseness"]
    CONDITIONS_TO_DO = conditions

    def norm(v: float) -> float:
        return round(v * 10, 1)

    def group_by(field: str) -> dict:
        d: dict[tuple, list[float]] = defaultdict(list)
        for r in good_records:
            if r.get(field) is not None:
                d[(r["answer_model_key"], r["condition"])].append(r[field])
        return d

    grouped_overall = group_by("overall")

    p(f"{'═'*70}")
    p(f"  SciDQA — LLM-as-Judge Evaluation Report")
    p(f"  Cross-reference judging  |  Scale: 1–100  |  Avg of 2 judges per answer")
    p(f"{'═'*70}")
    p()

    p(f"{'─'*70}")
    p(f"  SECTION 1 — Overall ALS per Model (averaged across all conditions)")
    p(f"  The single headline number. Higher = better overall answer quality.")
    p(f"{'─'*70}")
    p(f"  {'Model':<22}  {'ALS (1–100)':>12}  {'# judgments':>13}")
    p(f"  {'─'*22}  {'─'*12}  {'─'*13}")
    for k in answer_keys:
        all_vals = [v for cond in CONDITIONS_TO_DO
                    for v in grouped_overall.get((k, cond), [])]
        if all_vals:
            p(f"  {labels[k]:<22}  {norm(statistics.mean(all_vals)):>12.1f}  {len(all_vals):>13,}")
        else:
            p(f"  {labels[k]:<22}  {'—':>12}  {'0':>13}")
    p()

    p(f"{'─'*70}")
    p(f"  SECTION 2 — ALS Score by Condition  (matches SciDQA paper Table 5)")
    p(f"  Which RAG setup produced the best answers for each model?")
    p(f"{'─'*70}")
    col = "".join(f" {labels[k][:13]:>13}" for k in answer_keys)
    p(f"  {'Condition':<17}{col}   Best model")
    p(f"  {'─'*17}" + "".join(f" {'─'*13}" for _ in answer_keys) + "   ──────────")
    for cond in CONDITIONS_TO_DO:
        scores = {k: norm(statistics.mean(grouped_overall[(k, cond)]))
                  for k in answer_keys if grouped_overall.get((k, cond))}
        best_k = max(scores, key=scores.get) if scores else None
        row = f"  {cond:<17}"
        for k in answer_keys:
            row += f" {scores[k]:>13.1f}" if k in scores else f" {'—':>13}"
        row += f"   {labels[best_k]}" if best_k else ""
        p(row)
    p()

    p(f"{'─'*70}")
    p(f"  SECTION 3 — Per-Dimension Scores averaged across all conditions")
    p(f"  Where is each model strong or weak?")
    p(f"{'─'*70}")
    p(f"  {'Dimension':<15}" + "".join(f" {labels[k][:13]:>13}" for k in answer_keys))
    p(f"  {'─'*15}" + "".join(f" {'─'*13}" for _ in answer_keys))
    for dim in DIMS:
        dim_vals: dict[str, list[float]] = defaultdict(list)
        for r in good_records:
            if r.get(dim) is not None:
                dim_vals[r["answer_model_key"]].append(r[dim])
        row = f"  {dim.capitalize():<15}"
        for k in answer_keys:
            row += f" {norm(statistics.mean(dim_vals[k])):>13.1f}" if dim_vals[k] else f" {'—':>13}"
        p(row)
    p()

    p(f"{'─'*70}")
    p(f"  SECTION 4 — Per-Dimension × Condition (full detail)")
    p(f"{'─'*70}")
    for dim in DIMS:
        dim_grouped = group_by(dim)
        p(f"  {dim.capitalize()} (1–100):")
        p(f"  {'Condition':<17}" + "".join(f" {labels[k][:13]:>13}" for k in answer_keys))
        p(f"  {'─'*17}" + "".join(f" {'─'*13}" for _ in answer_keys))
        for cond in CONDITIONS_TO_DO:
            row = f"  {cond:<17}"
            for k in answer_keys:
                vals = dim_grouped.get((k, cond), [])
                row += f" {norm(statistics.mean(vals)):>13.1f}" if vals else f" {'—':>13}"
            p(row)
        p()

    p(f"{'─'*70}")
    p(f"  SECTION 5 — Scores Given BY Each Judge")
    p(f"  If two judges give very different averages, treat results with caution.")
    p(f"{'─'*70}")
    judge_vals: dict[str, list[float]] = defaultdict(list)
    for r in good_records:
        judge_vals[r["judge_key"]].append(r["overall"])
    p(f"  {'Judge model':<22}  {'Avg score given':>15}  {'# judgments':>13}")
    p(f"  {'─'*22}  {'─'*15}  {'─'*13}")
    for jk in answer_keys:
        vals = judge_vals.get(jk, [])
        if vals:
            p(f"  {labels[jk]:<22}  {norm(statistics.mean(vals)):>15.1f}  {len(vals):>13,}")
        else:
            p(f"  {labels[jk]:<22}  {'—':>15}  {'0':>13}")
    p()

    p(f"{'─'*70}")
    p(f"  SECTION 6 — Data Quality & Error Summary")
    p(f"{'─'*70}")
    total        = len(all_records)
    n_api_errors = len([r for r in all_records if r.get("error")])
    n_success    = len([r for r in all_records if is_successful_judgment(r)])
    n_parse_fail = len([r for r in all_records
                        if not r.get("error") and not is_successful_judgment(r)])

    p(f"  Total judgments (deduped)    : {total:,}")
    p(f"  Successful (all 4 dims)      : {n_success:,}  ({100*n_success/max(total,1):.1f}%)")
    p(f"  API errors                   : {n_api_errors:,}  ({100*n_api_errors/max(total,1):.1f}%)")
    p(f"  Score parse failures         : {n_parse_fail:,}  ({100*n_parse_fail/max(total,1):.1f}%)")
    p()

    error_codes: dict[str, int] = defaultdict(int)
    for r in all_records:
        if r.get("error"):
            m = re.search(r"Error code[:\s]+(\d+)", r["error"])
            code = m.group(1) if m else "other"
            error_codes[code] += 1

    if error_codes:
        explanations = {
            "500": "LiteLLM gateway internal error (triggered by long inputs)",
            "429": "rate limit exceeded",
            "502": "bad gateway — server temporarily unavailable",
        }
        p(f"  API Error Breakdown:")
        for code, cnt in sorted(error_codes.items(), key=lambda x: -x[1]):
            note = explanations.get(code, "")
            p(f"    HTTP {code} : {cnt:>5,}  — {note}")
        p()

    p(f"  Errors by Judge Model:")
    p(f"  {'Judge':<22}  {'API errors':>10}  {'Parse fails':>12}  {'Successful':>10}")
    p(f"  {'─'*22}  {'─'*10}  {'─'*12}  {'─'*10}")
    for jk in answer_keys:
        jrecs    = [r for r in all_records if r["judge_key"] == jk]
        j_err    = sum(1 for r in jrecs if r.get("error"))
        j_parse  = sum(1 for r in jrecs if not r.get("error") and not is_successful_judgment(r))
        j_ok     = sum(1 for r in jrecs if is_successful_judgment(r))
        p(f"  {labels[jk]:<22}  {j_err:>10,}  {j_parse:>12,}  {j_ok:>10,}")
    p()

    p(f"  Errors by Answer Model:")
    p(f"  {'Answer model':<22}  {'API errors':>10}  {'Parse fails':>12}  {'Successful':>10}")
    p(f"  {'─'*22}  {'─'*10}  {'─'*12}  {'─'*10}")
    for ak in answer_keys:
        arecs    = [r for r in all_records if r["answer_model_key"] == ak]
        a_err    = sum(1 for r in arecs if r.get("error"))
        a_parse  = sum(1 for r in arecs if not r.get("error") and not is_successful_judgment(r))
        a_ok     = sum(1 for r in arecs if is_successful_judgment(r))
        p(f"  {labels[ak]:<22}  {a_err:>10,}  {a_parse:>12,}  {a_ok:>10,}")
    p()
    p(f"  Raw 1–10 scores in JSONL. Report values = raw × 10.")
    p(f"{'═'*70}")

    with open(output_report, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    for model_key in answer_keys:
        model_records = [r for r in good_records if r["answer_model_key"] == model_key]
        label         = labels[model_key]
        judge_labels  = [labels[jk] for jk in answer_keys if jk != model_key]
        lines: list[str] = []

        def pm(s: str = "") -> None:
            lines.append(s)

        pm(f"{'═'*65}")
        pm(f"  LLM-as-Judge Report — {label}")
        pm(f"  Judged by: {' and '.join(judge_labels)}")
        pm(f"  Scale: 1–100  |  Avg of 2 judges per answer")
        pm(f"{'═'*65}")
        pm()

        pm(f"  Overall ALS per Condition:")
        pm(f"  {'Condition':<17}  {'ALS':>7}  {'# samples':>10}")
        pm(f"  {'─'*17}  {'─'*7}  {'─'*10}")
        for cond in CONDITIONS_TO_DO:
            vals = [r["overall"] for r in model_records if r["condition"] == cond]
            if vals:
                pm(f"  {cond:<17}  {norm(statistics.mean(vals)):>7.1f}  {len(vals):>10,}")
            else:
                pm(f"  {cond:<17}  {'—':>7}  {'0':>10}")
        overall_all = [r["overall"] for r in model_records]
        pm(f"  {'─'*17}  {'─'*7}  {'─'*10}")
        pm(f"  {'ALL CONDITIONS':<17}  {norm(statistics.mean(overall_all)):>7.1f}  {len(overall_all):>10,}"
           if overall_all else f"  {'ALL CONDITIONS':<17}  {'—':>7}  {'0':>10}")
        pm()

        pm(f"  Per-Dimension Scores (averaged across all conditions):")
        pm(f"  {'Dimension':<15}  {'Score':>7}")
        pm(f"  {'─'*15}  {'─'*7}")
        for dim in DIMS:
            vals = [r[dim] for r in model_records if r.get(dim) is not None]
            pm(f"  {dim.capitalize():<15}  {norm(statistics.mean(vals)):>7.1f}" if vals
               else f"  {dim.capitalize():<15}  {'—':>7}")
        pm()

        pm(f"  Per-Dimension × Condition:")
        dim_header = "".join(f" {d[:11]:>11}" for d in ["Relevance", "Accuracy", "Complete.", "Concise."])
        pm(f"  {'Condition':<17}{dim_header}")
        pm(f"  {'─'*17}" + "".join(f" {'─'*11}" for _ in DIMS))
        for cond in CONDITIONS_TO_DO:
            row = f"  {cond:<17}"
            for dim in DIMS:
                vals = [r[dim] for r in model_records
                        if r["condition"] == cond and r.get(dim) is not None]
                row += f" {norm(statistics.mean(vals)):>11.1f}" if vals else f" {'—':>11}"
            pm(row)
        pm()

        pm(f"  Scores by Individual Judge:")
        pm(f"  {'Judge':<22}  {'Avg score':>9}  {'# judgments':>12}")
        pm(f"  {'─'*22}  {'─'*9}  {'─'*12}")
        for jk in answer_keys:
            if jk == model_key:
                continue
            jvals = [r["overall"] for r in model_records if r["judge_key"] == jk]
            if jvals:
                pm(f"  {labels[jk]:<22}  {norm(statistics.mean(jvals)):>9.1f}  {len(jvals):>12,}")
            else:
                pm(f"  {labels[jk]:<22}  {'—':>9}  {'0':>12}")
        pm()
        pm(f"{'═'*65}")

        with open(output_report_per_model[model_key], "w") as f:
            f.write("\n".join(lines) + "\n")

    print(f"\n  → Saved JSONL       : {os.path.basename(output_jsonl)}")
    print(f"  → Saved combined    : {os.path.basename(output_report)}")
    for k, path in output_report_per_model.items():
        print(f"  → Saved {labels[k]:<18}: {os.path.basename(path)}")
    print()


if __name__ == "__main__":
    main()
