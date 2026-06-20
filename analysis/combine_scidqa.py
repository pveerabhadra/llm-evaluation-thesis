"""
combine_scidqa.py
-----------------
Merges all SciDQA result batches for each model, handles re-runs and retries,
deduplicates by (question_id, condition), and produces:
  1. A combined JSONL per model  →  analysis/scidqa_<model>_combined.jsonl
  2. A text report                →  analysis/scidqa_combined_report.txt

Merge order matters — later files overwrite earlier ones for the same
(id, condition) key so that retries always win over original error records.

Merge order used:
  Gemma-4 : offset0 → offset500 → … → offset2500 → retries (retries last)
  GPT-OSS : v1 → v2 → … → v6  (no retries needed)
  Qwen 3.5: archive/v1-v4 → retries_v1v4 → v5 → v6  (retries replace bad 4k records)

Known remaining issues in Qwen 3.5 (flagged in report, not silently dropped):
  - 78 null-content errors spread across conditions
  - 413 thinking-only records (response_text == reasoning) concentrated in
    no_retrieval (384) and long_context (29) — these need a max_tokens=16384 retry

Usage:
    python3 combine_scidqa.py          # print report only
    python3 combine_scidqa.py --save   # also write combined JSONL + report file
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import Counter, defaultdict

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
THESIS_ROOT = os.path.dirname(SCRIPT_DIR)
SCIDQA_DIR  = os.path.join(THESIS_ROOT, "SciDQA")

CONDITIONS = ["no_retrieval", "rag_top3", "rag_top5", "rag_dense", "long_context"]

# ── File discovery ─────────────────────────────────────────────────────────────

def _glob(pattern: str) -> list[str]:
    return sorted(glob.glob(os.path.join(SCIDQA_DIR, pattern)))


_lc_retry = os.path.join(SCIDQA_DIR, "scidqa_qwen3.5_longcontext_retry.jsonl")

MODEL_FILES: dict[str, list[str]] = {
    "gemma-4-31B": (
        _glob("scidqa_gemma4_offset*.jsonl") +
        [os.path.join(SCIDQA_DIR, "scidqa_gemma4_retries.jsonl")]
    ),
    "gpt-oss-120b": (
        _glob("scidqa_gptoss_v[123456].jsonl")
    ),
    "Qwen3.5-122B": (
        sorted(glob.glob(os.path.join(SCIDQA_DIR, "archive", "qwen_4k_archive", "scidqa_qwen3.5_v*.jsonl"))) +
        [os.path.join(SCIDQA_DIR, "scidqa_qwen3.5_retries_v1v4.jsonl")] +
        _glob("scidqa_qwen3.5_v[56].jsonl") +
        # Long-context 16k retry — included automatically once the file exists
        ([_lc_retry] if os.path.exists(_lc_retry) else [])
    ),
}


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_and_dedup(files: list[str]) -> tuple[list[dict], int]:
    """
    Load files in order, dedup by (id, condition) keeping the LAST occurrence.
    Returns (records, n_removed).
    """
    seen: dict[tuple, dict] = {}
    total = 0
    for fpath in files:
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    seen[(r["id"], r["condition"])] = r
                    total += 1
    records = list(seen.values())
    return records, total - len(records)


# ── Analysis ───────────────────────────────────────────────────────────────────

def _safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def analyse_model(records: list[dict], model_label: str) -> dict:
    """Compute per-condition and overall stats."""
    total    = len(records)
    errors   = [r for r in records if r.get("error")]
    answered = [r for r in records if not r.get("error")]

    # Detect thinking-only (response_text == reasoning — Qwen-specific)
    thinking_only = [
        r for r in answered
        if r.get("response_text") and r.get("reasoning")
        and r["response_text"] == r["reasoning"]
    ]

    # Per-condition breakdown
    cond_stats: dict[str, dict] = {}
    for cond in CONDITIONS:
        cond_recs  = [r for r in records if r["condition"] == cond]
        cond_ans   = [r for r in cond_recs if not r.get("error")]
        cond_errs  = [r for r in cond_recs if r.get("error")]
        cond_think = [
            r for r in cond_ans
            if r.get("response_text") and r.get("reasoning")
            and r["response_text"] == r["reasoning"]
        ]

        cond_stats[cond] = {
            "total"            : len(cond_recs),
            "answered"         : len(cond_ans),
            "errors"           : len(cond_errs),
            "thinking_only"    : len(cond_think),
            "rouge_1"          : _safe_mean([r["rouge_1"]   for r in cond_ans if r.get("rouge_1")   is not None]),
            "rouge_2"          : _safe_mean([r["rouge_2"]   for r in cond_ans if r.get("rouge_2")   is not None]),
            "rouge_l"          : _safe_mean([r["rouge_l"]   for r in cond_ans if r.get("rouge_l")   is not None]),
            "rouge_avg"        : _safe_mean([r["rouge_avg"] for r in cond_ans if r.get("rouge_avg") is not None]),
            "ngram_grounding"  : _safe_mean([r["ngram_grounding_score"] for r in cond_ans if r.get("ngram_grounding_score") is not None]),
            "no_answer_pct"    : sum(1 for r in cond_ans if r.get("no_answer_signal")) / max(len(cond_ans), 1) * 100,
            "avg_latency"      : _safe_mean([r["latency_s"] for r in cond_ans if r.get("latency_s")]),
            "avg_tokens"       : _safe_mean([r["completion_tokens"] for r in cond_ans if r.get("completion_tokens")]),
            "avg_response_len" : _safe_mean([r["response_length_chars"] for r in cond_ans if r.get("response_length_chars")]),
            "rag_chars"        : _safe_mean([r["rag_chars_given"] for r in cond_ans if r.get("rag_chars_given")]),
        }

    return {
        "model"         : model_label,
        "total"         : total,
        "errors"        : len(errors),
        "thinking_only" : len(thinking_only),
        "answered"      : len(answered),
        "unique_ids"    : len({r["id"] for r in records}),
        "cond_stats"    : cond_stats,
    }


# ── Printing ───────────────────────────────────────────────────────────────────

def fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"


def print_model_report(stats: dict, files_used: list[str], n_removed: int) -> list[str]:
    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)
        print(s)

    w = 70
    p(f"\n{'═' * w}")
    p(f"  {stats['model']}")
    p(f"{'═' * w}")
    p(f"  Files (in merge order):")
    for f in files_used:
        p(f"    {os.path.basename(f)}")
    p()
    p(f"  Total records loaded : {stats['total'] + n_removed}")
    p(f"  Duplicates removed   : {n_removed}  (retries always win over original errors)")
    p(f"  Unique (id,cond) kept: {stats['total']}")
    p(f"  Unique question IDs  : {stats['unique_ids']}")
    p(f"  Errors remaining     : {stats['errors']}")
    p(f"  Thinking-only records: {stats['thinking_only']}  (response = internal reasoning only)")

    if stats["thinking_only"] > 0:
        p(f"  ⚠  Thinking-only records affect ROUGE averages — these are Qwen token-exhaustion records.")
        p(f"     Run the max_tokens=16384 retry to fix remaining 413 records.")
    if stats["errors"] > 0:
        p(f"  ⚠  Errored records are excluded from ROUGE/grounding averages below.")
    p()

    # Per-condition table
    p(f"  {'Condition':<15} {'R-1':>7} {'R-2':>7} {'R-L':>7} {'R-avg':>7} {'Grnd':>7} {'No-Ans%':>8} {'Errors':>7} {'Think':>6}")
    p(f"  {'─'*15} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*8} {'─'*7} {'─'*6}")
    for cond in CONDITIONS:
        cs = stats["cond_stats"][cond]
        p(
            f"  {cond:<15} "
            f"{cs['rouge_1']:>7.4f} "
            f"{cs['rouge_2']:>7.4f} "
            f"{cs['rouge_l']:>7.4f} "
            f"{cs['rouge_avg']:>7.4f} "
            f"{cs['ngram_grounding']:>7.4f} "
            f"{cs['no_answer_pct']:>7.1f}% "
            f"{cs['errors']:>7} "
            f"{cs['thinking_only']:>6}"
        )
    p()

    p(f"  {'Condition':<15} {'Latency':>8} {'Tokens':>8} {'RespLen':>9} {'RAGchars':>10}")
    p(f"  {'─'*15} {'─'*8} {'─'*8} {'─'*9} {'─'*10}")
    for cond in CONDITIONS:
        cs = stats["cond_stats"][cond]
        p(
            f"  {cond:<15} "
            f"{cs['avg_latency']:>7.1f}s "
            f"{cs['avg_tokens']:>8.0f} "
            f"{cs['avg_response_len']:>9.0f} "
            f"{cs['rag_chars']:>10.0f}"
        )
    return lines


def print_comparison_table(all_stats: list[dict]) -> list[str]:
    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)
        print(s)

    models = [s["model"] for s in all_stats]
    w = 70
    p(f"\n{'═' * w}")
    p(f"  CROSS-MODEL COMPARISON — ROUGE-avg per condition")
    p(f"{'═' * w}")
    header = f"  {'Condition':<15}" + "".join(f" {m[:14]:>14}" for m in models)
    p(header)
    p(f"  {'─'*15}" + "".join(f" {'─'*14}" for _ in models))
    for cond in CONDITIONS:
        row = f"  {cond:<15}"
        for s in all_stats:
            v = s["cond_stats"][cond]["rouge_avg"]
            row += f" {v:>14.4f}"
        p(row)

    p()
    p(f"  CROSS-MODEL COMPARISON — n-gram grounding score per condition")
    p(f"  {'─'*15}" + "".join(f" {'─'*14}" for _ in models))
    for cond in CONDITIONS:
        row = f"  {cond:<15}"
        for s in all_stats:
            v = s["cond_stats"][cond]["ngram_grounding"]
            row += f" {v:>14.4f}"
        p(row)

    p()
    p(f"  CROSS-MODEL COMPARISON — no-answer rate (%) per condition")
    p(f"  {'─'*15}" + "".join(f" {'─'*14}" for _ in models))
    for cond in CONDITIONS:
        row = f"  {cond:<15}"
        for s in all_stats:
            v = s["cond_stats"][cond]["no_answer_pct"]
            row += f" {v:>13.1f}%"
        p(row)

    p()
    p(f"  SUMMARY")
    p(f"  {'─'*15}" + "".join(f" {'─'*14}" for _ in models))
    p(f"  {'Total records':<15}" + "".join(f" {s['total']:>14,}" for s in all_stats))
    p(f"  {'Errors':<15}" + "".join(f" {s['errors']:>14,}" for s in all_stats))
    p(f"  {'Thinking-only':<15}" + "".join(f" {s['thinking_only']:>14,}" for s in all_stats))
    return lines


# ── Save ───────────────────────────────────────────────────────────────────────

def save_combined(records: list[dict], model_key: str) -> str:
    fname = os.path.join(SCRIPT_DIR, f"scidqa_{model_key}_combined.jsonl")
    with open(fname, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return fname


def save_report(all_lines: list[str]) -> str:
    fname = os.path.join(SCRIPT_DIR, "scidqa_combined_report.txt")
    with open(fname, "w") as f:
        f.write("\n".join(all_lines) + "\n")
    return fname


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Combine and analyse SciDQA result batches")
    parser.add_argument("--save", action="store_true",
                        help="Write combined JSONL files + report to disk")
    args = parser.parse_args()

    print(f"\n{'═' * 70}")
    print(f"  SciDQA — Combined Analysis")
    print(f"  SciDQA dir: {SCIDQA_DIR}")
    print(f"{'═' * 70}")
    print(f"  Expected: 2,937 questions × 5 conditions = 14,685 records per model")
    print(f"  Conditions: {', '.join(CONDITIONS)}")

    all_stats: list[dict] = []
    all_lines: list[str] = []
    KEY_MAP = {
        "gemma-4-31B"  : "gemma4",
        "gpt-oss-120b" : "gptoss",
        "Qwen3.5-122B" : "qwen3.5",
    }

    for model_label, files in MODEL_FILES.items():
        existing_files = [f for f in files if os.path.exists(f)]
        if not existing_files:
            print(f"\n  [{model_label}] No files found — skipping")
            continue

        records, n_removed = load_and_dedup(existing_files)
        stats = analyse_model(records, model_label)
        all_stats.append(stats)

        model_lines = print_model_report(stats, existing_files, n_removed)
        all_lines.extend(model_lines)

        if args.save:
            key = KEY_MAP.get(model_label, model_label)
            out = save_combined(records, key)
            print(f"  → Saved: {os.path.basename(out)}")

    if len(all_stats) > 1:
        comp_lines = print_comparison_table(all_stats)
        all_lines.extend(comp_lines)

    print(f"\n{'═' * 70}")
    print(f"  Total records analysed: {sum(s['total'] for s in all_stats):,}")
    print(f"{'═' * 70}\n")

    if args.save:
        out = save_report(all_lines)
        print(f"  → Report saved: {os.path.basename(out)}\n")


if __name__ == "__main__":
    main()
