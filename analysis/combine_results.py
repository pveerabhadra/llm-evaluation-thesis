from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import Counter, defaultdict

THESIS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where to find result files per language
RESULT_DIRS = {
    "en": os.path.join(THESIS_ROOT, "MMLU"),
    "de": os.path.join(THESIS_ROOT, "MMLU German"),
}

# File patterns per model per language
FILE_PATTERNS = {
    "en": {
        "gpt-oss-120b" : "mmlu_en_gptoss_v*.jsonl",
        "gemma-4-31B"  : "mmlu_en_gemma4_v*.jsonl",
        "Qwen3.5-122B" : "mmlu_en_qwen3.5_v*.jsonl",
    },
    "de": {
        "gpt-oss-120b" : "mmlu_de_gptoss_v*.jsonl",
        "gemma-4-31B"  : "mmlu_de_gemma4_v*.jsonl",
        "Qwen3.5-122B" : "mmlu_de_qwen3.5_v*.jsonl",
    },
}


def load_model_files(pattern: str, result_dir: str) -> tuple[list[dict], list[str]]:
    """Load all matching JSONL files for one model. Returns (records, filenames_used)."""
    files = sorted(glob.glob(os.path.join(result_dir, pattern)))
    # Exclude pilot files (pilot_* prefix) to keep only real runs
    files = [f for f in files if not os.path.basename(f).startswith("pilot_")]
    records = []
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records, [os.path.basename(f) for f in files]


def find_duplicates(records: list[dict]) -> list[str]:
    """Return list of question texts that appear more than once (before dedup)."""
    counts = Counter(r["question"] for r in records)
    return [q for q, n in counts.items() if n > 1]


def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    """
    Remove duplicate questions, keeping the LAST occurrence.
    Records must be passed in file order (v1, v2, v3…) so that re-runs
    with better settings win over earlier runs with the same question.
    Returns (deduped_records, n_removed).
    """
    seen: dict[str, dict] = {}
    for r in records:
        seen[r["question"]] = r   # later record overwrites earlier
    deduped = list(seen.values())
    return deduped, len(records) - len(deduped)


def analyse_model(records: list[dict], model_label: str) -> dict:
    """Compute all stats for one model's combined records."""
    total    = len(records)
    errors   = [r for r in records if r.get("error")]
    answered = [r for r in records if not r.get("error")]
    correct  = [r for r in answered if r.get("is_correct")]

    latencies = [r["latency_s"] for r in records if r.get("latency_s")]
    comp_toks = [r["completion_tokens"] for r in records if r.get("completion_tokens")]

    # Per-subject breakdown
    subject_correct = defaultdict(int)
    subject_total   = defaultdict(int)
    for r in answered:
        subj = r.get("subject", "unknown")
        subject_total[subj] += 1
        if r.get("is_correct"):
            subject_correct[subj] += 1

    subject_acc = {
        s: subject_correct[s] / subject_total[s]
        for s in subject_total
    }

    # Parse method breakdown
    parse_counts = Counter(r.get("parse_method") for r in answered)

    return {
        "model"        : model_label,
        "total"        : total,
        "answered"     : len(answered),
        "errors"       : len(errors),
        "error_rate"   : len(errors) / total if total else 0,
        "correct"      : len(correct),
        "accuracy_overall"  : len(correct) / total if total else 0,
        "accuracy_answered" : len(correct) / len(answered) if answered else 0,
        "avg_latency"  : statistics.mean(latencies) if latencies else 0,
        "avg_completion_tokens": statistics.mean(comp_toks) if comp_toks else 0,
        "subject_acc"  : subject_acc,
        "parse_counts" : dict(parse_counts),
        "n_subjects"   : len(subject_total),
    }


def print_model_report(stats: dict, files_used: list[str], duplicates: list[str]) -> None:
    w = 65
    print(f"\n{'═' * w}")
    print(f"  {stats['model']}")
    print(f"{'═' * w}")
    print(f"  Files used    : {', '.join(files_used) if files_used else 'none found'}")
    print()
    print(f"  Total records : {stats['total']}")
    print(f"  Answered      : {stats['answered']}  ({stats['answered']/max(stats['total'],1)*100:.1f}%)")
    print(f"  Errors        : {stats['errors']}  ({stats['error_rate']*100:.1f}%)")
    print(f"  Correct       : {stats['correct']}")
    print()
    print(f"  Overall accuracy  : {stats['accuracy_overall']*100:.2f}%  (correct / total including errors)")
    print(f"  Answered accuracy : {stats['accuracy_answered']*100:.2f}%  (correct / answered only)")
    print()
    print(f"  Avg latency   : {stats['avg_latency']:.1f}s")
    print(f"  Avg comp. tok : {stats['avg_completion_tokens']:.0f}")
    print()

    # Duplicate check
    n_removed = stats.get("n_removed", 0)
    if duplicates:
        print(f"  ℹ {len(duplicates)} duplicate questions found across batches → {n_removed} records removed.")
        print(f"    (Kept the LATEST version of each duplicate — better token settings win.)")
        for q in duplicates[:3]:
            print(f"    e.g.: {q[:80]}…")
        print()
    else:
        print(f"  ✓ No duplicate questions — all batches are non-overlapping")
        print()

    # Parse methods
    print(f"  Parse methods:")
    for method, cnt in sorted(stats['parse_counts'].items(), key=lambda x: -x[1]):
        print(f"    {method:<20} {cnt:>5} responses")
    print()

    # Per-subject table
    print(f"  {'Subject':<45} {'Correct':>7}  {'Total':>6}  {'Acc':>6}")
    print(f"  {'─'*45} {'─'*7}  {'─'*6}  {'─'*6}")

    subject_acc  = stats["subject_acc"]
    all_subjects = sorted(subject_acc.keys())

    # Group by accuracy tier for easier reading
    perfect   = [(s, subject_acc[s]) for s in all_subjects if subject_acc[s] == 1.0]
    strong    = [(s, subject_acc[s]) for s in all_subjects if 0.8 <= subject_acc[s] < 1.0]
    moderate  = [(s, subject_acc[s]) for s in all_subjects if 0.6 <= subject_acc[s] < 0.8]
    weak      = [(s, subject_acc[s]) for s in all_subjects if subject_acc[s] < 0.6]

    # Find correct + total per subject
    answered_recs = [r for r in [] ]  # placeholder; we re-derive below
    # For the table we need raw counts, not just % — recalculate here from stats
    # (We print from subject_acc; compute total from accuracy × n using answered records)
    # Since we only have aggregated stats, print acc % only
    for s in sorted(subject_acc.keys()):
        acc = subject_acc[s]
        marker = "  " if acc >= 0.8 else "⚠ " if acc >= 0.6 else "✗ "
        print(f"  {marker}{s:<43} {acc*100:>7.1f}%")

    # Summary tier counts
    print()
    print(f"  Accuracy tiers across {stats['n_subjects']} subjects:")
    print(f"    ≥ 90%  : {sum(1 for a in subject_acc.values() if a >= 0.9):>3} subjects")
    print(f"    80–89% : {sum(1 for a in subject_acc.values() if 0.8 <= a < 0.9):>3} subjects")
    print(f"    60–79% : {sum(1 for a in subject_acc.values() if 0.6 <= a < 0.8):>3} subjects")
    print(f"    < 60%  : {sum(1 for a in subject_acc.values() if a < 0.6):>3} subjects")


def print_comparison_table(all_stats: list[dict]) -> None:
    """Side-by-side accuracy comparison across all models."""
    print(f"\n{'═' * 65}")
    print(f"  CROSS-MODEL COMPARISON")
    print(f"{'═' * 65}")

    models = [s["model"] for s in all_stats]
    header = f"  {'Subject':<40}" + "".join(f" {m[:12]:>12}" for m in models)
    print(header)
    print(f"  {'─'*40}" + "".join(f" {'─'*12}" for _ in models))

    # Get all subjects across all models
    all_subjects = sorted(set(
        s for stats in all_stats for s in stats["subject_acc"]
    ))

    for subj in all_subjects:
        row = f"  {subj:<40}"
        for stats in all_stats:
            acc = stats["subject_acc"].get(subj)
            if acc is None:
                row += f"  {'—':>12}"
            else:
                row += f"  {acc*100:>11.1f}%"
        print(row)

    print(f"  {'─'*40}" + "".join(f" {'─'*12}" for _ in models))
    overall_row = f"  {'OVERALL':40}"
    for stats in all_stats:
        overall_row += f"  {stats['accuracy_overall']*100:>11.2f}%"
    print(overall_row)

    answered_row = f"  {'ON ANSWERED':40}"
    for stats in all_stats:
        answered_row += f"  {stats['accuracy_answered']*100:>11.2f}%"
    print(answered_row)

    print()
    print(f"  {'TOTAL QUESTIONS':40}" + "".join(f"  {s['total']:>12,}" for s in all_stats))
    print(f"  {'ERRORS':40}" + "".join(f"  {s['errors']:>12,}" for s in all_stats))
    print(f"  {'AVG LATENCY (s)':40}" + "".join(f"  {s['avg_latency']:>11.1f}s" for s in all_stats))
    print(f"  {'AVG COMPLETION TOKENS':40}" + "".join(f"  {s['avg_completion_tokens']:>12.0f}" for s in all_stats))


def save_combined(records: list[dict], model_key: str, lang: str, result_dir: str) -> str:
    """Write all records for one model to a single combined JSONL file."""
    fname = os.path.join(result_dir, f"mmlu_{lang}_{model_key}_combined.jsonl")
    with open(fname, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return fname


def save_report(report_lines: list[str], lang: str) -> str:
    """Save the full printed report as a text file."""
    analysis_dir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(analysis_dir, f"mmlu_{lang}_combined_report.txt")
    with open(fname, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    return fname


class _Tee:
    """Write to both stdout and a buffer simultaneously."""
    def __init__(self, real_stdout):
        import io
        self._real = real_stdout
        self._buf  = io.StringIO()
    def write(self, s):
        self._real.write(s)
        self._buf.write(s)
    def flush(self):
        self._real.flush()
    def getvalue(self):
        return self._buf.getvalue()


def main():
    import sys
    parser = argparse.ArgumentParser(description="Combine and analyse MMLU result batches")
    parser.add_argument("--lang", choices=["en", "de"], default="en",
                        help="Language: en (English MMLU) or de (German MMLU)")
    parser.add_argument("--save", action="store_true",
                        help="Write combined JSONL + report to disk")
    args = parser.parse_args()

    lang       = args.lang
    result_dir = RESULT_DIRS[lang]
    patterns   = FILE_PATTERNS[lang]

    # Capture all output for report file when --save is used
    tee = _Tee(sys.stdout)
    if args.save:
        sys.stdout = tee

    lang_label = "English" if lang == "en" else "German"
    print(f"\n{'═' * 65}")
    print(f"  MMLU {lang_label} — Combined Analysis")
    print(f"  Source dir: {result_dir}")
    print(f"{'═' * 65}")

    all_stats    = []
    report_lines = []

    for model_label, pattern in patterns.items():
        records, files_used = load_model_files(pattern, result_dir)

        if not records:
            print(f"\n  [{model_label}] No files found matching '{pattern}' — skipping")
            continue

        duplicates          = find_duplicates(records)
        records, n_removed  = deduplicate(records)
        stats               = analyse_model(records, model_label)
        stats["n_removed"]  = n_removed
        all_stats.append(stats)

        print_model_report(stats, files_used, duplicates)

        if args.save:
            # Map model label to short key
            key_map = {"gpt-oss-120b": "gptoss", "gemma-4-31B": "gemma4", "Qwen3.5-122B": "qwen3.5"}
            key = key_map.get(model_label, model_label.replace("-", "_"))
            out = save_combined(records, key, lang, result_dir)
            print(f"  → Saved combined file: {os.path.basename(out)}")

    if len(all_stats) > 1:
        print_comparison_table(all_stats)

    print(f"\n{'═' * 65}")
    total_q = sum(s["total"] for s in all_stats)
    print(f"  Total records analysed: {total_q:,}")
    print(f"  Run again after each batch to verify no duplicates crept in.")
    print(f"{'═' * 65}\n")

    if args.save:
        sys.stdout = tee._real
        out = save_report(tee.getvalue().splitlines(), lang)
        print(f"  → Report saved: {os.path.basename(out)}\n")


if __name__ == "__main__":
    main()
