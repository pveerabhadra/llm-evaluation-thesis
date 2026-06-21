from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict

from tqdm import tqdm

try:
    from bert_score import score as bert_score_fn
except ImportError:
    raise SystemExit("Missing dependency: pip3 install bert-score")

from transformers import AutoTokenizer as _AutoTokenizer

_scibert_tokenizer = _AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COMBINED_FILES = {
    "gemma-4-31B"  : os.path.join(SCRIPT_DIR, "scidqa_gemma4_combined.jsonl"),
    "gpt-oss-120b" : os.path.join(SCRIPT_DIR, "scidqa_gptoss_combined.jsonl"),
    "Qwen3.5-122B" : os.path.join(SCRIPT_DIR, "scidqa_qwen3.5_combined.jsonl"),
}

ALL_CONDITIONS = ["no_retrieval", "rag_top3", "rag_top5", "rag_dense", "long_context"]

OUTPUT_JSONL  = os.path.join(SCRIPT_DIR, "scidqa_bertscore.jsonl")
OUTPUT_REPORT = os.path.join(SCRIPT_DIR, "scidqa_bertscore_report.txt")

ROBERTA  = "roberta-large"            # matches SciDQA paper
SCIBERT  = "allenai/scibert_scivocab_uncased"  # domain-adapted for scientific text

MODEL_LABEL_MAP = {
    "gemma4"  : "gemma-4-31B",
    "gptoss"  : "gpt-oss-120b",
    "qwen3.5" : "Qwen3.5-122B",
}

def _avg(vals: list) -> float:
    clean = [v for v in vals if v is not None]
    return statistics.mean(clean) if clean else 0.0


def _truncate_to_512(texts: list[str]) -> list[str]:
    """Tokenize with SciBERT tokenizer and decode back, guaranteeing ≤512 tokens."""
    result = []
    for text in texts:
        if not text:
            result.append("")
            continue
        ids = _scibert_tokenizer(
            text, truncation=True, max_length=512,
            add_special_tokens=True, return_tensors=None
        )["input_ids"]
        result.append(_scibert_tokenizer.decode(ids, skip_special_tokens=True))
    return result


def run_bertscore(predictions: list[str], references: list[str],
                  model_type: str, batch_size: int = 32) -> tuple[list[float], list[float], list[float]]:
    """Returns (precision_list, recall_list, f1_list) as Python floats."""
    # SciBERT is BERT-base with hard 512 token limit — truncate via tokenizer
    is_bert_base = "scibert" in model_type.lower() or "bert-base" in model_type.lower()
    if is_bert_base:
        predictions = _truncate_to_512(predictions)
        references  = _truncate_to_512(references)
        batch_size  = min(batch_size, 16)

    P, R, F1 = bert_score_fn(
        predictions,
        references,
        model_type=model_type,
        lang="en",
        batch_size=batch_size,
        verbose=False,
    )
    return (
        [round(float(v), 4) for v in P],
        [round(float(v), 4) for v in R],
        [round(float(v), 4) for v in F1],
    )


def main():
    parser = argparse.ArgumentParser(description="BERTScore evaluation for SciDQA")
    parser.add_argument("--model",     choices=["gemma4","gptoss","qwen3.5"], default=None)
    parser.add_argument("--condition", choices=ALL_CONDITIONS, default=None)
    args = parser.parse_args()

    model_filter     = MODEL_LABEL_MAP.get(args.model) if args.model else None
    conditions_to_do = [args.condition] if args.condition else ALL_CONDITIONS

    already_scored: set[tuple] = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    already_scored.add((r["id"], r["condition"], r["model"]))
        if already_scored:
            print(f"Resuming — {len(already_scored)} records already scored.\n")

    write_fh = open(OUTPUT_JSONL, "a")

    # model_label → condition → list of F1 values
    rb_stats:  dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sci_stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    all_results: list[dict] = []

    try:
        for model_label, fpath in COMBINED_FILES.items():
            if model_filter and model_label != model_filter:
                continue
            if not os.path.exists(fpath):
                print(f"[{model_label}] File not found: {fpath} — skipping")
                continue

            print(f"\n{'═'*60}")
            print(f"  {model_label}")
            print(f"{'═'*60}")

            # Load records for this model, grouped by condition
            cond_records: dict[str, list[dict]] = defaultdict(list)
            with open(fpath) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("condition") not in conditions_to_do:
                        continue
                    if r.get("error") or not r.get("response_text") or not r.get("gold_answer"):
                        continue
                    key = (r["id"], r["condition"], r["model"])
                    if key in already_scored:
                        continue
                    cond_records[r["condition"]].append(r)

            for condition, records in cond_records.items():
                if not records:
                    continue

                predictions = [r["response_text"] for r in records]
                references  = [r["gold_answer"]   for r in records]

                print(f"\n  Condition: {condition}  ({len(records)} records)")

                print(f"    Running BERTScore with {ROBERTA}...")
                rb_P, rb_R, rb_F1 = run_bertscore(predictions, references, ROBERTA, batch_size=32)

                print(f"    Running BERTScore with {SCIBERT}...")
                sci_P, sci_R, sci_F1 = run_bertscore(predictions, references, SCIBERT, batch_size=16)

                for i, rec in enumerate(records):
                    result = {
                        "id"        : rec["id"],
                        "model"     : rec["model"],
                        "condition" : condition,
                        "pid"       : rec["pid"],
                        # RoBERTa scores (matches SciDQA paper)
                        "bertscore_P"   : rb_P[i],
                        "bertscore_R"   : rb_R[i],
                        "bertscore_F1"  : rb_F1[i],
                        # SciBERT scores (thesis extension)
                        "scibert_P"     : sci_P[i],
                        "scibert_R"     : sci_R[i],
                        "scibert_F1"    : sci_F1[i],
                        # Reference scores for cross-referencing
                        "rouge_avg"     : rec.get("rouge_avg"),
                        "no_answer_signal": rec.get("no_answer_signal"),
                    }
                    all_results.append(result)
                    rb_stats[model_label][condition].append(rb_F1[i])
                    sci_stats[model_label][condition].append(sci_F1[i])

                    if write_fh:
                        write_fh.write(json.dumps(result) + "\n")
                        write_fh.flush()

                print(f"    RoBERTa F1 avg: {_avg(rb_F1):.4f}  |  SciBERT F1 avg: {_avg(sci_F1):.4f}")

    finally:
        if write_fh:
            write_fh.close()

    report_lines: list[str] = []

    def p(s: str = "") -> None:
        report_lines.append(s)
        print(s)

    models_shown = [m for m in COMBINED_FILES if not model_filter or m == model_filter]

    p(f"\n{'═'*68}")
    p(f"  BERTScore Report — SciDQA")
    p(f"{'═'*68}")
    p(f"  RoBERTa = {ROBERTA}  (matches SciDQA paper)")
    p(f"  SciBERT = {SCIBERT}")
    p(f"  Metric reported: F1  (precision/recall available in JSONL)")
    p()

    for metric_label, stats in [("RoBERTa F1 (paper-comparable)", rb_stats),
                                  ("SciBERT F1  (thesis extension)", sci_stats)]:
        p(f"── {metric_label} ──────────────────────────────────")
        p(f"  {'Condition':<15}" + "".join(f" {m[:14]:>14}" for m in models_shown))
        p(f"  {'─'*15}" + "".join(f" {'─'*14}" for _ in models_shown))
        for cond in conditions_to_do:
            row = f"  {cond:<15}"
            for m in models_shown:
                vals = stats[m].get(cond, [])
                row += f" {_avg(vals):>14.4f}" if vals else f" {'—':>14}"
            p(row)
        p()

    # Cross-metric comparison: does SciBERT change the rankings?
    p(f"── SciBERT vs RoBERTa delta (SciBERT − RoBERTa) ──────────────")
    p(f"  Positive = SciBERT gives higher score (domain vocab helps)")
    p(f"  Negative = SciBERT gives lower score  (stricter about scientific accuracy)")
    p()
    p(f"  {'Condition':<15}" + "".join(f" {m[:14]:>14}" for m in models_shown))
    p(f"  {'─'*15}" + "".join(f" {'─'*14}" for _ in models_shown))
    for cond in conditions_to_do:
        row = f"  {cond:<15}"
        for m in models_shown:
            rb  = _avg(rb_stats[m].get(cond,  []))
            sci = _avg(sci_stats[m].get(cond, []))
            delta = sci - rb
            sign  = "+" if delta >= 0 else ""
            row += f" {sign}{delta:>13.4f}" if rb > 0 else f" {'—':>14}"
        p(row)

    p()
    p(f"  Total records scored: {len(all_results):,}")

    with open(OUTPUT_REPORT, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\n  → Saved JSONL : {os.path.basename(OUTPUT_JSONL)}")
    print(f"  → Saved report: {os.path.basename(OUTPUT_REPORT)}")


if __name__ == "__main__":
    main()
