"""
thesis_tables.py
----------------
Generates all thesis-ready tables from combined evaluation data.

Tables produced:
  Table 1 — SciDQA main results: ROUGE × BERTScore × NLI × (LLM-judge where available)
  Table 2 — SciDQA RAG delta over no_retrieval
  Table 3 — SciDQA mismatch / hallucination analysis
  Table 4 — MMLU English + German accuracy summary
  Table 5 — Efficiency (latency + tokens per model × task)

Sources used:
  analysis/scidqa_{model}_combined.jsonl   — ROUGE, grounding, no-answer, latency
  analysis/scidqa_bertscore.jsonl          — BERTScore F1 (RoBERTa + SciBERT)
  analysis/scidqa_nli_faithfulness.jsonl   — NLI faithfulness (RAG conditions)
  analysis/scidqa_llm_judge.jsonl          — ALS score (Gemma-4 only; partial)
  SciDQA/scidqa_mismatch_*_summary.txt     — mismatch/hallucination summaries
  MMLU/mmlu_en_*_combined.jsonl            — English MMLU
  MMLU German/mmlu_de_*_combined.jsonl     — German MMLU

Usage:
  cd analysis && python3 thesis_tables.py
  cd analysis && python3 thesis_tables.py --save
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict

THESIS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_KEYS  = ["gemma4",  "gptoss",  "qwen3.5"]
MODEL_NAMES = {
    "gemma4" : "gemma-4-31B",
    "gptoss" : "gpt-oss-120b",
    "qwen3.5": "Qwen3.5-122B",
}
MODEL_FILE_IDS = {
    "gemma4" : "RedHatAI/gemma-4-31B-it-FP8-Dynamic",
    "gptoss" : "openai/gpt-oss-120b",
    "qwen3.5": "Qwen/Qwen3.5-122B-A10B-FP8",
}

CONDITIONS  = ["no_retrieval", "rag_top3", "rag_top5", "rag_dense", "long_context"]
COND_LABELS = {
    "no_retrieval": "No Retrieval",
    "rag_top3"    : "RAG top-3",
    "rag_top5"    : "RAG top-5",
    "rag_dense"   : "RAG dense",
    "long_context": "Long Context",
}
RAG_CONDITIONS = ["rag_top3", "rag_top5", "rag_dense"]

W = 88


# ── Loaders ─────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"  [WARN] Missing: {path}", file=sys.stderr)
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_scidqa_combined(model_key: str) -> list[dict]:
    return load_jsonl(os.path.join(ANALYSIS_DIR, f"scidqa_{model_key}_combined.jsonl"))


def load_mmlu_combined(lang: str, model_key: str) -> list[dict]:
    dirs = {"en": os.path.join(THESIS_ROOT, "MMLU"),
            "de": os.path.join(THESIS_ROOT, "MMLU German")}
    return load_jsonl(os.path.join(dirs[lang], f"mmlu_{lang}_{model_key}_combined.jsonl"))


# ── Aggregators ──────────────────────────────────────────────────────────────────

def scidqa_rouge_stats(records: list[dict]) -> dict[str, dict]:
    """Per-condition ROUGE and related stats. Excludes error records."""
    out: dict[str, dict] = {c: defaultdict(list) for c in CONDITIONS}
    for r in records:
        if r.get("error"):
            continue
        rt = r.get("response_text") or ""
        if not rt.strip():
            continue
        c = r.get("condition")
        if c not in out:
            continue
        for metric in ["rouge_1", "rouge_2", "rouge_l", "rouge_avg",
                       "ngram_grounding_score", "no_answer_signal",
                       "latency_s", "completion_tokens", "total_tokens",
                       "response_length_chars"]:
            v = r.get(metric)
            if v is not None:
                out[c][metric].append(float(v))
    result = {}
    for c, d in out.items():
        result[c] = {k: statistics.mean(v) if v else 0.0 for k, v in d.items()}
        result[c]["n"] = len(d.get("rouge_avg", []))
    return result


def bertscore_stats(records: list[dict]) -> dict[str, dict[str, dict]]:
    """
    Returns {model_id: {condition: {bertscore_f1, scibert_f1, n}}}.
    model field in the bertscore JSONL uses short key (gemma4/gptoss/qwen3.5).
    """
    data: dict[str, dict[str, dict]] = {k: {c: defaultdict(list) for c in CONDITIONS} for k in MODEL_KEYS}
    for r in records:
        mk = r.get("model")           # short key like "gemma4"
        if mk not in data:
            # try matching by full model id
            for k, fid in MODEL_FILE_IDS.items():
                if fid in (mk or ""):
                    mk = k
                    break
        if mk not in data:
            continue
        cond = r.get("condition")
        if cond not in CONDITIONS:
            continue
        for field in ["bertscore_F1", "scibert_F1"]:
            v = r.get(field)
            if v is not None:
                data[mk][cond][field].append(float(v))
    result: dict[str, dict[str, dict]] = {}
    for mk in MODEL_KEYS:
        result[mk] = {}
        for c in CONDITIONS:
            d = data[mk][c]
            result[mk][c] = {
                "bertscore_f1": statistics.mean(d["bertscore_F1"]) if d["bertscore_F1"] else None,
                "scibert_f1":   statistics.mean(d["scibert_F1"])   if d["scibert_F1"]   else None,
                "n": len(d["bertscore_F1"]),
            }
    return result


def nli_stats(records: list[dict]) -> dict[str, dict[str, dict]]:
    """Returns {model_id: {condition: {nli_faithfulness, n}}}."""
    data: dict[str, dict[str, list]] = {k: {c: [] for c in RAG_CONDITIONS} for k in MODEL_KEYS}
    for r in records:
        # NLI JSONL uses full model ID in 'model' field
        raw_model = r.get("model", "")
        mk = None
        for k, fid in MODEL_FILE_IDS.items():
            if fid == raw_model:
                mk = k
                break
        if mk is None:
            continue
        cond = r.get("condition")
        if cond not in RAG_CONDITIONS:
            continue
        v = r.get("nli_faithfulness")
        if v is not None:
            data[mk][cond].append(float(v))
    result: dict[str, dict[str, dict]] = {}
    for mk in MODEL_KEYS:
        result[mk] = {}
        for c in RAG_CONDITIONS:
            vals = data[mk][c]
            result[mk][c] = {"nli": statistics.mean(vals) if vals else None, "n": len(vals)}
    return result


def als_stats(records: list[dict]) -> dict[str, dict[str, dict]]:
    """
    ALS = Average LLM Score.
    Currently only Gemma-4 answers were judged. Returns {answer_model: {condition: als}}.
    ALS is mean of dimension scores (1–10 scale), averaged across judges.
    """
    data: dict[str, dict[str, list]] = {k: {c: [] for c in CONDITIONS} for k in MODEL_KEYS}
    for r in records:
        if r.get("error"):
            continue
        ak = r.get("answer_model_key")
        if ak not in data:
            continue
        cond = r.get("condition")
        if cond not in CONDITIONS:
            continue
        # overall field is the mean of 4 dimensions
        overall = r.get("overall")
        if overall is not None:
            data[ak][cond].append(float(overall))
    result: dict[str, dict[str, dict]] = {}
    for mk in MODEL_KEYS:
        result[mk] = {}
        for c in CONDITIONS:
            vals = data[mk][c]
            result[mk][c] = {
                "als": statistics.mean(vals) if vals else None,
                "als_100": statistics.mean(vals) / 10 * 100 if vals else None,
                "n": len(vals),
            }
    return result


def mmlu_subject_accuracy(records: list[dict]) -> dict[str, float]:
    answered = [r for r in records if not r.get("error")]
    correct  = [r for r in answered if r.get("is_correct")]
    return {
        "total":    len(records),
        "answered": len(answered),
        "correct":  len(correct),
        "acc_overall": len(correct) / len(records) if records else 0,
        "acc_answered": len(correct) / len(answered) if answered else 0,
        "avg_latency": statistics.mean([r["latency_s"] for r in records if r.get("latency_s")] or [0]),
        "avg_tokens": statistics.mean([r["completion_tokens"] for r in records if r.get("completion_tokens")] or [0]),
    }


# ── Mismatch summary parser ──────────────────────────────────────────────────────

def parse_mismatch_summary(model_key: str) -> dict:
    """
    Reads scidqa_mismatch_{model}_*_summary.txt and extracts key stats.
    Returns dict with source attribution counts and ROUGE.
    """
    import glob as _glob
    pattern = os.path.join(THESIS_ROOT, "SciDQA",
                           f"scidqa_mismatch_{model_key}_*summary.txt")
    files = _glob.glob(pattern)
    if not files:
        return {}
    with open(files[0]) as f:
        text = f.read()

    def extract(label: str) -> float | None:
        m = re.search(rf"{re.escape(label)}\s*[:\s]+([0-9.]+)", text)
        return float(m.group(1)) if m else None

    def extract_count(label: str) -> int | None:
        m = re.search(rf"{re.escape(label)}\s*:\s*(\d+)", text)
        return int(m.group(1)) if m else None

    def extract_pct(label: str) -> float | None:
        m = re.search(rf"{re.escape(label)}\s*:\s*\d+\s+\(([0-9.]+)%\)", text)
        return float(m.group(1)) if m else None

    return {
        "total":             extract_count("Total records"),
        "no_answer_pct":     extract_pct("No-answer"),
        "excerpt_pct":       extract_pct("EXCERPT"),
        "training_mem_pct":  extract_pct("TRAINING MEMORY"),
        "not_found_pct":     extract_pct("NOT FOUND"),
        "rouge_avg":         extract("Avg ROUGE-avg"),
    }


# ── Formatting helpers ───────────────────────────────────────────────────────────

def pct(v: float | None, decimals: int = 1) -> str:
    return f"{v:.{decimals}f}" if v is not None else " N/A"


def header(title: str) -> None:
    print(f"\n{'═' * W}")
    print(f"  {title}")
    print(f"{'═' * W}")


def hline(widths: list[int]) -> str:
    return "  " + "  ".join("─" * w for w in widths)


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="Write tables to analysis/thesis_tables_report.txt")
    args = parser.parse_args()

    class _Tee:
        def __init__(self, real):
            import io
            self._real = real
            self._buf = io.StringIO()
        def write(self, s):
            self._real.write(s)
            self._buf.write(s)
        def flush(self):
            self._real.flush()
        def getvalue(self):
            return self._buf.getvalue()

    tee = _Tee(sys.stdout)
    if args.save:
        sys.stdout = tee

    # ── Load all data ──────────────────────────────────────────────────────────
    print("  Loading data...", file=sys.stderr)

    scidqa_raw = {mk: load_scidqa_combined(mk) for mk in MODEL_KEYS}
    bs_records = load_jsonl(os.path.join(ANALYSIS_DIR, "scidqa_bertscore.jsonl"))
    nli_records = load_jsonl(os.path.join(ANALYSIS_DIR, "scidqa_nli_faithfulness.jsonl"))
    lj_records = load_jsonl(os.path.join(ANALYSIS_DIR, "scidqa_llm_judge.jsonl"))

    rouge_by_model = {mk: scidqa_rouge_stats(scidqa_raw[mk]) for mk in MODEL_KEYS}
    bs_by_model    = bertscore_stats(bs_records)
    nli_by_model   = nli_stats(nli_records)
    als_by_model   = als_stats(lj_records)
    mismatch_by_model = {mk: parse_mismatch_summary(mk) for mk in MODEL_KEYS}

    mmlu_en = {mk: mmlu_subject_accuracy(load_mmlu_combined("en", mk)) for mk in MODEL_KEYS}
    mmlu_de = {mk: mmlu_subject_accuracy(load_mmlu_combined("de", mk)) for mk in MODEL_KEYS}

    print("  Generating tables...\n", file=sys.stderr)

    # ════════════════════════════════════════════════════════════════════════════
    # TABLE 1 — SciDQA Main Results
    # ════════════════════════════════════════════════════════════════════════════
    header("TABLE 1 — SciDQA Main Results (2,937 questions × 5 conditions)")
    print("""
  Metrics:
    R-avg   = average of ROUGE-1, ROUGE-2, ROUGE-L
    BS-R    = BERTScore F1 (RoBERTa-large; paper-comparable)
    BS-Sci  = BERTScore F1 (SciBERT; domain-adapted)
    NLI     = NLI faithfulness (RAG/long-context conditions only)
    ALS     = Average LLM Score 0–100 (Gemma-4 judged by GPT-OSS + Qwen; others N/A)
    No-Ans% = fraction of responses triggering no-answer phrases
""")

    col_cond = 14
    col_m    = 8
    header_row = f"  {'Condition':<{col_cond}}"
    for mk in MODEL_KEYS:
        header_row += f"  {MODEL_NAMES[mk]:>26}"
    print(header_row)

    metric_row = f"  {'':<{col_cond}}"
    for _ in MODEL_KEYS:
        metric_row += f"  {'R-avg':>4} {'BS-R':>5} {'BS-Sci':>6} {'NLI':>5} {'ALS':>4} {'No-Ans%':>7}"
    print(metric_row)
    print(f"  {'─'*col_cond}" + ("  " + "─"*4 + " " + "─"*5 + " " + "─"*6 + " " + "─"*5 + " " + "─"*4 + " " + "─"*7) * 3)

    for cond in CONDITIONS:
        row = f"  {COND_LABELS[cond]:<{col_cond}}"
        for mk in MODEL_KEYS:
            r   = rouge_by_model[mk].get(cond, {})
            bs  = bs_by_model[mk].get(cond, {})
            nli = nli_by_model[mk].get(cond, {}) if cond in RAG_CONDITIONS else {}
            als = als_by_model[mk].get(cond, {})

            r_avg   = f"{r.get('rouge_avg', 0):.3f}"
            bs_r    = f"{bs.get('bertscore_f1') or 0:.3f}" if bs.get("bertscore_f1") else "  N/A"
            bs_sci  = f"{bs.get('scibert_f1') or 0:.3f}"   if bs.get("scibert_f1") else "  N/A"
            nli_v   = f"{nli.get('nli') or 0:.3f}"          if nli.get("nli") else "  N/A"
            als_v   = f"{als.get('als_100') or 0:.1f}"      if als.get("als") else " N/A"
            no_ans  = f"{r.get('no_answer_signal', 0)*100:.1f}%"

            row += f"  {r_avg:>4} {bs_r:>5} {bs_sci:>6} {nli_v:>5} {als_v:>4} {no_ans:>7}"
        print(row)

    print(f"\n  Note: NLI faithfulness only applies to RAG conditions (context grounding check).")
    print(f"  Note: ALS currently available for Gemma-4 only (GPT-OSS and Qwen judging complete).")
    print(f"        To complete ALS for all models, run: python3 analysis/llm_judge.py")

    # ════════════════════════════════════════════════════════════════════════════
    # TABLE 1b — Granular ROUGE breakdown
    # ════════════════════════════════════════════════════════════════════════════
    header("TABLE 1b — SciDQA ROUGE Breakdown (R-1 / R-2 / R-L / R-avg)")
    col_c = 14
    hdr = f"  {'Condition':<{col_c}}"
    for mk in MODEL_KEYS:
        hdr += f"  {MODEL_NAMES[mk]:>22}"
    print(hdr)
    sub = f"  {'':<{col_c}}"
    for _ in MODEL_KEYS:
        sub += f"  {'R-1':>4} {'R-2':>4} {'R-L':>4} {'R-avg':>5}"
    print(sub)
    print(f"  {'─'*col_c}" + ("  " + "─"*4 + " " + "─"*4 + " " + "─"*4 + " " + "─"*5) * 3)

    for cond in CONDITIONS:
        row = f"  {COND_LABELS[cond]:<{col_c}}"
        for mk in MODEL_KEYS:
            r = rouge_by_model[mk].get(cond, {})
            row += (f"  {r.get('rouge_1', 0):.3f}"
                    f" {r.get('rouge_2', 0):.3f}"
                    f" {r.get('rouge_l', 0):.3f}"
                    f" {r.get('rouge_avg', 0):.4f}")
        print(row)

    # ════════════════════════════════════════════════════════════════════════════
    # TABLE 2 — RAG Delta over No-Retrieval
    # ════════════════════════════════════════════════════════════════════════════
    header("TABLE 2 — SciDQA RAG Benefit: Delta over No-Retrieval Baseline")
    print("""
  Shows improvement each retrieval condition provides over no_retrieval.
  Positive = better with retrieval. Computed for ROUGE-avg and BERTScore F1.
""")
    hdr = f"  {'Condition':<14}"
    for mk in MODEL_KEYS:
        hdr += f"  {MODEL_NAMES[mk]:>26}"
    print(hdr)
    sub = f"  {'':<14}"
    for _ in MODEL_KEYS:
        sub += f"  {'ΔR-avg':>6} {'ΔBS-R':>6} {'ΔNLI':>6}"
    print(sub)
    print(f"  {'─'*14}" + ("  " + "─"*6 + " " + "─"*6 + " " + "─"*6) * 3)

    for cond in CONDITIONS:
        if cond == "no_retrieval":
            continue
        row = f"  {COND_LABELS[cond]:<14}"
        for mk in MODEL_KEYS:
            base_r   = rouge_by_model[mk].get("no_retrieval", {}).get("rouge_avg", 0)
            cond_r   = rouge_by_model[mk].get(cond, {}).get("rouge_avg", 0)
            base_bs  = bs_by_model[mk].get("no_retrieval", {}).get("bertscore_f1") or 0
            cond_bs  = bs_by_model[mk].get(cond, {}).get("bertscore_f1") or 0
            delta_r  = cond_r  - base_r
            delta_bs = cond_bs - base_bs
            nli_v    = nli_by_model[mk].get(cond, {}).get("nli")
            nli_str  = f"{nli_v:+.3f}" if nli_v is not None else "   N/A"
            row += f"  {delta_r:+6.3f} {delta_bs:+6.3f} {nli_str:>6}"
        print(row)

    # ════════════════════════════════════════════════════════════════════════════
    # TABLE 3 — Mismatch / Hallucination Analysis
    # ════════════════════════════════════════════════════════════════════════════
    header("TABLE 3 — SciDQA Mismatch Condition (Wrong-Paper RAG)")
    print("""
  Model is given BM25 top-3 chunks from a DIFFERENT paper than the question requires.
  Measures: does the model faithfully refuse, use the wrong context, or use memory?

  Source attribution categories (mutually exclusive):
    EXCERPT         — model answered using the wrong paper's text (hallucination)
    TRAINING MEMORY — model ignored context and answered from training data
    NOT FOUND       — model correctly stated the answer was not in the provided context
    No-Answer %     — fraction of responses with explicit abstention phrases
""")

    col_m2 = 18
    print(f"  {'Metric':<{col_m2}}" + "".join(f"  {MODEL_NAMES[mk]:>14}" for mk in MODEL_KEYS))
    print(f"  {'─'*col_m2}" + "".join(f"  {'─'*14}" for _ in MODEL_KEYS))

    for label, key in [
        ("Questions answered", "total"),
        ("EXCERPT %",          "excerpt_pct"),
        ("TRAINING MEMORY %",  "training_mem_pct"),
        ("NOT FOUND %",        "not_found_pct"),
        ("No-Answer %",        "no_answer_pct"),
        ("ROUGE-avg",          "rouge_avg"),
    ]:
        row = f"  {label:<{col_m2}}"
        for mk in MODEL_KEYS:
            v = mismatch_by_model.get(mk, {}).get(key)
            if v is None:
                row += f"  {'N/A':>14}"
            elif key == "total":
                row += f"  {int(v):>14,}"
            elif key == "rouge_avg":
                row += f"  {v:>14.4f}"
            else:
                row += f"  {v:>13.1f}%"
        print(row)

    # Compare mismatch ROUGE vs rag_top3 ROUGE
    print(f"\n  ── ROUGE comparison: mismatch vs rag_top3 (correct paper) ──")
    print(f"  {'Metric':<{col_m2}}" + "".join(f"  {MODEL_NAMES[mk]:>14}" for mk in MODEL_KEYS))
    for label, source, cond in [
        ("rag_top3 ROUGE-avg",   "rouge", "rag_top3"),
        ("mismatch ROUGE-avg",   "mismatch", None),
        ("Delta (mismatch−rag)", "delta", None),
    ]:
        row = f"  {label:<{col_m2}}"
        for mk in MODEL_KEYS:
            if source == "rouge":
                v = rouge_by_model[mk].get("rag_top3", {}).get("rouge_avg", 0)
                row += f"  {v:>14.4f}"
            elif source == "mismatch":
                v = mismatch_by_model.get(mk, {}).get("rouge_avg")
                row += f"  {v:>14.4f}" if v is not None else f"  {'N/A':>14}"
            else:
                rag_v  = rouge_by_model[mk].get("rag_top3", {}).get("rouge_avg", 0)
                mis_v  = mismatch_by_model.get(mk, {}).get("rouge_avg")
                if mis_v is not None:
                    row += f"  {mis_v - rag_v:>+14.4f}"
                else:
                    row += f"  {'N/A':>14}"
        print(row)

    print(f"""
  Interpretation:
    Large negative delta (mismatch − rag_top3) → model heavily relies on retrieved context
    Small delta → model draws mainly from training memory regardless of context
    HIGH NOT FOUND % → faithful behaviour (ideal for RAG trustworthiness)
    HIGH EXCERPT %   → risk of context-based hallucination""")

    # ════════════════════════════════════════════════════════════════════════════
    # TABLE 4 — MMLU Summary
    # ════════════════════════════════════════════════════════════════════════════
    header("TABLE 4 — MMLU Accuracy Summary (English and German)")
    print(f"""
  All 57 MMLU subjects, ~14k questions per model per language.
  Accuracy = correct / total (including errors).
""")
    col_l = 22
    print(f"  {'Metric':<{col_l}}" + "".join(f"  {MODEL_NAMES[mk]:>14}" for mk in MODEL_KEYS))
    print(f"  {'─'*col_l}" + "".join(f"  {'─'*14}" for _ in MODEL_KEYS))

    for lang_label, stats in [("English accuracy", mmlu_en), ("German accuracy", mmlu_de)]:
        row = f"  {lang_label:<{col_l}}"
        for mk in MODEL_KEYS:
            row += f"  {stats[mk]['acc_overall']*100:>13.2f}%"
        print(row)

    en_de_drops = {mk: (mmlu_en[mk]['acc_overall'] - mmlu_de[mk]['acc_overall']) * 100 for mk in MODEL_KEYS}
    drop_row = f"  {'EN → DE drop':<{col_l}}"
    for mk in MODEL_KEYS:
        drop_row += f"  {en_de_drops[mk]:>+13.2f}%"
    print(drop_row)

    print(f"  {'─'*col_l}" + "".join(f"  {'─'*14}" for _ in MODEL_KEYS))

    for label, key in [("Total questions (EN)", "en"), ("Total questions (DE)", "de")]:
        row = f"  {label:<{col_l}}"
        for mk in MODEL_KEYS:
            src = mmlu_en if key == "en" else mmlu_de
            row += f"  {src[mk]['total']:>14,}"
        print(row)

    # ════════════════════════════════════════════════════════════════════════════
    # TABLE 5 — Efficiency
    # ════════════════════════════════════════════════════════════════════════════
    header("TABLE 5 — Efficiency: Average Latency and Token Usage")
    print(f"""
  Latency = wall-clock time per question (seconds).
  Tokens  = completion tokens per question (includes thinking tokens for GPT-OSS/Qwen).
""")

    # SciDQA efficiency
    print(f"  ── SciDQA ──")
    col_t = 16
    print(f"  {'Condition':<14}" + "".join(f"  {MODEL_NAMES[mk]:>22}" for mk in MODEL_KEYS))
    print(f"  {'':<14}" + "".join(f"  {'Latency':>8} {'Tokens':>8}" for _ in MODEL_KEYS))
    print(f"  {'─'*14}" + ("  " + "─"*8 + " " + "─"*8) * 3)
    for cond in CONDITIONS:
        row = f"  {COND_LABELS[cond]:<14}"
        for mk in MODEL_KEYS:
            r = rouge_by_model[mk].get(cond, {})
            lat = r.get("latency_s", 0)
            tok = r.get("completion_tokens", 0)
            row += f"  {lat:>7.1f}s {tok:>8.0f}"
        print(row)

    print(f"\n  Note: Tokens = completion tokens only (model output).")
    print(f"  Long Context prompt tokens are large (~26k–28k) due to full paper text;"
          f" these are not shown above.")

    # MMLU efficiency
    print(f"\n  ── MMLU ──")
    print(f"  {'Task':<20}" + "".join(f"  {MODEL_NAMES[mk]:>22}" for mk in MODEL_KEYS))
    print(f"  {'':<20}" + "".join(f"  {'Latency':>8} {'Tokens':>8}" for _ in MODEL_KEYS))
    print(f"  {'─'*20}" + ("  " + "─"*8 + " " + "─"*8) * 3)
    for task_label, stats in [("MMLU English", mmlu_en), ("MMLU German", mmlu_de)]:
        row = f"  {task_label:<20}"
        for mk in MODEL_KEYS:
            row += f"  {stats[mk]['avg_latency']:>7.1f}s {stats[mk]['avg_tokens']:>8.0f}"
        print(row)

    print(f"\n  Note: GPT-OSS and Qwen are thinking models — completion tokens include")
    print(f"  internal chain-of-thought reasoning before the final answer.")

    print(f"\n{'═' * W}")
    print(f"  End of thesis tables. Generated from combined JSONL files in analysis/.")
    print(f"{'═' * W}\n")

    if args.save:
        sys.stdout = tee._real
        out = os.path.join(ANALYSIS_DIR, "thesis_tables_report.txt")
        with open(out, "w") as f:
            f.write(tee.getvalue())
        print(f"  → Report saved: thesis_tables_report.txt\n")


if __name__ == "__main__":
    main()
