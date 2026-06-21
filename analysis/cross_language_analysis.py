from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from scipy import stats as scipy_stats

THESIS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MMLU_CATEGORIES = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "formal_logic",
        "high_school_biology", "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios",
        "philosophy", "prehistory", "professional_law", "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_psychology", "human_sexuality", "political_science",
        "professional_psychology", "public_relations", "security_studies",
        "sociology", "us_foreign_policy",
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition",
        "professional_accounting", "professional_medicine", "virology",
    ],
}

def subject_to_category(subject: str) -> str:
    for cat, subjects in MMLU_CATEGORIES.items():
        if subject in subjects:
            return cat
    return "Other"


MODEL_KEYS = {
    "gptoss":  "gpt-oss-120b",
    "gemma4":  "gemma-4-31B",
    "qwen3.5": "Qwen3.5-122B",
}

def load_combined(lang: str, model_key: str) -> list[dict]:
    """Load a combined MMLU JSONL file."""
    dirs = {
        "en": os.path.join(THESIS_ROOT, "MMLU"),
        "de": os.path.join(THESIS_ROOT, "MMLU German"),
    }
    path = os.path.join(dirs[lang], f"mmlu_{lang}_{model_key}_combined.jsonl")
    if not os.path.exists(path):
        print(f"  [WARN] File not found: {path}")
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def subject_accuracy(records: list[dict]) -> dict[str, dict]:
    """
    Returns {subject: {"correct": int, "total": int, "acc": float}}.
    Only includes answered records (no error).
    """
    acc: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        if r.get("error"):
            continue
        subj = r.get("subject", "unknown")
        acc[subj]["total"] += 1
        if r.get("is_correct"):
            acc[subj]["correct"] += 1
    result = {}
    for subj, d in acc.items():
        result[subj] = {
            "correct": d["correct"],
            "total":   d["total"],
            "acc":     d["correct"] / d["total"] if d["total"] else 0.0,
        }
    return result


def mcnemar_test(records_a: list[dict], records_b: list[dict], key_fn) -> dict:
    """
    Perform McNemar's test comparing model A vs model B on the same questions.

    key_fn: function that extracts a match key from a record
            (e.g. question text for EN, or (subject, orig_q_id) for DE)

    Returns dict with n_b01, n_b10, statistic, p_value, significant (α=0.05).

    McNemar's contingency:
      b01 = A wrong, B correct   (B better)
      b10 = A correct, B wrong   (A better)
    χ² = (|b01 - b10| - 1)² / (b01 + b10)   [continuity-corrected]
    Under H0: both models perform equally well.
    """
    map_a = {key_fn(r): r.get("is_correct", False) for r in records_a if not r.get("error")}
    map_b = {key_fn(r): r.get("is_correct", False) for r in records_b if not r.get("error")}
    shared = set(map_a) & set(map_b)

    b00 = b01 = b10 = b11 = 0
    for k in shared:
        a_ok, b_ok = map_a[k], map_b[k]
        if     a_ok and     b_ok: b11 += 1
        if     a_ok and not b_ok: b10 += 1
        if not a_ok and     b_ok: b01 += 1
        if not a_ok and not b_ok: b00 += 1

    n = b01 + b10
    if n == 0:
        return {"b01": b01, "b10": b10, "n_discordant": 0,
                "chi2": 0.0, "p_value": 1.0, "significant": False,
                "n_shared": len(shared)}

    # Continuity-corrected McNemar
    chi2 = (abs(b01 - b10) - 1.0) ** 2 / n
    p = 1 - scipy_stats.chi2.cdf(chi2, df=1)
    return {
        "b01": b01, "b10": b10, "n_discordant": n, "n_shared": len(shared),
        "chi2": round(chi2, 3), "p_value": round(p, 4),
        "significant": p < 0.05,
    }


def bootstrap_ci(records: list[dict], n_boot: int = 2000, alpha: float = 0.05) -> dict:
    """95% bootstrap CI on overall accuracy (answered records only)."""
    import random
    answered = [r for r in records if not r.get("error")]
    if not answered:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    n = len(answered)
    correct_flags = [int(r.get("is_correct", False)) for r in answered]
    boot_means = []
    for _ in range(n_boot):
        sample = random.choices(correct_flags, k=n)
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(alpha / 2 * n_boot)]
    hi = boot_means[int((1 - alpha / 2) * n_boot)]
    obs_mean = sum(correct_flags) / n
    return {"mean": obs_mean, "ci_lo": lo, "ci_hi": hi}


def paired_ttest_drops(drops_a: list[float], drops_b: list[float]) -> dict:
    """
    Test whether model A's EN→DE accuracy drop significantly differs from model B's.
    Uses paired t-test (same 57 subjects for both models).
    """
    paired = [(a, b) for a, b in zip(drops_a, drops_b) if a is not None and b is not None]
    if len(paired) < 3:
        return {"t": None, "p": None, "significant": False}
    diffs = [a - b for a, b in paired]
    t, p = scipy_stats.ttest_1samp(diffs, 0)
    return {"t": round(t, 3), "p": round(p, 4), "significant": p < 0.05}


W = 80

def header(title: str) -> None:
    print(f"\n{'═' * W}")
    print(f"  {title}")
    print(f"{'═' * W}")


def subheader(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, W - len(title) - 4)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="Write report to analysis/mmlu_crosslang_report.txt")
    args = parser.parse_args()

    # Capture output for optional save
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

    MODEL_ORDER = [("gptoss", "gpt-oss-120b"), ("gemma4", "gemma-4-31B"), ("qwen3.5", "Qwen3.5-122B")]
    MODEL_LABELS = [label for _, label in MODEL_ORDER]

    en_records: dict[str, list[dict]] = {}
    de_records: dict[str, list[dict]] = {}
    for key, label in MODEL_ORDER:
        en_records[key] = load_combined("en", key)
        de_records[key] = load_combined("de", key)

    en_acc: dict[str, dict[str, dict]] = {k: subject_accuracy(en_records[k]) for k, _ in MODEL_ORDER}
    de_acc: dict[str, dict[str, dict]] = {k: subject_accuracy(de_records[k]) for k, _ in MODEL_ORDER}

    # Union of all subjects — sorted by category then alphabetically
    all_subject_set = set(
        s for k, _ in MODEL_ORDER
        for d in [en_acc[k], de_acc[k]]
        for s in d
    )
    # Build category from DE records first, fallback to static map
    _subj_cat_tmp: dict[str, str] = {}
    for key, _ in MODEL_ORDER:
        for r in de_records[key]:
            s = r.get("subject")
            c = r.get("mmlu_category")
            if s and c:
                _subj_cat_tmp[s] = c
    for s in all_subject_set:
        if s not in _subj_cat_tmp:
            _subj_cat_tmp[s] = subject_to_category(s)

    cat_order = ["STEM", "Humanities", "Social Sciences", "Other"]
    all_subjects = sorted(
        all_subject_set,
        key=lambda s: (cat_order.index(_subj_cat_tmp.get(s, "Other")), s)
    )

    # Subject → category (use DE records which have mmlu_category field)
    subj_cat: dict[str, str] = {}
    for key, _ in MODEL_ORDER:
        for r in de_records[key]:
            s = r.get("subject")
            c = r.get("mmlu_category")
            if s and c:
                subj_cat[s] = c
    # Fill any gaps from the static map
    for s in all_subjects:
        if s not in subj_cat:
            subj_cat[s] = subject_to_category(s)

    header("MMLU Cross-Language Analysis — English vs German Accuracy")

    col_w = 44
    model_col_w = 14
    col_labels = "".join(f"  {'EN':>5} {'DE':>5} {'Δ':>5}" for _ in MODEL_ORDER)
    print(f"\n  {'Subject':<{col_w}}" + col_labels)
    print(f"  {'Category':<{col_w}}" + "".join(f"  {'─'*5} {'─'*5} {'─'*5}" for _ in MODEL_ORDER))

    # Group by category
    cats = ["STEM", "Humanities", "Social Sciences", "Other"]
    cat_en_accs: dict[str, dict[str, list[float]]] = {c: {k: [] for k, _ in MODEL_ORDER} for c in cats}
    cat_de_accs: dict[str, dict[str, list[float]]] = {c: {k: [] for k, _ in MODEL_ORDER} for c in cats}
    subject_drops: dict[str, list[float]] = {k: [] for k, _ in MODEL_ORDER}
    subject_en_list: dict[str, list[float]] = {k: [] for k, _ in MODEL_ORDER}
    subject_de_list: dict[str, list[float]] = {k: [] for k, _ in MODEL_ORDER}

    prev_cat = None
    for subj in all_subjects:
        cat = subj_cat.get(subj, "Other")
        if cat != prev_cat:
            print(f"\n  ── {cat} ──")
            prev_cat = cat

        row = f"  {subj:<{col_w}}"
        for key, _ in MODEL_ORDER:
            en_a = en_acc[key].get(subj, {}).get("acc")
            de_a = de_acc[key].get(subj, {}).get("acc")
            drop = (en_a - de_a) if en_a is not None and de_a is not None else None

            en_str = f"{en_a*100:5.1f}%" if en_a is not None else "  N/A"
            de_str = f"{de_a*100:5.1f}%" if de_a is not None else "  N/A"
            drop_str = f"{drop*100:+5.1f}%" if drop is not None else "   N/A"

            row += f"  {en_str} {de_str} {drop_str}"

            if en_a is not None and de_a is not None:
                cat_en_accs[cat][key].append(en_a)
                cat_de_accs[cat][key].append(de_a)
                subject_drops[key].append(drop)
                subject_en_list[key].append(en_a)
                subject_de_list[key].append(de_a)
        print(row)

    # Header legend
    print(f"\n  Legend: EN = English accuracy, DE = German accuracy, Δ = EN − DE (positive = drop)")
    print(f"  Columns: " + " | ".join(f"{label}" for _, label in MODEL_ORDER))

    header("Per-MMLU-Category Summary")
    cat_header = f"  {'Category':<22}" + "".join(f"  {'EN':>6} {'DE':>6} {'Δ':>6}  {'N':>3}" for _ in MODEL_ORDER)
    print(cat_header)
    print(f"  {'─'*22}" + "".join(f"  {'─'*6} {'─'*6} {'─'*6}  {'─'*3}" for _ in MODEL_ORDER))

    for cat in cats:
        row = f"  {cat:<22}"
        for key, _ in MODEL_ORDER:
            en_vals = cat_en_accs[cat][key]
            de_vals = cat_de_accs[cat][key]
            if not en_vals or not de_vals:
                row += f"  {'N/A':>6} {'N/A':>6} {'N/A':>6}  {'?':>3}"
                continue
            en_mean = statistics.mean(en_vals) * 100
            de_mean = statistics.mean(de_vals) * 100
            drop    = en_mean - de_mean
            row += f"  {en_mean:6.1f}% {de_mean:6.1f}% {drop:+6.1f}%  {len(en_vals):3d}"
        print(row)

    header("Overall Accuracy Summary")
    print(f"\n  {'Metric':<35}" + "".join(f"  {label:>14}" for _, label in MODEL_ORDER))
    print(f"  {'─'*35}" + "".join(f"  {'─'*14}" for _ in MODEL_ORDER))

    # Bootstrap CIs
    en_overall: dict[str, dict] = {}
    de_overall: dict[str, dict] = {}
    for key, _ in MODEL_ORDER:
        print(f"  Computing bootstrap CI for {key}...", file=sys.stderr)
        en_overall[key] = bootstrap_ci(en_records[key])
        de_overall[key] = bootstrap_ci(de_records[key])

    en_accs_overall = [en_overall[k]["mean"] for k, _ in MODEL_ORDER]
    de_accs_overall = [de_overall[k]["mean"] for k, _ in MODEL_ORDER]

    def _row(label: str, vals: list[str]) -> None:
        print(f"  {label:<35}" + "".join(f"  {v:>14}" for v in vals))

    _row("English overall accuracy",
         [f"{en_overall[k]['mean']*100:.2f}%" for k, _ in MODEL_ORDER])
    _row("English 95% CI (bootstrap)",
         [f"[{en_overall[k]['ci_lo']*100:.1f}%–{en_overall[k]['ci_hi']*100:.1f}%]"
          for k, _ in MODEL_ORDER])
    _row("German overall accuracy",
         [f"{de_overall[k]['mean']*100:.2f}%" for k, _ in MODEL_ORDER])
    _row("German 95% CI (bootstrap)",
         [f"[{de_overall[k]['ci_lo']*100:.1f}%–{de_overall[k]['ci_hi']*100:.1f}%]"
          for k, _ in MODEL_ORDER])
    _row("EN → DE accuracy drop",
         [f"{(en_overall[k]['mean'] - de_overall[k]['mean'])*100:+.2f}%"
          for k, _ in MODEL_ORDER])

    avg_subj_drop = {k: statistics.mean(subject_drops[k]) if subject_drops[k] else 0
                     for k, _ in MODEL_ORDER}
    _row("Avg subject-level drop",
         [f"{avg_subj_drop[k]*100:+.2f}%" for k, _ in MODEL_ORDER])

    header("Statistical Significance — McNemar's Test (pairwise model comparisons)")

    print("""
  McNemar's test on binary correct/incorrect decisions across the SAME questions.
  H₀: the two models make errors on the same questions (no performance difference).
  Continuity-corrected χ², df=1. Significance level: α = 0.05.

  b₀₁ = questions where Model A wrong, Model B correct  (B better on these)
  b₁₀ = questions where Model A correct, Model B wrong  (A better on these)
""")

    for lang, lang_label, recs, key_fn in [
        ("en", "English", en_records, lambda r: r.get("question", "")),
        ("de", "German",  de_records, lambda r: (r.get("subject",""), r.get("original_question_id", -1))),
    ]:
        subheader(f"MMLU {lang_label}")
        pairs = [
            ("gptoss", "gemma4"),
            ("gptoss", "qwen3.5"),
            ("gemma4", "qwen3.5"),
        ]
        for a_key, b_key in pairs:
            a_label = MODEL_KEYS[a_key]
            b_label = MODEL_KEYS[b_key]
            res = mcnemar_test(recs[a_key], recs[b_key], key_fn)
            sig = "✓ significant" if res["significant"] else "✗ not significant"
            direction = ""
            if res["b01"] != res["b10"]:
                better = b_label if res["b01"] > res["b10"] else a_label
                direction = f"  ({better} better on discordant questions)"
            print(f"  {a_label} vs {b_label}")
            print(f"    n_shared={res['n_shared']:,}  b₀₁={res['b01']:,}  b₁₀={res['b10']:,}"
                  f"  χ²={res['chi2']}  p={res['p_value']}  → {sig}{direction}")
        print()

    header("Paired t-test — Subject-Level Accuracy Drops (EN − DE)")
    print("""
  Tests whether one model's EN→DE drop significantly differs from another's.
  Uses a paired t-test on the 57 subject-level differences in accuracy drop.
  H₀: the two models degrade by the same amount across subjects.
""")
    pairs = [("gptoss", "gemma4"), ("gptoss", "qwen3.5"), ("gemma4", "qwen3.5")]
    for a_key, b_key in pairs:
        a_label = MODEL_KEYS[a_key]
        b_label = MODEL_KEYS[b_key]
        res = paired_ttest_drops(subject_drops[a_key], subject_drops[b_key])
        sig = "✓ significant" if res["significant"] else "✗ not significant"
        a_drop = statistics.mean(subject_drops[a_key]) * 100 if subject_drops[a_key] else 0
        b_drop = statistics.mean(subject_drops[b_key]) * 100 if subject_drops[b_key] else 0
        print(f"  {a_label} (avg drop {a_drop:+.2f}%) vs {b_label} (avg drop {b_drop:+.2f}%)")
        if res["t"] is not None:
            print(f"    t={res['t']}  p={res['p']}  n=57 subjects  → {sig}")
        else:
            print(f"    Insufficient data.")
        print()

    header("Top-10 Subjects with Largest Average EN→DE Accuracy Drop")
    avg_drops = {}
    for subj in all_subjects:
        drops = []
        for key, _ in MODEL_ORDER:
            en_a = en_acc[key].get(subj, {}).get("acc")
            de_a = de_acc[key].get(subj, {}).get("acc")
            if en_a is not None and de_a is not None:
                drops.append(en_a - de_a)
        if drops:
            avg_drops[subj] = statistics.mean(drops)

    sorted_drops = sorted(avg_drops.items(), key=lambda x: -x[1])
    print(f"\n  {'Subject':<42} {'Avg Drop':>10}  {'Category'}")
    print(f"  {'─'*42} {'─'*10}  {'─'*20}")
    for subj, drop in sorted_drops[:10]:
        cat = subj_cat.get(subj, "Other")
        print(f"  {subj:<42} {drop*100:+10.1f}%  {cat}")

    print()
    header("Top-10 Subjects Most Robust to Language Change (smallest drop)")
    print(f"\n  {'Subject':<42} {'Avg Drop':>10}  {'Category'}")
    print(f"  {'─'*42} {'─'*10}  {'─'*20}")
    for subj, drop in sorted_drops[-10:][::-1]:
        cat = subj_cat.get(subj, "Other")
        print(f"  {subj:<42} {drop*100:+10.1f}%  {cat}")

    print(f"\n{'═' * W}\n")

    if args.save:
        sys.stdout = tee._real
        analysis_dir = os.path.dirname(os.path.abspath(__file__))
        out = os.path.join(analysis_dir, "mmlu_crosslang_report.txt")
        with open(out, "w") as f:
            f.write(tee.getvalue())
        print(f"  → Report saved: mmlu_crosslang_report.txt\n")


if __name__ == "__main__":
    main()
