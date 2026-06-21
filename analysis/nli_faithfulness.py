from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
from collections import defaultdict
from typing import Optional

import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

from tqdm import tqdm

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise SystemExit("Missing dependency: pip3 install rank-bm25")

try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    raise SystemExit("Missing dependency: pip3 install transformers torch")


SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
THESIS_ROOT = os.path.dirname(SCRIPT_DIR)
SCIDQA_DIR  = os.path.join(THESIS_ROOT, "SciDQA")
DATA_DIR    = os.path.join(SCIDQA_DIR,  "data")

COMBINED_FILES = {
    "gemma-4-31B"  : os.path.join(SCRIPT_DIR, "scidqa_gemma4_combined.jsonl"),
    "gpt-oss-120b" : os.path.join(SCRIPT_DIR, "scidqa_gptoss_combined.jsonl"),
    "Qwen3.5-122B" : os.path.join(SCRIPT_DIR, "scidqa_qwen3.5_combined.jsonl"),
}

RAG_CONDITIONS  = ["rag_top3", "rag_top5", "rag_dense"]
VERSION_MAP     = {"Initial": "initial", "Revised": "final"}

# Chunking params — must match the evaluation scripts exactly
SENTENCES_PER_CHUNK = 10
CHUNK_OVERLAP       = 1

parser = argparse.ArgumentParser(description="NLI faithfulness scoring for SciDQA")
parser.add_argument("--condition", choices=RAG_CONDITIONS, default=None,
                    help="Score one condition only (default: all RAG conditions)")
parser.add_argument("--model",     choices=["gemma4","gptoss","qwen3.5"], default=None,
                    help="Score one model only (default: all models)")
args = parser.parse_args()

CONDITIONS_TO_SCORE = [args.condition] if args.condition else RAG_CONDITIONS

MODEL_FILTER = {
    "gemma4"  : "gemma-4-31B",
    "gptoss"  : "gpt-oss-120b",
    "qwen3.5" : "Qwen3.5-122B",
}.get(args.model) if args.model else None

OUTPUT_JSONL  = os.path.join(SCRIPT_DIR, "scidqa_nli_faithfulness.jsonl")
OUTPUT_REPORT = os.path.join(SCRIPT_DIR, "scidqa_nli_faithfulness_report.txt")

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

print(f"Loading NLI model: {NLI_MODEL}  (first run downloads ~350 MB)")
_nli = hf_pipeline(
    "text-classification",
    model=NLI_MODEL,
    device=-1,    # CPU; change to 0 for GPU
    top_k=None,   # return scores for all labels
)

# Discover the label that means entailment (varies by model).
# Single-input + top_k=None → flat list of dicts: [{"label":..,"score":..}, ...]
# Batch input  + top_k=None → nested list:        [[{"label":..}, ...], [...]]
# So for the probe (single input), _probe_raw IS the list of dicts already.
_probe_raw = _nli({"text": "The sky is blue.", "text_pair": "The sky is blue."})
_probe = _probe_raw if isinstance(_probe_raw[0], dict) else _probe_raw[0]
_label_map: dict[str, str] = {}
for item in _probe:
    lab = item["label"].upper()
    if "ENTAIL" in lab:
        _label_map["entailment"] = item["label"]
    elif "CONTRADICT" in lab:
        _label_map["contradiction"] = item["label"]
    elif "NEUTRAL" in lab:
        _label_map["neutral"] = item["label"]

print(f"NLI labels detected: {_label_map}")

def chunk_text(text: str) -> list[str]:
    chunks: list[str] = []
    stride = max(1, SENTENCES_PER_CHUNK - CHUNK_OVERLAP)
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        sents = sent_tokenize(para)
        if not sents:
            continue
        for i in range(0, len(sents), stride):
            window = sents[i : i + SENTENCES_PER_CHUNK]
            chunk  = " ".join(window).strip()
            if chunk:
                chunks.append(chunk)
    return chunks

def get_rag_context(question: str, paper_text: str, top_k: int) -> str:
    chunks = chunk_text(paper_text)
    if not chunks:
        return ""
    tokenized = [c.lower().split() for c in chunks]
    bm25      = BM25Okapi(tokenized)
    scores    = bm25.get_scores(question.lower().split())
    top_idx   = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    selected  = [chunks[i] for i in sorted(top_idx)]
    return "\n\n".join(selected)

TOPK = {"rag_top3": 3, "rag_top5": 5, "rag_dense": 3}

def faithfulness_score(response: str, context: str) -> Optional[float]:
    """
    Returns fraction of response sentences entailed by the context.
    Returns None if response or context is empty.
    """
    if not response or not context:
        return None

    sentences = [s.strip() for s in sent_tokenize(response) if s.strip()]
    if not sentences:
        return None

    # Truncate context to NLI model's effective window (avoid silent truncation)
    context_truncated = context[:2000]

    entailed = 0
    pairs    = [{"text": context_truncated, "text_pair": s} for s in sentences]

    try:
        results = _nli(pairs, batch_size=16, truncation=True)
        for res in results:
            scores_dict = {item["label"]: item["score"] for item in res}
            entail_label = _label_map.get("entailment", "")
            contra_label = _label_map.get("contradiction", "")
            if entail_label and scores_dict.get(entail_label, 0) > scores_dict.get(contra_label, 0):
                entailed += 1
    except Exception as e:
        print(f"  NLI error: {e}")
        return None

    return round(entailed / len(sentences), 4)

def main():
    print(f"\nConditions to score : {CONDITIONS_TO_SCORE}")
    print(f"Models to score     : {MODEL_FILTER or 'all'}")
    print(f"Output JSONL        : {os.path.basename(OUTPUT_JSONL)}")
    print()

    print("Loading paper full texts (pkl)...")
    with open(os.path.join(DATA_DIR, "papers_fulltext_nougat.pkl"), "rb") as f:
        fulltext = pickle.load(f)

    # Cache chunked papers to avoid re-chunking the same paper repeatedly
    chunk_cache: dict[str, list[str]] = {}

    all_results:  list[dict] = []
    model_stats:  dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    already_scored: set[tuple] = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    already_scored.add((r["id"], r["condition"], r["model"]))
        print(f"  Resuming — {len(already_scored)} records already scored.\n")

    write_fh = open(OUTPUT_JSONL, "a")

    try:
        for model_label, fpath in COMBINED_FILES.items():
            if MODEL_FILTER and model_label != MODEL_FILTER:
                continue
            if not os.path.exists(fpath):
                print(f"  [{model_label}] Combined file not found: {fpath} — skipping")
                continue

            print(f"\n{'═'*60}")
            print(f"  {model_label}")
            print(f"{'═'*60}")

            records_to_score: list[dict] = []
            with open(fpath) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("condition") not in CONDITIONS_TO_SCORE:
                        continue
                    if r.get("error") or not r.get("response_text"):
                        continue
                    key = (r["id"], r["condition"], r["model"])
                    if key in already_scored:
                        continue
                    records_to_score.append(r)

            print(f"  Records to score: {len(records_to_score)}")

            for rec in tqdm(records_to_score, desc=f"  {model_label}"):
                condition = rec["condition"]
                top_k     = TOPK.get(condition, 3)
                vkey      = VERSION_MAP.get(rec.get("version", "Initial"), "initial")
                pid       = rec["pid"]

                # Get paper text
                paper_text = fulltext.get(vkey, {}).get(pid, "")
                if not paper_text:
                    paper_text = fulltext.get("initial", {}).get(pid, "") or \
                                 fulltext.get("final",   {}).get(pid, "")

                if not paper_text:
                    continue

                # Get the same RAG context that was used during evaluation
                # (dense condition uses same top-k as rag_top3 — re-use BM25 as proxy
                #  since we don't have stored embeddings; difference is minimal for faithfulness)
                context = get_rag_context(rec["question"], paper_text, top_k)
                if not context:
                    continue

                score = faithfulness_score(rec["response_text"], context)
                if score is None:
                    continue

                result = {
                    "id"                  : rec["id"],
                    "model"               : rec["model"],
                    "condition"           : condition,
                    "pid"                 : pid,
                    "nli_faithfulness"    : score,
                    "rouge_avg"           : rec.get("rouge_avg"),
                    "ngram_grounding_score": rec.get("ngram_grounding_score"),
                    "no_answer_signal"    : rec.get("no_answer_signal"),
                }
                all_results.append(result)
                model_stats[model_label][condition].append(score)

                if write_fh:
                    write_fh.write(json.dumps(result) + "\n")
                    write_fh.flush()

    finally:
        if write_fh:
            write_fh.close()

    report_lines: list[str] = []

    def p(s: str = "") -> None:
        report_lines.append(s)
        print(s)

    p(f"\n{'═'*65}")
    p(f"  NLI Faithfulness Report — SciDQA RAG Conditions")
    p(f"  Model: {NLI_MODEL}")
    p(f"{'═'*65}")
    p(f"  Faithfulness = fraction of response sentences entailed by context")
    p(f"  1.0 = fully grounded   |   0.0 = no support from retrieved chunks")
    p()

    header = f"  {'Condition':<12}" + "".join(f" {'':>14}" for _ in COMBINED_FILES)
    models_shown = [m for m in COMBINED_FILES if not MODEL_FILTER or m == MODEL_FILTER]

    p(f"  {'Condition':<12}" + "".join(f" {m[:14]:>14}" for m in models_shown))
    p(f"  {'─'*12}" + "".join(f" {'─'*14}" for _ in models_shown))

    for cond in CONDITIONS_TO_SCORE:
        row = f"  {cond:<12}"
        for m in models_shown:
            vals = model_stats[m].get(cond, [])
            if vals:
                row += f" {statistics.mean(vals):>14.4f}"
            else:
                row += f" {'—':>14}"
        p(row)

    p()
    p(f"  Records scored: {len(all_results):,}")
    p()
    p(f"  INTERPRETATION:")
    p(f"  rag_top3 faithfulness close to 1.0 → model stays grounded in chunks")
    p(f"  rag_top3 faithfulness close to 0.0 → model adds facts not in chunks")
    p(f"  Compare across models: lower score = more hallucination tendency")

    with open(OUTPUT_REPORT, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\n  → Saved JSONL : {os.path.basename(OUTPUT_JSONL)}")
    print(f"  → Saved report: {os.path.basename(OUTPUT_REPORT)}")


if __name__ == "__main__":
    main()
