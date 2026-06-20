"""
german_gptoss.py  —  German MMLU  —  GPT-OSS-120B
-------------------------------------------------
Dataset : openai/MMMLU (DE_DE) — 14,042 questions, 57 subjects, professional translation.
Model   : openai/gpt-oss-120b
Output  : mmlu_de_gptoss_v{N}.jsonl  +  mmlu_de_gemma4_v{N}_summary.txt

Batch size mirrors English runs: MMLU_N=1000 (~17 questions/subject × 57 subjects).
Advance offset between batches with MMLU_OFFSET env var (0, 17, 34, 51 …).
Set MMLU_PROMPT_LANG=en to run the English-prompt ablation condition.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import time
import threading
import concurrent.futures
from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

# ── API / model ────────────────────────────────────────────────────────────────

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "https://litellm.uni-osnabrueck.de/v1")
LITELLM_API_KEY  = os.getenv("LITELLM_API_KEY", "")
MODEL_NAME       = "openai/gpt-oss-120b"

client = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)

# ── Dataset ────────────────────────────────────────────────────────────────────

DATASET_NAME = os.environ.get("MMLU_DATASET", "openai/MMMLU")
DATASET_LANG = "DE_DE"

# ── Run parameters ─────────────────────────────────────────────────────────────

N_QUESTIONS     = int(os.environ.get("MMLU_N",      1000))
QUESTION_OFFSET = int(os.environ.get("MMLU_OFFSET",    0))
MAX_WORKERS     = 40
RATE_LIMIT      = 100

# ── Prompt language ────────────────────────────────────────────────────────────

PROMPT_LANGUAGE = os.environ.get("MMLU_PROMPT_LANG", "de")

SYSTEM_PROMPTS = {
    "de": (
        "Sie sind ein Multiple-Choice-Prüfungsassistent. "
        'Antworten Sie NUR mit einem JSON-Objekt in genau diesem Format: {"answer": "X"} '
        "wobei X der einzelne Buchstabe A, B, C oder D ist. "
        "Fügen Sie keinen weiteren Text, keine Erklärung und keine Formatierung "
        "außerhalb des JSON-Objekts ein."
    ),
    "en": (
        "You are a multiple-choice exam assistant. "
        'Respond with ONLY a JSON object in this exact format: {"answer": "X"} '
        "where X is the single letter A, B, C, or D. "
        "Do not include any other text, explanation, or formatting outside the JSON object."
    ),
}

if PROMPT_LANGUAGE not in SYSTEM_PROMPTS:
    raise ValueError(f"MMLU_PROMPT_LANG must be 'de' or 'en', got: {PROMPT_LANGUAGE!r}")

SYSTEM_PROMPT = SYSTEM_PROMPTS[PROMPT_LANGUAGE]

# ── MMLU meta-categories (Hendrycks et al. 2021) ──────────────────────────────

MMLU_CATEGORY = {
    "abstract_algebra": "STEM", "anatomy": "STEM", "astronomy": "STEM",
    "college_biology": "STEM", "college_chemistry": "STEM",
    "college_computer_science": "STEM", "college_mathematics": "STEM",
    "college_physics": "STEM", "computer_security": "STEM",
    "conceptual_physics": "STEM", "electrical_engineering": "STEM",
    "elementary_mathematics": "STEM", "formal_logic": "STEM",
    "high_school_biology": "STEM", "high_school_chemistry": "STEM",
    "high_school_computer_science": "STEM", "high_school_mathematics": "STEM",
    "high_school_physics": "STEM", "high_school_statistics": "STEM",
    "machine_learning": "STEM",
    "high_school_european_history": "Humanities", "high_school_us_history": "Humanities",
    "high_school_world_history": "Humanities", "international_law": "Humanities",
    "jurisprudence": "Humanities", "logical_fallacies": "Humanities",
    "moral_disputes": "Humanities", "moral_scenarios": "Humanities",
    "philosophy": "Humanities", "prehistory": "Humanities",
    "professional_law": "Humanities", "world_religions": "Humanities",
    "econometrics": "Social Sciences", "high_school_geography": "Social Sciences",
    "high_school_government_and_politics": "Social Sciences",
    "high_school_macroeconomics": "Social Sciences",
    "high_school_microeconomics": "Social Sciences",
    "high_school_psychology": "Social Sciences", "human_sexuality": "Social Sciences",
    "professional_psychology": "Social Sciences", "public_relations": "Social Sciences",
    "security_studies": "Social Sciences", "sociology": "Social Sciences",
    "us_foreign_policy": "Social Sciences",
    "business_ethics": "Other", "clinical_knowledge": "Other",
    "college_medicine": "Other", "global_facts": "Other", "human_aging": "Other",
    "management": "Other", "marketing": "Other", "medical_genetics": "Other",
    "miscellaneous": "Other", "nutrition": "Other",
    "professional_accounting": "Other", "professional_medicine": "Other",
    "virology": "Other",
}

# ── Subjects ───────────────────────────────────────────────────────────────────

SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

# ── Output paths ───────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE       = "mmlu_de_gptoss"


def _next_version(base: str) -> str:
    existing = glob.glob(os.path.join(_SCRIPT_DIR, f"{base}_v*.jsonl"))
    nums = []
    for f in existing:
        m = re.search(rf"{re.escape(base)}_v(\d+)\.jsonl$", os.path.basename(f))
        if m:
            nums.append(int(m.group(1)))
    return f"{base}_v{max(nums) + 1}" if nums else f"{base}_v1"


_RUN_NAME    = _next_version(_BASE)
OUTPUT_FILE  = os.path.join(_SCRIPT_DIR, f"{_RUN_NAME}.jsonl")
SUMMARY_FILE = os.path.join(_SCRIPT_DIR, f"{_RUN_NAME}_summary.txt")

# ── Rate limiter ───────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, max_per_minute: int):
        self._interval  = 60.0 / max_per_minute
        self._lock      = threading.Lock()
        self._last_call = 0.0

    def acquire(self) -> None:
        with self._lock:
            now  = time.time()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()


rate_limiter = RateLimiter(RATE_LIMIT)
write_lock   = threading.Lock()

# ── Dataset loading ────────────────────────────────────────────────────────────

def _normalize_subject(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_").replace("-", "_")


def _detect_schema(columns: list[str]) -> dict:
    cols = set(columns)
    if "Question" in cols:       q_col = "Question"
    elif "instruction" in cols:  q_col = "instruction"
    elif "question" in cols:     q_col = "question"
    else: raise ValueError(f"No question column found. Available: {sorted(cols)}")

    if all(c in cols for c in ("A", "B", "C", "D")):     choice_mode = "upper"
    elif all(f"option_{c}" in cols for c in "abcd"):      choice_mode = "separate"
    elif "choices" in cols:                                choice_mode = "list"
    else: raise ValueError(f"No choices columns found. Available: {sorted(cols)}")

    if "Subject" in cols:        subj_col = "Subject"
    elif "subject" in cols:      subj_col = "subject"
    elif "category" in cols:     subj_col = "category"
    else: raise ValueError(f"No subject column found. Available: {sorted(cols)}")

    if "Answer" in cols:         ans_col = "Answer"
    elif "answer" in cols:       ans_col = "answer"
    else: raise ValueError(f"No answer column found. Available: {sorted(cols)}")

    return {"q_col": q_col, "choice_mode": choice_mode, "subj_col": subj_col, "ans_col": ans_col}


def _extract_record(item: dict, schema: dict) -> dict | None:
    q_col, choice_mode = schema["q_col"], schema["choice_mode"]
    subj_col, ans_col  = schema["subj_col"], schema["ans_col"]

    question = item.get(q_col, "").strip()
    if not question:
        return None

    if choice_mode == "upper":
        choices = [str(item.get(c, "")).strip() for c in ("A", "B", "C", "D")]
    elif choice_mode == "separate":
        choices = [str(item.get(f"option_{c}", "")).strip() for c in ("a", "b", "c", "d")]
    else:
        choices = [str(c).strip() for c in item.get("choices", [])]

    if len(choices) != 4 or not all(choices):
        return None

    ans = item.get(ans_col)
    if isinstance(ans, int) and 0 <= ans <= 3:
        answer_letter, answer_idx = ["A", "B", "C", "D"][ans], ans
    elif isinstance(ans, str) and ans.strip().upper() in ("A", "B", "C", "D"):
        answer_letter = ans.strip().upper()
        answer_idx    = ord(answer_letter) - ord("A")
    else:
        return None

    raw_id = item.get("Unnamed: 0")
    if raw_id is None:
        raw_id = item.get("id")

    return {
        "subject":              _normalize_subject(str(item.get(subj_col, ""))),
        "question":             question,
        "choices":              choices,
        "answer_idx":           answer_idx,
        "answer_letter":        answer_letter,
        "original_question_id": int(raw_id) if raw_id is not None else None,
    }


def load_mmlu_de_sample(subjects: list[str], n: int, offset: int = 0) -> list[dict]:
    per_subject = max(1, n // len(subjects))
    subject_set = set(subjects)
    samples: list[dict] = []

    print(f"  Loading {DATASET_NAME!r} (config={DATASET_LANG!r}, split=test)…")
    try:
        ds     = load_dataset(DATASET_NAME, DATASET_LANG, split="test")
        schema = _detect_schema(ds.column_names)
        print(f"  Schema: q='{schema['q_col']}' choices='{schema['choice_mode']}' subj='{schema['subj_col']}'")
        print(f"  Total rows: {len(ds):,}")

        by_subject: dict[str, list[dict]] = defaultdict(list)
        for item in ds:
            rec = _extract_record(dict(item), schema)
            if rec and rec["subject"] in subject_set:
                by_subject[rec["subject"]].append(rec)

        for subj in subjects:
            items = by_subject.get(subj, [])
            start, end = min(offset, len(items)), min(offset + per_subject, len(items))
            if start >= end:
                print(f"  Warning: {subj}: exhausted at offset={offset} ({len(items)} total)")
                continue
            for idx, item in enumerate(items[start:end], start=start):
                item["subject_question_idx"] = idx
            samples.extend(items[start:end])

        print(f"  Loaded {len(samples)} questions.\n")
        return samples[:n]

    except Exception as e:
        print(f"  Unified loading failed: {e}\n  Trying per-subject fallback…")

    for subject in subjects:
        try:
            ds     = load_dataset(DATASET_NAME, subject, split="test")
            schema = _detect_schema(ds.column_names)
            recs   = [r for r in (_extract_record(dict(item), schema) for item in ds) if r]
            for r in recs:
                r["subject"] = subject
            start, end = min(offset, len(recs)), min(offset + per_subject, len(recs))
            if start >= end:
                continue
            for idx, rec in enumerate(recs[start:end], start=start):
                rec["subject_question_idx"] = idx
            samples.extend(recs[start:end])
        except Exception as ex:
            print(f"  Warning: could not load {subject}: {ex}")

    print(f"  Loaded {len(samples)} questions.\n")
    return samples[:n]

# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(question: str, choices: list[str]) -> str:
    labels       = ["A", "B", "C", "D"]
    choices_text = "\n".join(f"{l}. {t}" for l, t in zip(labels, choices))
    return f"Frage: {question}\n\n{choices_text}"

# ── Model call ─────────────────────────────────────────────────────────────────

def call_model(prompt: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        rate_limiter.acquire()
        try:
            t0       = time.time()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=8192,
                temperature=0.0,
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
                "prompt_tokens":     usage.prompt_tokens     if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "total_tokens":      usage.total_tokens      if usage else None,
            }
            if content is None:
                return {"text": None, "reasoning": reasoning, "latency": latency,
                        "tokens": tokens, "error": "null content from model"}
            return {"text": content.strip(), "reasoning": reasoning,
                    "latency": latency, "tokens": tokens, "error": None}

        except Exception as e:
            err_str = str(e)
            tqdm.write(f"  [attempt {attempt+1}/{retries}] Error: {err_str[:200]}")
            if "401" in err_str:
                return {"text": None, "latency": None, "tokens": {}, "error": "unauthorised (401)"}
            if "404" in err_str:
                return {"text": None, "latency": None, "tokens": {}, "error": "model unavailable (404)"}
            wait = 15 if "No deployments available" in err_str else 65 if ("Rate limit" in err_str or "429" in err_str) else 2 ** attempt
            if attempt < retries - 1:
                tqdm.write(f"  Waiting {wait}s before retry…")
                time.sleep(wait)

    return {"text": None, "latency": None, "tokens": {}, "error": "max retries exceeded"}

# ── Answer parser ──────────────────────────────────────────────────────────────

def parse_answer_letter(response_text: str | None) -> tuple[str | None, str]:
    if not response_text:
        return None, "failed"
    t = response_text.strip()
    try:
        clean = re.sub(r'^```[a-z]*\s*|\s*```$', '', t, flags=re.IGNORECASE).strip()
        data  = json.loads(clean)
        if isinstance(data, dict):
            letter = str(data.get("answer", "")).strip().upper()
            if letter in {"A", "B", "C", "D"}:
                return letter, "json"
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r'\{\s*"answer"\s*:\s*"([A-D])"\s*\}', t, re.IGNORECASE)
    if m:
        return m.group(1).upper(), "json_search"
    if len(t) == 1 and t.upper() in {"A", "B", "C", "D"}:
        return t.upper(), "regex"
    m = re.match(r'^([A-D])[^A-Za-z]', t, re.IGNORECASE)
    if m:
        return m.group(1).upper(), "regex"
    m = re.search(r'\b(?:answer|Antwort)\b\s*(?:is\s+|ist\s+|:\s*)?([A-D])\b', t, re.IGNORECASE)
    if m:
        return m.group(1).upper(), "regex"
    for line in t.splitlines():
        line = line.strip()
        if re.fullmatch(r'[A-D]', line, re.IGNORECASE):
            return line.upper(), "regex"
        m = re.match(r'^([A-D])[^A-Za-z]', line, re.IGNORECASE)
        if m:
            return m.group(1).upper(), "regex"
    return None, "failed"

# ── Per-question evaluation ────────────────────────────────────────────────────

def evaluate_question(args: tuple) -> dict:
    i, q     = args
    prompt   = build_prompt(q["question"], q["choices"])
    response = call_model(prompt)

    predicted, parse_method = parse_answer_letter(response["text"])
    is_correct = predicted == q["answer_letter"]

    record = {
        "id":                    i + 1,
        "model":                 MODEL_NAME,
        "language":              "de",
        "dataset_name":          DATASET_NAME,
        "prompt_language":       PROMPT_LANGUAGE,
        "condition":             "no_retrieval",
        "original_question_id":  q.get("original_question_id"),
        "subject_question_idx":  q.get("subject_question_idx"),
        "mmlu_category":         MMLU_CATEGORY.get(q["subject"], "Unknown"),
        "subject":               q["subject"],
        "question":              q["question"],
        "choices":               q["choices"],
        "answer_letter":         q["answer_letter"],
        "prompt_sent":           prompt,
        "response_text":         response["text"],
        "reasoning":             response.get("reasoning"),
        "predicted_letter":      predicted,
        "parse_method":          parse_method,
        "is_correct":            is_correct,
        "latency_s":             response["latency"],
        "prompt_tokens":         (response.get("tokens") or {}).get("prompt_tokens"),
        "completion_tokens":     (response.get("tokens") or {}).get("completion_tokens"),
        "total_tokens":          (response.get("tokens") or {}).get("total_tokens"),
        "error":                 response["error"],
    }

    with write_lock:
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    return record

# ── Main evaluation loop ───────────────────────────────────────────────────────

def run_evaluation():
    per_subject = max(1, N_QUESTIONS // len(SUBJECTS))

    print("=" * 65)
    print("MMLU Evaluation  —  German  —  GPT-OSS-120B")
    print(f"Model          : {MODEL_NAME}")
    print(f"Gateway        : {LITELLM_BASE_URL}")
    print(f"Dataset        : {DATASET_NAME}  (lang={DATASET_LANG})")
    print(f"Prompt lang    : {PROMPT_LANGUAGE}  ({'German' if PROMPT_LANGUAGE == 'de' else 'English (ablation)'})")
    print(f"Subjects       : {len(SUBJECTS)} × ~{per_subject} questions  (offset={QUESTION_OFFSET})")
    print(f"N              : {N_QUESTIONS} questions (target)")
    print(f"Workers        : {MAX_WORKERS}  |  Rate limit: {RATE_LIMIT} req/min")
    print(f"Output         : {OUTPUT_FILE}")
    print("=" * 65)

    print("\nLoading German MMLU data…")
    questions = load_mmlu_de_sample(SUBJECTS, N_QUESTIONS, offset=QUESTION_OFFSET)
    print(f"Loaded {len(questions)} questions.\n")

    if not questions:
        print("No questions available — check dataset or offset.")
        return

    results: list[dict | None] = [None] * len(questions)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(evaluate_question, (i, q)): i for i, q in enumerate(questions)}
        with tqdm(total=len(questions), desc="Evaluating", unit="q") as pbar:
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    results[i] = {"id": i + 1, "error": str(e), "is_correct": False}
                pbar.update(1)

    valid_results = [r for r in results if r is not None]
    correct  = sum(1 for r in valid_results if r.get("is_correct"))
    errors   = sum(1 for r in valid_results if r.get("error"))
    answered = len(valid_results) - errors
    accuracy = correct / answered if answered > 0 else 0

    df          = pd.DataFrame(valid_results)
    df_answered = df[df["error"].isna()]

    lines = [
        "=" * 65,
        "RESULTS SUMMARY — German MMLU — GPT-OSS-120B",
        "=" * 65,
        f"Model           : {MODEL_NAME}",
        f"Dataset         : {DATASET_NAME}  (lang={DATASET_LANG})",
        f"Prompt language : {PROMPT_LANGUAGE}",
        f"Offset          : {QUESTION_OFFSET}",
        f"Questions asked : {len(questions)}",
        f"Answered        : {answered}",
        f"Errors          : {errors}",
        f"Correct         : {correct}",
        f"Accuracy        : {accuracy:.1%}",
        "",
    ]

    if not df_answered.empty:
        lines.append(f"  {'Subject':<40} {'Correct':>7}  {'Total':>5}  {'Accuracy':>8}")
        lines.append(f"  {'-'*40} {'-'*7}  {'-'*5}  {'-'*8}")
        breakdown = df_answered.groupby("subject")["is_correct"].agg(["sum", "count"])
        breakdown["accuracy"] = breakdown["sum"] / breakdown["count"]
        for subject, row in breakdown.iterrows():
            lines.append(f"  {subject:<40} {int(row['sum']):>7}  {int(row['count']):>5}  {row['accuracy']:>8.1%}")
        lines.append("")
        lines.append("Parse method breakdown:")
        for method, cnt in df_answered["parse_method"].value_counts().items():
            lines.append(f"  {method:<15} {cnt:>4} responses")

    lines.append(f"\nResults : {OUTPUT_FILE}")
    lines.append(f"Summary : {SUMMARY_FILE}")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    with open(SUMMARY_FILE, "w") as sf:
        sf.write(summary_text + "\n")


if __name__ == "__main__":
    run_evaluation()
