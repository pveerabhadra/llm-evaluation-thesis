"""
mmlu_german_gemma4.py
---------------------
German MMLU evaluation — Gemma-4-31B

DATASET
  openai/MMMLU (config="DE_DE")
  OpenAI's official multilingual MMLU — professional human translation of all 14,042 questions
  across all 57 subjects into German (and 13 other languages). Released alongside GPT-4 eval.
  Cite as: OpenAI Multilingual MMLU (2023), https://huggingface.co/datasets/openai/MMMLU
  Columns: Question, A, B, C, D, Answer (letter), Subject (English name matching cais/mmlu)

  alexandrainst/m_mmlu was tried first but has NO subject column — it is a flat
  sample of ~200 questions used for few-shot prompting, not suitable for per-subject evaluation.

  Override dataset via MMLU_DATASET env var if needed.

PROMPT LANGUAGE
  Default: fully German system prompt (MMLU_PROMPT_LANG=de).
  Set MMLU_PROMPT_LANG=en to run an English-prompt ablation on the same German questions.
  This is a key experimental lever for your thesis: lower performance with de vs en prompt
  isolates instruction-following costs from knowledge costs.

OUTPUT
  Files written to this directory (MMLU German/) so combine_results.py --lang de picks them up.
  Naming: mmlu_de_gemma4_v{N}.jsonl + mmlu_de_gemma4_v{N}_summary.txt

PILOT DEFAULTS (env-var overridable)
  MMLU_N=114        2 questions per subject × 57 subjects (closest even split to 100)
  MMLU_OFFSET=0     fresh start; German offset is independent of English offset
  MMLU_PROMPT_LANG=de

CROSS-LANGUAGE COMPARABILITY
  - Same model, same API, same temperature=0, same JSON answer format as English runs.
  - Extra record fields (language, dataset_name, prompt_language) enable EN/DE joins later.
  - QUESTION_OFFSET=0 here ≠ any English offset, so no overlap risk.
  - combine_results.py deduplicates on r["question"] (German text) — safe, no collision with EN.
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

LITELLM_BASE_URL = "https://litellm.uni-osnabrueck.de/v1"
LITELLM_API_KEY  = os.getenv("LITELLM_API_KEY", "")
MODEL_NAME       = "google/gemma-4-31B-it"

client = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)

# ── Dataset ────────────────────────────────────────────────────────────────────

DATASET_NAME = os.environ.get("MMLU_DATASET", "openai/MMMLU")
DATASET_LANG = "DE_DE"   # dataset config / language code (German)

# ── Run parameters ─────────────────────────────────────────────────────────────

# Pilot: 2 per subject × 57 subjects = 114  (nearest clean split above 100)
N_QUESTIONS     = int(os.environ.get("MMLU_N",      114))
QUESTION_OFFSET = int(os.environ.get("MMLU_OFFSET",   0))
MAX_WORKERS     = 5
RATE_LIMIT      = 15   # requests / minute

# ── Prompt language ────────────────────────────────────────────────────────────

# de = German system prompt (default — tests model in fully German context)
# en = English system prompt on German questions (ablation: isolates instruction-language effect)
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

# ── Subjects (same 57 as English MMLU for direct comparison) ──────────────────

# ── MMLU meta-categories ──────────────────────────────────────────────────────
# The 57 subjects fall into 4 domains. Storing this per-record enables
# domain-level EN/DE comparison in the thesis (e.g., does the language gap
# differ across STEM vs Humanities?).

MMLU_CATEGORY = {
    # STEM (20) — Hendrycks et al. 2021 grouping
    "abstract_algebra":                    "STEM",
    "anatomy":                             "STEM",
    "astronomy":                           "STEM",
    "college_biology":                     "STEM",
    "college_chemistry":                   "STEM",
    "college_computer_science":            "STEM",
    "college_mathematics":                 "STEM",
    "college_physics":                     "STEM",
    "computer_security":                   "STEM",
    "conceptual_physics":                  "STEM",
    "electrical_engineering":              "STEM",
    "elementary_mathematics":              "STEM",
    "formal_logic":                        "STEM",
    "high_school_biology":                 "STEM",
    "high_school_chemistry":               "STEM",
    "high_school_computer_science":        "STEM",
    "high_school_mathematics":             "STEM",
    "high_school_physics":                 "STEM",
    "high_school_statistics":              "STEM",
    "machine_learning":                    "STEM",
    # Humanities (12)
    "high_school_european_history":        "Humanities",
    "high_school_us_history":              "Humanities",
    "high_school_world_history":           "Humanities",
    "international_law":                   "Humanities",
    "jurisprudence":                       "Humanities",
    "logical_fallacies":                   "Humanities",
    "moral_disputes":                      "Humanities",
    "moral_scenarios":                     "Humanities",
    "philosophy":                          "Humanities",
    "prehistory":                          "Humanities",
    "professional_law":                    "Humanities",
    "world_religions":                     "Humanities",
    # Social Sciences (12)
    "econometrics":                        "Social Sciences",
    "high_school_geography":               "Social Sciences",
    "high_school_government_and_politics": "Social Sciences",
    "high_school_macroeconomics":          "Social Sciences",
    "high_school_microeconomics":          "Social Sciences",
    "high_school_psychology":              "Social Sciences",
    "human_sexuality":                     "Social Sciences",
    "professional_psychology":             "Social Sciences",
    "public_relations":                    "Social Sciences",
    "security_studies":                    "Social Sciences",
    "sociology":                           "Social Sciences",
    "us_foreign_policy":                   "Social Sciences",
    # Other (13)
    "business_ethics":                     "Other",
    "clinical_knowledge":                  "Other",
    "college_medicine":                    "Other",
    "global_facts":                        "Other",
    "human_aging":                         "Other",
    "management":                          "Other",
    "marketing":                           "Other",
    "medical_genetics":                    "Other",
    "miscellaneous":                       "Other",
    "nutrition":                           "Other",
    "professional_accounting":             "Other",
    "professional_medicine":               "Other",
    "virology":                            "Other",
}

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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # = MMLU German/
_BASE       = "mmlu_de_gemma4"


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
    """Enforces a minimum gap between API calls across threads."""

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
    """Map dataset subject names to canonical English MMLU underscore names.

    openai/MMMLU uses space-separated names (e.g. "abstract algebra"),
    while cais/mmlu and our SUBJECTS list use underscores ("abstract_algebra").
    """
    return raw.strip().lower().replace(" ", "_").replace("-", "_")


def _detect_schema(columns: list[str]) -> dict:
    """Detect column name schema from the loaded dataset's column list.

    Handles three known schemas:
      openai/MMMLU    : Question, A, B, C, D, Answer, Subject
      alexandrainst   : instruction, option_a/b/c/d, answer (no subject — not usable)
      cais/mmlu-style : question, choices (list), answer (int), subject
    """
    cols = set(columns)

    # Question column
    if "Question" in cols:
        q_col = "Question"
    elif "instruction" in cols:
        q_col = "instruction"
    elif "question" in cols:
        q_col = "question"
    else:
        raise ValueError(
            f"Cannot find question column. Available columns: {sorted(cols)}\n"
            "Set MMLU_DATASET to a different German MMLU dataset if needed."
        )

    # Choices columns
    if all(c in cols for c in ("A", "B", "C", "D")):
        choice_mode = "upper"         # openai/MMMLU: A, B, C, D (uppercase single letter)
    elif all(f"option_{c}" in cols for c in ("a", "b", "c", "d")):
        choice_mode = "separate"      # alexandrainst: option_a, option_b, option_c, option_d
    elif "choices" in cols:
        choice_mode = "list"          # cais/mmlu: choices is a list of 4 strings
    else:
        raise ValueError(
            f"Cannot find choices columns. Available columns: {sorted(cols)}"
        )

    # Subject column
    if "Subject" in cols:
        subj_col = "Subject"
    elif "subject" in cols:
        subj_col = "subject"
    elif "category" in cols:
        subj_col = "category"
    elif "topic" in cols:
        subj_col = "topic"
    else:
        raise ValueError(
            f"Cannot find subject column. Available columns: {sorted(cols)}\n"
            f"Note: alexandrainst/m_mmlu has no subject column — use openai/MMMLU instead."
        )

    # Answer column (letter or int)
    if "Answer" in cols:
        ans_col = "Answer"
    elif "answer" in cols:
        ans_col = "answer"
    else:
        raise ValueError(f"Cannot find answer column. Available columns: {sorted(cols)}")

    return {
        "q_col":       q_col,
        "choice_mode": choice_mode,
        "subj_col":    subj_col,
        "ans_col":     ans_col,
    }


def _extract_record(item: dict, schema: dict) -> dict | None:
    """Convert a raw dataset row to a normalised sample dict. Returns None on bad rows."""
    q_col       = schema["q_col"]
    choice_mode = schema["choice_mode"]
    subj_col    = schema["subj_col"]
    ans_col     = schema["ans_col"]

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
        answer_letter = ["A", "B", "C", "D"][ans]
        answer_idx    = ans
    elif isinstance(ans, str) and ans.strip().upper() in ("A", "B", "C", "D"):
        answer_letter = ans.strip().upper()
        answer_idx    = ord(answer_letter) - ord("A")
    else:
        return None

    raw_subj = str(item.get(subj_col, "")).strip()
    subject  = _normalize_subject(raw_subj)

    # Global question ID from openai/MMMLU ("Unnamed: 0" column, 0-indexed).
    # This is the same row index as cais/mmlu — store it so EN and DE records
    # can be joined at the question level after both runs are complete.
    # Note: must check `is not None` — value 0 is valid but falsy.
    raw_id = item.get("Unnamed: 0")
    if raw_id is None:
        raw_id = item.get("id")
    original_question_id = int(raw_id) if raw_id is not None else None

    return {
        "subject":              subject,
        "question":             question,
        "choices":              choices,
        "answer_idx":           answer_idx,
        "answer_letter":        answer_letter,
        "original_question_id": original_question_id,
    }


def load_mmlu_de_sample(subjects: list[str], n: int, offset: int = 0) -> list[dict]:
    """Load German MMLU questions from the configured dataset.

    Tries two loading strategies:
      1. Unified: one dataset config for all subjects (alexandrainst/m_mmlu style).
      2. Per-subject: separate config per subject (cais/mmlu / community-datasets style).

    Returns at most `n` samples, with at most ceil(n/len(subjects)) per subject,
    starting at `offset` within each subject's available questions.
    """
    per_subject = max(1, math.ceil(n / len(subjects)))
    subject_set = set(subjects)
    samples: list[dict] = []

    # ── Strategy 1: Unified dataset (all subjects in one config) ──────────────
    print(f"  Loading {DATASET_NAME!r} (config={DATASET_LANG!r}, split=test)…")
    try:
        ds = load_dataset(DATASET_NAME, DATASET_LANG, split="test")
        schema = _detect_schema(ds.column_names)
        print(f"  Schema detected: question='{schema['q_col']}', "
              f"choices='{schema['choice_mode']}', subject='{schema['subj_col']}'")
        print(f"  Total rows in dataset: {len(ds):,}")

        by_subject: dict[str, list[dict]] = defaultdict(list)
        skipped_subjects: set[str] = set()

        for item in ds:
            rec = _extract_record(dict(item), schema)
            if rec is None:
                continue
            subj = rec["subject"]
            if subj not in subject_set:
                skipped_subjects.add(subj)
                continue
            by_subject[subj].append(rec)

        if skipped_subjects:
            print(f"  Note: {len(skipped_subjects)} dataset subjects not in SUBJECTS list "
                  f"(first 5: {sorted(skipped_subjects)[:5]})")

        for subj in subjects:
            items = by_subject.get(subj, [])
            start = min(offset, len(items))
            end   = min(offset + per_subject, len(items))
            if start >= end:
                if items:
                    print(f"  Warning: {subj}: only {len(items)} questions, "
                          f"offset={offset} exhausts it — skipping")
                else:
                    print(f"  Warning: {subj}: no questions found in dataset")
                continue
            for idx, item in enumerate(items[start:end], start=start):
                item["subject_question_idx"] = idx
            samples.extend(items[start:end])

        print(f"  Loaded {len(samples)} questions via unified strategy.\n")
        return samples[:n]

    except Exception as e:
        print(f"  Unified loading failed: {e}")
        print(f"  Trying per-subject loading (cais/mmlu-compatible)…\n")

    # ── Strategy 2: Per-subject configs (fallback) ────────────────────────────
    for subject in subjects:
        print(f"  Loading: {subject}…")
        try:
            ds    = load_dataset(DATASET_NAME, subject, split="test")
            schema = _detect_schema(ds.column_names)
            recs  = [_extract_record(dict(item), schema) for item in ds]
            recs  = [r for r in recs if r is not None]
            for r in recs:
                r["subject"] = subject   # enforce canonical name

            start = min(offset, len(recs))
            end   = min(offset + per_subject, len(recs))
            if start >= end:
                print(f"  Warning: not enough questions in {subject} at offset {offset}")
                continue
            for idx, rec in enumerate(recs[start:end], start=start):
                rec["subject_question_idx"] = idx
            samples.extend(recs[start:end])

        except Exception as ex:
            print(f"  Warning: could not load {subject}: {ex}")

    print(f"  Loaded {len(samples)} questions via per-subject strategy.\n")
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

            if "No deployments available" in err_str:
                wait = 15
            elif "Rate limit exceeded" in err_str or "429" in err_str:
                wait = 65
            else:
                wait = 2 ** attempt

            if attempt < retries - 1:
                tqdm.write(f"  Waiting {wait}s before retry…")
                time.sleep(wait)

    return {"text": None, "latency": None, "tokens": {}, "error": "max retries exceeded"}


# ── Answer parser (language-agnostic: JSON format is always English A/B/C/D) ──

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

    m = re.search(
        r'\b(?:the\s+)?(?:correct\s+)?(?:answer|Antwort)\b\s*(?:is\s+|ist\s+|:\s*)?([A-D])\b',
        t, re.IGNORECASE,
    )
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
        # ── Identity & comparability ──────────────────────────────────────────
        "id":                    i + 1,
        "model":                 MODEL_NAME,
        "language":              "de",
        "dataset_name":          DATASET_NAME,
        "prompt_language":       PROMPT_LANGUAGE,
        "condition":             "no_retrieval",
        # ── Question linking ──────────────────────────────────────────────────
        # original_question_id = "Unnamed: 0" from openai/MMMLU.
        # This is the same row index as cais/mmlu, so you can JOIN German and
        # English records on this field after running both evaluations.
        # subject_question_idx = 0-based position within this subject's question
        # list. Matches the `offset` used in English runs for offset alignment.
        # mmlu_category groups the 57 subjects into STEM / Humanities /
        # Social Sciences / Other (Hendrycks et al. 2021) for domain analysis.
        "original_question_id":  q.get("original_question_id"),
        "subject_question_idx":  q.get("subject_question_idx"),
        "mmlu_category":         MMLU_CATEGORY.get(q["subject"], "Unknown"),
        # ── Question ─────────────────────────────────────────────────────────
        "subject":               q["subject"],
        "question":              q["question"],
        "choices":               q["choices"],
        "answer_letter":         q["answer_letter"],
        "prompt_sent":           prompt,
        # ── Model output ─────────────────────────────────────────────────────
        "response_text":         response["text"],
        "reasoning":             response.get("reasoning"),
        "predicted_letter":      predicted,
        "parse_method":          parse_method,
        "is_correct":            is_correct,
        # ── Performance metadata ──────────────────────────────────────────────
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
    per_subject = max(1, math.ceil(N_QUESTIONS / len(SUBJECTS)))

    print("=" * 65)
    print("MMLU Evaluation  —  German  —  gemma-4-31B")
    print(f"Model          : {MODEL_NAME}")
    print(f"Gateway        : {LITELLM_BASE_URL}")
    print(f"Dataset        : {DATASET_NAME}  (lang={DATASET_LANG})")
    print(f"Prompt language: {PROMPT_LANGUAGE}  "
          f"({'German system prompt' if PROMPT_LANGUAGE == 'de' else 'English system prompt (ablation)'})")
    print(f"Subjects       : {len(SUBJECTS)} subjects × ~{per_subject} questions each "
          f"(offset={QUESTION_OFFSET})")
    print(f"N              : {N_QUESTIONS} questions (target)")
    print(f"Workers        : {MAX_WORKERS}  |  Rate limit: {RATE_LIMIT} req/min")
    print(f"Output         : {OUTPUT_FILE}")
    print("=" * 65)

    print("\nLoading German MMLU data…")
    questions = load_mmlu_de_sample(SUBJECTS, N_QUESTIONS, offset=QUESTION_OFFSET)
    print(f"Loaded {len(questions)} questions.\n")

    if len(questions) == 0:
        print("No questions available — check dataset name or offset.")
        return

    results: list[dict | None] = [None] * len(questions)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(evaluate_question, (i, q)): i
            for i, q in enumerate(questions)
        }
        with tqdm(total=len(questions), desc="Evaluating", unit="q") as pbar:
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    results[i] = {"id": i + 1, "error": str(e), "is_correct": False}
                pbar.update(1)

    valid_results = [r for r in results if r is not None]
    correct       = sum(1 for r in valid_results if r.get("is_correct"))
    errors        = sum(1 for r in valid_results if r.get("error"))
    answered      = len(valid_results) - errors
    accuracy      = correct / answered if answered > 0 else 0

    df          = pd.DataFrame(valid_results)
    df_answered = df[df["error"].isna()]

    lines = [
        "=" * 65,
        "RESULTS SUMMARY — German MMLU — gemma-4-31B",
        "=" * 65,
        f"Model           : {MODEL_NAME}",
        f"Dataset         : {DATASET_NAME}  (lang={DATASET_LANG})",
        f"Prompt language : {PROMPT_LANGUAGE}",
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
            lines.append(
                f"  {subject:<40} {int(row['sum']):>7}  {int(row['count']):>5}  {row['accuracy']:>8.1%}"
            )
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
