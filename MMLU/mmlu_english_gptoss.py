"""
MMLU English Evaluation — GPT-OSS 120B
========================================
Evaluates 570 questions across 57 MMLU subjects (English).
See mmlu_english_gemma4.py for full details.

Usage:
  python3 mmlu_english_gptoss.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import time
import threading
import concurrent.futures

import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "https://litellm.uni-osnabrueck.de/v1")
LITELLM_API_KEY  = os.getenv("LITELLM_API_KEY", "")
MODEL_NAME       = "openai/gpt-oss-120b"

client = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)

N_QUESTIONS     = int(os.environ.get("MMLU_N",     570))
QUESTION_OFFSET = int(os.environ.get("MMLU_OFFSET", 20))
MAX_WORKERS     = 5
RATE_LIMIT      = 15

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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE       = "mmlu_en_gptoss"


def _next_version(base: str) -> str:
    existing = glob.glob(os.path.join(_SCRIPT_DIR, f"{base}_v*.jsonl"))
    nums = []
    for f in existing:
        m = re.search(rf"{re.escape(base)}_v(\d+)\.jsonl$", os.path.basename(f))
        if m:
            nums.append(int(m.group(1)))
    return f"{base}_v{max(nums) + 1}" if nums else f"{base}_v1"


_RUN_NAME    = _next_version(_BASE)
OUTPUT_FILE  = f"{_RUN_NAME}.jsonl"
SUMMARY_FILE = f"{_RUN_NAME}_summary.txt"

SYSTEM_PROMPT = (
    "You are a multiple-choice exam assistant. "
    "Respond with ONLY a JSON object in this exact format: {\"answer\": \"X\"} "
    "where X is the single letter A, B, C, or D. "
    "Do not include any other text, explanation, or formatting outside the JSON object."
)


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


def build_prompt(question: str, choices: list[str]) -> str:
    labels       = ["A", "B", "C", "D"]
    choices_text = "\n".join(f"{l}. {t}" for l, t in zip(labels, choices))
    return f"Question: {question}\n\n{choices_text}"


def load_mmlu_sample(subjects: list[str], n: int, offset: int = 0) -> list[dict]:
    per_subject = max(1, n // len(subjects))
    samples: list[dict] = []

    for subject in subjects:
        print(f"  Loading: {subject}...")
        try:
            ds    = load_dataset("cais/mmlu", subject, split="test")
            start = min(offset, len(ds))
            end   = min(offset + per_subject, len(ds))
            if start >= end:
                print(f"  Warning: not enough questions in {subject} at offset {offset} (only {len(ds)} total)")
                continue
            for item in ds.select(range(start, end)):
                samples.append({
                    "subject":       subject,
                    "question":      item["question"],
                    "choices":       item["choices"],
                    "answer_idx":    item["answer"],
                    "answer_letter": ["A", "B", "C", "D"][item["answer"]],
                })
        except Exception as e:
            print(f"  Warning: could not load {subject}: {e}")

    return samples[:n]


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
                return {"text": None, "reasoning": reasoning, "latency": latency, "tokens": tokens, "error": "null content from model"}
            return {"text": content.strip(), "reasoning": reasoning, "latency": latency, "tokens": tokens, "error": None}

        except Exception as e:
            err_str = str(e)
            tqdm.write(f"  [attempt {attempt+1}/{retries}] Error: {err_str[:200]}")

            if "401" in err_str:
                return {"text": None, "latency": None, "error": "unauthorised (401)"}
            if "404" in err_str:
                return {"text": None, "latency": None, "error": "model unavailable (404)"}

            if "No deployments available" in err_str:
                wait = 15
            elif "Rate limit exceeded" in err_str or "429" in err_str:
                wait = 65
            else:
                wait = 2 ** attempt

            if attempt < retries - 1:
                tqdm.write(f"  Waiting {wait}s before retry...")
                time.sleep(wait)

    return {"text": None, "latency": None, "error": "max retries exceeded"}


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
        r'\b(?:the\s+)?(?:correct\s+)?answer\b\s*(?:is\s+|:\s*)?([A-D])\b',
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


def evaluate_question(args: tuple) -> dict:
    i, q     = args
    prompt   = build_prompt(q["question"], q["choices"])
    response = call_model(prompt)

    predicted, parse_method = parse_answer_letter(response["text"])
    is_correct = predicted == q["answer_letter"]

    record = {
        "id":                i + 1,
        "model":             MODEL_NAME,
        "condition":         "no_retrieval",
        "subject":           q["subject"],
        "question":          q["question"],
        "choices":           q["choices"],
        "answer_letter":     q["answer_letter"],
        "prompt_sent":       prompt,
        "response_text":     response["text"],
        "reasoning":         response.get("reasoning"),
        "predicted_letter":  predicted,
        "parse_method":      parse_method,
        "is_correct":        is_correct,
        "latency_s":         response["latency"],
        "prompt_tokens":     (response.get("tokens") or {}).get("prompt_tokens"),
        "completion_tokens": (response.get("tokens") or {}).get("completion_tokens"),
        "total_tokens":      (response.get("tokens") or {}).get("total_tokens"),
        "error":             response["error"],
    }

    with write_lock:
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    return record


def run_evaluation():
    print("=" * 60)
    print("MMLU Evaluation  —  English  —  gpt-oss-120b")
    print(f"Model    : {MODEL_NAME}")
    print(f"Gateway  : {LITELLM_BASE_URL}")
    print(f"Subjects : {len(SUBJECTS)} subjects × ~10 questions each (offset={QUESTION_OFFSET})")
    print(f"N        : {N_QUESTIONS} questions")
    print(f"Workers  : {MAX_WORKERS}  |  Rate limit: {RATE_LIMIT} req/min")
    print(f"Output   : {OUTPUT_FILE}")
    print("=" * 60)

    print("\nLoading MMLU data...")
    questions = load_mmlu_sample(SUBJECTS, N_QUESTIONS, offset=QUESTION_OFFSET)
    print(f"Loaded {len(questions)} questions.\n")

    if len(questions) == 0:
        print("No questions available at this offset — dataset fully covered.")
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
        "=" * 60,
        "RESULTS SUMMARY",
        "=" * 60,
        f"Model           : {MODEL_NAME}",
        f"Total questions : {len(questions)}",
        f"Answered        : {answered}",
        f"Errors          : {errors}",
        f"Correct         : {correct}",
        f"Accuracy        : {accuracy:.1%}",
        "",
    ]

    if not df_answered.empty:
        lines.append(f"  {'Subject':<35} {'Correct':>7}  {'Total':>5}  {'Accuracy':>8}")
        lines.append(f"  {'-'*35} {'-'*7}  {'-'*5}  {'-'*8}")
        breakdown = df_answered.groupby("subject")["is_correct"].agg(["sum", "count"])
        breakdown["accuracy"] = breakdown["sum"] / breakdown["count"]
        for subject, row in breakdown.iterrows():
            lines.append(
                f"  {subject:<35} {int(row['sum']):>7}  {int(row['count']):>5}  {row['accuracy']:>8.1%}"
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
