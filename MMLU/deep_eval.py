"""
deep_eval.py
------------
Re-queries the API for a targeted subset of questions and captures the
full chain-of-thought reasoning so you can evaluate WHY a model answered
the way it did.

Use this for qualitative thesis analysis — run it on wrong answers and
disagreements, then read the reasoning to understand model behaviour.

Usage:
  python3 deep_eval.py --mode wrong                  # all questions any model got wrong
  python3 deep_eval.py --mode disagree               # questions where models gave different answers
  python3 deep_eval.py --mode wrong --subject math   # wrong answers in mathematics only
  python3 deep_eval.py --ids 40 54 219               # specific question ids
  python3 deep_eval.py --mode wrong --model gptoss   # wrong answers for one model only

Output: deep_eval_<timestamp>.jsonl  +  deep_eval_<timestamp>_summary.txt
        Each record includes the full reasoning chain.
"""

from __future__ import annotations

import json
import time
import argparse
import threading
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm

# ── Credentials ───────────────────────────────────────────────────────────────
LITELLM_BASE_URL = "https://litellm.uni-osnabrueck.de/v1"
LITELLM_API_KEY  = os.getenv("LITELLM_API_KEY", "")

MODELS = {
    "gptoss" : "openai/gpt-oss-120b",
    "gemma4" : "google/gemma-4-31B-it",
    "qwen3.5": "Qwen/Qwen3.5-122B-A10B-FP8",
}

def _latest(pattern: str) -> str:
    import glob as _glob, os as _os
    script_dir = _os.path.dirname(_os.path.abspath(__file__))
    matches    = sorted(_glob.glob(_os.path.join(script_dir, pattern)))
    return _os.path.basename(matches[-1]) if matches else ""

RESULT_FILES = {
    "gptoss" : _latest("mmlu_en_gptoss_v*.jsonl"),
    "gemma4" : _latest("mmlu_en_gemma4_v*.jsonl"),
    "qwen3.5": _latest("mmlu_en_qwen3.5_v*.jsonl"),
}

SYSTEM_PROMPT = (
    "You are a multiple-choice exam assistant. "
    "Respond with ONLY a JSON object in this exact format: {\"answer\": \"X\"} "
    "where X is the single letter A, B, C, or D. "
    "Do not include any other text, explanation, or formatting outside the JSON object."
)

client    = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)
_lock     = threading.Lock()
_last_req = 0.0
RATE_GAP  = 4.0   # seconds between requests (15/min)

GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def rate_wait():
    global _last_req
    with _lock:
        now  = time.time()
        wait = RATE_GAP - (now - _last_req)
        if wait > 0:
            time.sleep(wait)
        _last_req = time.time()


def load_results(fname: str) -> dict[int, dict]:
    out = {}
    try:
        with open(fname) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    out[r["id"]] = r
    except FileNotFoundError:
        pass
    return out


def build_prompt(question: str, choices: list[str]) -> str:
    labels = ["A", "B", "C", "D"]
    return "Question: " + question + "\n\n" + \
           "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))


def query_model(model_key: str, prompt: str, retries: int = 3) -> dict:
    model_name = MODELS[model_key]
    for attempt in range(retries):
        rate_wait()
        try:
            t0       = time.time()
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=4096,
                temperature=0.0,
            )
            latency   = round(time.time() - t0, 3)
            msg       = response.choices[0].message
            content   = msg.content
            reasoning = (
                getattr(msg, "reasoning_content", None)
                or (msg.model_extra or {}).get("reasoning_content")
            )
            if content is None:
                return {"text": None, "reasoning": reasoning, "latency": latency,
                        "error": "null content from model"}
            return {"text": content.strip(), "reasoning": reasoning,
                    "latency": latency, "error": None}
        except Exception as e:
            err = str(e)
            tqdm.write(f"  [attempt {attempt+1}] {model_key} error: {err[:150]}")
            if "401" in err or "404" in err:
                return {"text": None, "reasoning": None, "latency": None, "error": err[:100]}
            wait = 65 if "Rate limit exceeded" in err else (15 if "No deployments" in err else 2**attempt)
            if attempt < retries - 1:
                time.sleep(wait)
    return {"text": None, "reasoning": None, "latency": None, "error": "max retries exceeded"}


def parse_answer(text: str | None) -> str | None:
    import re, json as js
    if not text:
        return None
    t = text.strip()
    try:
        clean = re.sub(r'^```[a-z]*\s*|\s*```$', '', t, flags=re.IGNORECASE).strip()
        data  = js.loads(clean)
        if isinstance(data, dict):
            letter = str(data.get("answer", "")).strip().upper()
            if letter in {"A", "B", "C", "D"}:
                return letter
    except Exception:
        pass
    m = re.search(r'\b([A-D])\b', t)
    return m.group(1).upper() if m else None


def display_result(r: dict, model_key: str) -> None:
    gold      = r["answer_letter"]
    question  = r["question"]
    choices   = r["choices"]
    predicted = r.get("predicted_letter_rerun") or "?"
    correct   = predicted == gold
    reasoning = r.get("reasoning_rerun") or ""
    orig_pred = r.get("original_predicted") or "?"
    orig_ok   = orig_pred == gold

    labels = ["A", "B", "C", "D"]
    marker_col = GREEN if correct else RED

    print(f"\n{BOLD}{'─'*72}{RESET}")
    print(f"{BOLD}  id={r['id']}  [{r['subject']}]  model={MODELS[model_key]}{RESET}")
    print(f"{'─'*72}")
    print()
    # Question
    import textwrap
    print(textwrap.fill(question, width=90, initial_indent="  ", subsequent_indent="  "))
    print()
    for label, choice in zip(labels, choices):
        tick = f"{GREEN}✓{RESET}" if label == gold else " "
        print(f"    {tick} {BOLD}{label}.{RESET} {choice}")
    print()
    print(f"  Gold:      {GREEN}{BOLD}{gold}{RESET}")
    orig_col = GREEN if orig_ok else RED
    print(f"  Original:  {orig_col}{orig_pred}{RESET}  ({'correct' if orig_ok else 'WRONG'} in saved results)")
    print(f"  Re-run:    {marker_col}{predicted}{RESET}  ({'correct' if correct else 'WRONG'})")
    print()

    if reasoning:
        print(f"  {CYAN}{BOLD}── Chain of thought ──────────────────────────────────────────{RESET}")
        # Word-wrap reasoning
        wrapped = textwrap.fill(reasoning, width=88,
                                initial_indent="  ", subsequent_indent="  ")
        print(wrapped)
        print()
    else:
        print(f"  {DIM}(no reasoning content — model does not expose chain-of-thought){RESET}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["wrong", "disagree", "all"], default="wrong")
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--model",   choices=list(MODELS.keys()), default=None,
                        help="Restrict to one model (default: all)")
    parser.add_argument("--ids",     type=int, nargs="+", default=None,
                        help="Specific question IDs to re-run")
    args = parser.parse_args()

    print("Loading saved results...")
    saved = {key: load_results(fname) for key, fname in RESULT_FILES.items()}

    model_keys = [args.model] if args.model else list(MODELS.keys())

    # Build list of (model_key, q_id, original_record) to re-run
    tasks: list[tuple[str, int, dict]] = []

    if args.ids:
        for model_key in model_keys:
            for q_id in args.ids:
                rec = saved[model_key].get(q_id)
                if rec:
                    tasks.append((model_key, q_id, rec))
    else:
        all_ids = sorted(set().union(*[d.keys() for d in saved.values()]))
        for q_id in all_ids:
            records = {k: saved[k].get(q_id) for k in model_keys}
            base    = next((r for r in records.values() if r), None)
            if base is None:
                continue

            # Subject filter
            if args.subject and args.subject.lower() not in base.get("subject", "").lower():
                continue

            answers = {k: r.get("predicted_letter") for k, r in records.items() if r and not r.get("error")}

            if args.mode == "wrong":
                for k, r in records.items():
                    if r and r.get("predicted_letter") != r.get("answer_letter") and not r.get("error"):
                        tasks.append((k, q_id, r))
            elif args.mode == "disagree":
                unique = set(v for v in answers.values() if v)
                if len(unique) > 1:
                    for k, r in records.items():
                        if r:
                            tasks.append((k, q_id, r))
            else:  # all
                for k, r in records.items():
                    if r:
                        tasks.append((k, q_id, r))

    if not tasks:
        print("No questions match the filters.")
        return

    print(f"\nFound {len(tasks)} model×question pairs to re-run.")
    print(f"At 15 req/min this will take ~{len(tasks)*4//60}m {len(tasks)*4%60}s.")
    print("Press Enter to start, or Ctrl+C to cancel...")
    input()

    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"deep_eval_{ts}.jsonl"
    results_out : list[dict] = []

    for model_key, q_id, orig_rec in tqdm(tasks, desc="Re-running", unit="q"):
        prompt   = build_prompt(orig_rec["question"], orig_rec["choices"])
        response = query_model(model_key, prompt)

        predicted = parse_answer(response["text"])
        is_correct = predicted == orig_rec["answer_letter"]

        record = {
            "model_key"         : model_key,
            "model_name"        : MODELS[model_key],
            "id"                : q_id,
            "subject"           : orig_rec["subject"],
            "question"          : orig_rec["question"],
            "choices"           : orig_rec["choices"],
            "answer_letter"     : orig_rec["answer_letter"],
            "original_predicted": orig_rec.get("predicted_letter"),
            "predicted_rerun"   : predicted,
            "is_correct_rerun"  : is_correct,
            "reasoning_rerun"   : response["reasoning"],
            "response_text_rerun": response["text"],
            "latency_s"         : response["latency"],
            "error"             : response["error"],
        }
        results_out.append(record)

        with open(output_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ── Interactive browser over re-run results ────────────────────────────────
    print(f"\n\nRe-run complete. {len(results_out)} records saved to {output_file}")
    print("Press Enter to browse results interactively (q to quit)...")
    input()

    pos = 0
    while 0 <= pos < len(results_out):
        r = results_out[pos]
        r2 = {**r,
              "predicted_letter_rerun": r["predicted_rerun"],
              "reasoning_rerun"       : r.get("reasoning_rerun"),
              "original_predicted"    : r.get("original_predicted")}
        display_result(r2, r["model_key"])

        print(f"  {DIM}[Enter]=next  [b]=back  [q]=quit  ({pos+1}/{len(results_out)}){RESET}  ", end="")
        cmd = input().strip().lower()
        if cmd == "q":
            break
        elif cmd == "b":
            pos = max(0, pos - 1)
        else:
            pos += 1

    print("\nDone.")


if __name__ == "__main__":
    main()
