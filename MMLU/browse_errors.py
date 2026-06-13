"""
browse_errors.py
----------------
Browse only error questions across all 3 models.
Shows the reasoning chain so you can see where the model ran out of tokens.

Usage:
  python3 browse_errors.py                     # all errors across all models
  python3 browse_errors.py --model qwen        # only qwen errors (partial match)
  python3 browse_errors.py --subject law       # only errors in matching subjects

Controls:
  Enter       → next
  b + Enter   → back
  q + Enter   → quit
  <number>    → jump to that position
"""

import argparse
import glob
import json
import os
import textwrap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

MODEL_PATTERNS = {
    "gpt-oss-120b": "mmlu_en_gptoss_v*.jsonl",
    "Gemma-4-31B":  "mmlu_en_gemma4_v*.jsonl",
    "Qwen3.5-122B": "mmlu_en_qwen3.5_v*.jsonl",
}


def load_errors(model_filter: str = None, subject_filter: str = None) -> list[dict]:
    errors = []
    for model_name, pattern in MODEL_PATTERNS.items():
        if model_filter and model_filter.lower() not in model_name.lower():
            continue
        files = sorted(glob.glob(os.path.join(SCRIPT_DIR, pattern)))
        files = [f for f in files if "pilot" not in os.path.basename(f)]
        for fpath in files:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if not r.get("error"):
                        continue
                    if subject_filter and subject_filter.lower() not in r.get("subject", "").lower():
                        continue
                    r["_model_label"] = model_name
                    r["_file"] = os.path.basename(fpath)
                    errors.append(r)
    return errors


def wrap(text: str, width: int = 88, indent: str = "  ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def display_error(idx: int, total: int, r: dict) -> None:
    os.system("clear" if os.name != "nt" else "cls")

    subject  = r.get("subject", "?")
    question = r.get("question", "?")
    choices  = r.get("choices", [])
    gold     = r.get("answer_letter", "?")
    error    = r.get("error", "?")
    reasoning = r.get("reasoning") or ""
    latency  = r.get("latency_s")
    comp_tok = r.get("completion_tokens")
    model    = r["_model_label"]
    fname    = r["_file"]

    print(f"{BOLD}{'─'*70}{RESET}")
    print(f"{BOLD}  Error {idx}/{total}   [{subject}]   {model}{RESET}")
    print(f"  {DIM}file: {fname}{RESET}")
    print(f"{'─'*70}")
    print()
    print(wrap(question))
    print()

    labels = ["A", "B", "C", "D"]
    for label, choice in zip(labels, choices):
        marker = f"{GREEN}✓{RESET}" if label == gold else " "
        print(f"    {marker} {BOLD}{label}.{RESET} {choice}")

    print()
    print(f"  {BOLD}Gold answer : {GREEN}{gold}{RESET}")
    print(f"  {BOLD}Error       : {RED}{error}{RESET}")
    if latency:
        print(f"  Latency     : {latency:.1f}s")
    if comp_tok:
        limit_hit = comp_tok >= 8192
        tok_color = RED if limit_hit else RESET
        print(f"  Completion tokens : {tok_color}{comp_tok}{RESET}{'  ← hit 8192 token ceiling' if limit_hit else ''}")

    print(f"\n{'─'*70}")

    if reasoning:
        print(f"\n{BOLD}{CYAN}  ── Reasoning chain ──{RESET}")
        print(f"  {DIM}(model thought process — no final answer was written){RESET}\n")
        lines = reasoning.strip().split("\n")
        for line in lines:
            if line.strip():
                print(f"  {DIM}{line}{RESET}")
            else:
                print()
    else:
        print(f"\n  {DIM}(no reasoning captured — API returned nothing){RESET}")

    print(f"\n{'─'*70}")


def main():
    parser = argparse.ArgumentParser(description="Browse MMLU error questions")
    parser.add_argument("--model",   type=str, default=None, help="Filter by model name (partial match)")
    parser.add_argument("--subject", type=str, default=None, help="Filter by subject (partial match)")
    args = parser.parse_args()

    print("Loading errors...")
    errors = load_errors(model_filter=args.model, subject_filter=args.subject)

    if not errors:
        print("No errors found matching the filters.")
        return

    from collections import Counter
    by_model = Counter(r["_model_label"] for r in errors)
    print(f"\nFound {len(errors)} errors:")
    for model, count in by_model.most_common():
        print(f"  {model}: {count}")
    print("\nPress Enter to start...")
    input()

    pos = 0
    while 0 <= pos < len(errors):
        display_error(pos + 1, len(errors), errors[pos])
        print(f"\n  {DIM}[Enter]=next  [b]=back  [q]=quit  [number]=jump{RESET}  ", end="")
        cmd = input().strip().lower()

        if cmd == "q":
            break
        elif cmd == "b":
            pos = max(0, pos - 1)
        elif cmd.isdigit():
            target = int(cmd) - 1
            if 0 <= target < len(errors):
                pos = target
            else:
                print(f"  Out of range (1–{len(errors)}). Press Enter...")
                input()
        else:
            pos += 1

    print("\nDone.")


if __name__ == "__main__":
    main()
