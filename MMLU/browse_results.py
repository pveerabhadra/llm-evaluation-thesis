"""
browse_results.py
-----------------
Interactive browser for MMLU results across all 3 models.

Usage:
  python3 browse_results.py                        # browse all questions
  python3 browse_results.py --wrong                # only questions where any model got it wrong
  python3 browse_results.py --disagree             # only questions where models disagreed
  python3 browse_results.py --subject biology      # filter by subject (partial match)
  python3 browse_results.py --id 40                # jump straight to question id 40

Controls:
  Enter       → next question
  b + Enter   → back
  r + Enter   → toggle reasoning chain
  q + Enter   → quit
  <number>    → jump to that position
"""

import json
import argparse
import glob
import os
import textwrap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def _latest(pattern: str):
    matches = sorted(glob.glob(os.path.join(SCRIPT_DIR, pattern)))
    return os.path.basename(matches[-1]) if matches else None


FILES = {
    "gpt-oss-120b": _latest("mmlu_en_gptoss_v*.jsonl"),
    "gemma-4-31B":  _latest("mmlu_en_gemma4_v*.jsonl"),
    "Qwen3.5-122B": _latest("mmlu_en_qwen3.5_v*.jsonl"),
}


def load_model(fname: str) -> dict:
    path = os.path.join(SCRIPT_DIR, fname)
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["id"]] = r
    return out


def wrap(text: str, width: int = 90, indent: str = "  ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def display_question(idx: int, total: int, q_id: int, models: dict, show_reasoning: bool = False) -> None:
    os.system("clear" if os.name != "nt" else "cls")

    base     = next(r for r in models.values() if r is not None)
    subject  = base.get("subject", "?")
    question = base.get("question", "?")
    choices  = base.get("choices", [])
    gold     = base.get("answer_letter", "?")

    print(f"{BOLD}{'─'*70}{RESET}")
    print(f"{BOLD}  Question {idx}/{total}   id={q_id}   [{subject}]{RESET}")
    print(f"{'─'*70}")
    print()
    print(wrap(question, width=90))
    print()
    for label, choice in zip(["A", "B", "C", "D"], choices):
        marker = f"{GREEN}✓{RESET}" if label == gold else " "
        print(f"    {marker} {BOLD}{label}.{RESET} {choice}")
    print()
    print(f"  {BOLD}Gold answer: {GREEN}{gold}{RESET}")
    print()
    print(f"{'─'*70}")
    print(f"  {'Model':<20} {'Predicted':>9}  {'Correct':>8}  {'Latency':>9}  Response")
    print(f"  {'─'*20} {'─'*9}  {'─'*8}  {'─'*9}  {'─'*20}")

    for model_name, r in models.items():
        if r is None:
            print(f"  {model_name:<20} {'N/A':>9}  {'—':>8}  {'—':>9}  (no record)")
            continue

        predicted  = r.get("predicted_letter") or "?"
        error      = r.get("error")
        latency    = r.get("latency_s")
        response   = r.get("response_text") or ""
        is_correct = r.get("is_correct", False)
        lat_str    = f"{latency:.1f}s" if latency else "—"

        if error:
            status_str = f"{RED}ERROR{RESET}"
            pred_str   = f"{RED}{predicted:>9}{RESET}"
        elif is_correct:
            status_str = f"{GREEN}correct{RESET}"
            pred_str   = f"{GREEN}{predicted:>9}{RESET}"
        else:
            status_str = f"{RED}WRONG{RESET}"
            pred_str   = f"{RED}{predicted:>9}{RESET}"

        resp_display = response.strip()[:50].replace("\n", " ")
        if error:
            resp_display = f"{DIM}{error[:45]}{RESET}"

        print(f"  {model_name:<20} {pred_str}  {status_str:>16}  {lat_str:>9}  {DIM}{resp_display}{RESET}")

    print(f"{'─'*70}")

    if show_reasoning:
        any_reasoning = False
        for model_name, r in models.items():
            if r is None:
                continue
            reasoning = r.get("reasoning")
            if reasoning:
                any_reasoning = True
                print(f"\n{BOLD}{CYAN}  ── Reasoning: {model_name} ──{RESET}")
                for line in reasoning.strip().split("\n"):
                    print(f"  {DIM}{line}{RESET}")
        if not any_reasoning:
            print(f"\n  {DIM}(no reasoning captured for this question){RESET}")
        print(f"\n{'─'*70}")
    else:
        has_reasoning = any(r.get("reasoning") for r in models.values() if r is not None)
        if has_reasoning:
            print(f"  {DIM}Press r to show reasoning chain{RESET}")


def main():
    parser = argparse.ArgumentParser(description="Browse MMLU results interactively")
    parser.add_argument("--wrong",    action="store_true")
    parser.add_argument("--disagree", action="store_true")
    parser.add_argument("--subject",  type=str, default=None)
    parser.add_argument("--id",       type=int, default=None)
    args = parser.parse_args()

    print("Loading results...")
    data = {}
    for name, fname in FILES.items():
        if fname is None:
            print(f"  [skip] {name}: no result file found yet")
        else:
            data[name] = load_model(fname)
    if not data:
        print("No result files found. Run the evaluation scripts first.")
        return

    all_ids = sorted(set().union(*[d.keys() for d in data.values()]))

    filtered = []
    for q_id in all_ids:
        records = {name: data[name].get(q_id) for name in data}
        base    = next((r for r in records.values() if r), None)
        if base is None:
            continue
        if args.subject and args.subject.lower() not in base.get("subject", "").lower():
            continue
        answers     = [r.get("predicted_letter") for r in records.values() if r and not r.get("error")]
        all_correct = all(r.get("is_correct") for r in records.values() if r and not r.get("error"))
        if args.wrong and all_correct:
            continue
        if args.disagree and len(set(a for a in answers if a)) <= 1:
            continue
        filtered.append((q_id, records))

    if not filtered:
        print("No questions match the filters.")
        return

    print(f"Found {len(filtered)} questions. Press Enter to start...")
    input()

    start = 0
    if args.id is not None:
        matches = [i for i, (q_id, _) in enumerate(filtered) if q_id == args.id]
        if matches:
            start = matches[0]
        else:
            print(f"Question id={args.id} not found.")
            return

    pos = start
    show_reasoning = False
    while 0 <= pos < len(filtered):
        q_id, records = filtered[pos]
        display_question(pos + 1, len(filtered), q_id, records, show_reasoning=show_reasoning)
        print(f"\n  {DIM}[Enter]=next  [b]=back  [r]=reasoning  [q]=quit  [number]=jump{RESET}  ", end="")
        cmd = input().strip().lower()

        if cmd == "q":
            break
        elif cmd == "b":
            show_reasoning = False
            pos = max(0, pos - 1)
        elif cmd == "r":
            show_reasoning = not show_reasoning
        elif cmd.isdigit():
            show_reasoning = False
            target = int(cmd) - 1
            if 0 <= target < len(filtered):
                pos = target
            else:
                print(f"  Out of range (1–{len(filtered)}). Press Enter...")
                input()
        else:
            show_reasoning = False
            pos += 1

    print("\nDone.")


if __name__ == "__main__":
    main()
