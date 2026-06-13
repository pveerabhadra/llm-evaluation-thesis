"""
browse_scidqa.py
----------------
Interactive browser for SciDQA reading comprehension results.
Shows question, gold answer, and model responses side by side per condition.

Usage:
  python3 browse_scidqa.py                        # browse all questions
  python3 browse_scidqa.py --condition long_context
  python3 browse_scidqa.py --condition no_retrieval
  python3 browse_scidqa.py --id 5                 # jump to question id 5

Controls:
  Enter       → next
  b + Enter   → back
  r + Enter   → toggle reasoning chain
  q + Enter   → quit
  <number>    → jump to that position
"""

import argparse
import glob
import json
import os
import textwrap
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

CONDITIONS = ["no_retrieval", "long_context"]


def _latest(pattern: str):
    matches = sorted(glob.glob(os.path.join(SCRIPT_DIR, pattern)))
    return matches[-1] if matches else None


def load_results(path: str) -> dict:
    """Load jsonl and group by question id → condition → record."""
    grouped = defaultdict(dict)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            grouped[r["id"]][r["condition"]] = r
    return grouped


def wrap(text: str, width: int = 86, indent: str = "  ") -> str:
    if not text:
        return f"{indent}(empty)"
    lines = []
    for para in str(text).split("\n"):
        if para.strip():
            lines.append(textwrap.fill(para, width=width,
                                       initial_indent=indent,
                                       subsequent_indent=indent))
        else:
            lines.append("")
    return "\n".join(lines)


def display(idx: int, total: int, q_id: int, by_cond: dict,
            filter_cond: str = None, show_reasoning: bool = False) -> None:
    os.system("clear" if os.name != "nt" else "cls")

    base = next(iter(by_cond.values()))
    print(f"{BOLD}{'─'*72}{RESET}")
    print(f"{BOLD}  Q {idx}/{total}   id={q_id}   pid={base.get('pid','')}   "
          f"[{base.get('venue','')} {base.get('year','')}]{RESET}")
    print(f"{'─'*72}")
    print()
    print(f"{BOLD}  Question:{RESET}")
    print(wrap(base.get("question", ""), indent="    "))
    print()
    print(f"{BOLD}  Gold answer:{RESET}")
    print(f"{GREEN}{wrap(base.get('gold_answer',''), indent='    ')}{RESET}")
    print()

    conds_to_show = [filter_cond] if filter_cond else CONDITIONS
    for cond in conds_to_show:
        r = by_cond.get(cond)
        if r is None:
            continue

        error    = r.get("error")
        response = r.get("response_text") or ""
        latency  = r.get("latency_s")
        ptok     = r.get("prompt_tokens")
        ctok     = r.get("completion_tokens")
        pchars   = r.get("paper_chars_given", 0)

        lat_str  = f"{latency:.1f}s" if latency else "—"
        tok_str  = f"prompt={ptok:,}  completion={ctok}" if ptok else ""
        ctx_str  = f"  paper={pchars:,} chars" if pchars else ""

        print(f"{'─'*72}")
        cond_label = cond.upper().replace("_", " ")
        print(f"{BOLD}  [{cond_label}]{RESET}  latency={lat_str}  {DIM}{tok_str}{ctx_str}{RESET}")
        print()

        if error:
            print(f"  {RED}ERROR: {error}{RESET}")
        else:
            print(wrap(response, indent="    "))

        if show_reasoning:
            reasoning = r.get("reasoning")
            if reasoning:
                print(f"\n{BOLD}{CYAN}  ── Reasoning ──{RESET}")
                for line in reasoning.strip().split("\n"):
                    if line.strip():
                        print(f"  {DIM}{line}{RESET}")

        print()

    print(f"{'─'*72}")
    has_reasoning = any(
        by_cond.get(c, {}).get("reasoning")
        for c in conds_to_show
    )
    if has_reasoning and not show_reasoning:
        print(f"  {DIM}Press r to show reasoning chain{RESET}")


def main():
    parser = argparse.ArgumentParser(description="Browse SciDQA results")
    parser.add_argument("--condition", type=str, default=None,
                        help="Filter to one condition: no_retrieval or long_context")
    parser.add_argument("--id", type=int, default=None,
                        help="Jump to a specific question id")
    args = parser.parse_args()

    fpath = _latest("scidqa_pilot_gptoss_v*.jsonl")
    if fpath is None:
        print("No result file found. Run scidqa_pilot_gptoss.py first.")
        return

    print(f"Loading {os.path.basename(fpath)}...")
    grouped = load_results(fpath)
    q_ids   = sorted(grouped.keys())

    if not q_ids:
        print("No records found.")
        return

    print(f"  {len(q_ids)} questions loaded.")
    print(f"  Conditions: {list({c for d in grouped.values() for c in d})}")
    if args.condition:
        print(f"  Filtering to: {args.condition}")
    print("\nPress Enter to start...")
    input()

    start = 0
    if args.id is not None:
        if args.id in q_ids:
            start = q_ids.index(args.id)
        else:
            print(f"Question id={args.id} not found.")
            return

    pos            = start
    show_reasoning = False

    while 0 <= pos < len(q_ids):
        q_id    = q_ids[pos]
        by_cond = grouped[q_id]
        display(pos + 1, len(q_ids), q_id, by_cond,
                filter_cond=args.condition, show_reasoning=show_reasoning)

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
            if 0 <= target < len(q_ids):
                pos = target
            else:
                print(f"  Out of range (1–{len(q_ids)}). Press Enter...")
                input()
        else:
            show_reasoning = False
            pos += 1

    print("\nDone.")


if __name__ == "__main__":
    main()
