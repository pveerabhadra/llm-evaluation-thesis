from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # = MMLU German/
STATE_FILE = os.path.join(SCRIPT_DIR, "batch_state_german.json")

BATCH_N     = 1000
N_SUBJECTS  = 57
PER_SUBJECT = BATCH_N // N_SUBJECTS   # = 17

MODEL_SCRIPTS = [
    ("gemma4",  "german_gemma4.py"),
]


def load_state() -> dict:
    """Load state file. Initialises at offset=0 (German starts fresh)."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "batch_offset"           : 0,
        "batch_n"                : BATCH_N,
        "per_subject"            : PER_SUBJECT,
        "models_done_this_batch" : [],
        "history"                : [],
    }


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def advance_batch(state: dict) -> dict:
    """Record the completed batch and move to the next offset."""
    state["history"].append({
        "offset"     : state["batch_offset"],
        "per_subject": state["per_subject"],
        "completed"  : state["models_done_this_batch"],
        "finished"   : datetime.now().isoformat(timespec="seconds"),
    })
    state["batch_offset"]           += state["per_subject"]
    state["models_done_this_batch"]  = []
    return state


def run_model(script_name: str, offset: int, n: int) -> bool:
    """Run one model script via env vars. Returns True on success."""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    env = {**os.environ, "MMLU_OFFSET": str(offset), "MMLU_N": str(n)}

    print(f"\n{'='*60}")
    print(f"  Running : {script_name}")
    print(f"  Offset  : {offset}  (questions {offset}–{offset + n // N_SUBJECTS - 1} per subject)")
    print(f"  N       : {n} questions total  (~{n // N_SUBJECTS} per subject)")
    print(f"  Started : {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    t0     = time.time()
    result = subprocess.run([sys.executable, script_path], env=env, cwd=SCRIPT_DIR)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    if result.returncode == 0:
        print(f"  DONE    : {script_name}")
        print(f"  Time    : {elapsed / 60:.1f} minutes")
        print(f"  Finished: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")
        return True
    else:
        print(f"  ERROR   : {script_name} exited with code {result.returncode}")
        print(f"{'='*60}\n")
        return False


def print_status(state: dict) -> None:
    all_models = [m for m, _ in MODEL_SCRIPTS]
    pending    = [m for m in all_models if m not in state["models_done_this_batch"]]

    print(f"\n{'─'*55}")
    print(f"  batch_state_german.json  —  German MMLU")
    print(f"{'─'*55}")
    print(f"  Current batch offset  : {state['batch_offset']}")
    print(f"  Questions per batch   : {state['batch_n']}  (~{state['per_subject']} per subject)")
    print(f"  Models done this batch: {state['models_done_this_batch'] or '(none yet)'}")
    print(f"  Models still pending  : {pending or '(all done — run --next-batch)'}")
    if state.get("history"):
        print(f"\n  Completed batches ({len(state['history'])} total):")
        for h in state["history"]:
            print(f"    offset={h['offset']:>4}  per_subject={h['per_subject']}  "
                  f"models={h['completed']}  finished={h['finished']}")
    print(f"{'─'*55}\n")


def count_files(model_key: str) -> int:
    """Return the number of existing JSONL files for a model."""
    pattern = os.path.join(SCRIPT_DIR, f"mmlu_de_{model_key}_v*.jsonl")
    return len(glob.glob(pattern))


def count_new_records(model_key: str, files_before: int) -> int:
    """Count records in files created since files_before snapshot."""
    pattern = os.path.join(SCRIPT_DIR, f"mmlu_de_{model_key}_v*.jsonl")
    files   = sorted(glob.glob(pattern))
    new_files = files[files_before:]
    total = 0
    for f in new_files:
        with open(f) as fh:
            total += sum(1 for line in fh if line.strip())
    return total


def main():
    parser = argparse.ArgumentParser(description="Run German MMLU batches sequentially")
    parser.add_argument("--only",       type=str, default=None,
                        help="Run only this model for the current batch (gptoss | gemma4 | qwen3.5)")
    parser.add_argument("--once",       action="store_true",
                        help="Run only one batch then stop")
    parser.add_argument("--status",     action="store_true",
                        help="Show current state and exit")
    parser.add_argument("--next-batch", action="store_true",
                        help="Force-advance to next batch offset")
    args = parser.parse_args()

    state = load_state()

    if args.status:
        print_status(state)
        return

    if args.next_batch:
        print(f"Advancing from offset {state['batch_offset']} → "
              f"{state['batch_offset'] + state['per_subject']}")
        state = advance_batch(state)
        save_state(state)
        print_status(state)
        return

    all_models   = [m for m, _ in MODEL_SCRIPTS]
    batch_number = 0

    while True:
        batch_number += 1
        offset = state["batch_offset"]
        n      = state["batch_n"]

        if args.only:
            valid = [m for m, _ in MODEL_SCRIPTS]
            if args.only not in valid:
                print(f"Unknown model '{args.only}'. Choose from: {valid}")
                sys.exit(1)
            pending_models = [(m, s) for m, s in MODEL_SCRIPTS if m == args.only]
        else:
            pending_models = [
                (m, s) for m, s in MODEL_SCRIPTS
                if m not in state["models_done_this_batch"]
            ]

        if not pending_models:
            print(f"\nAll 3 models already completed for batch offset={offset}.")
            if args.once:
                print_status(state)
                return
            state = advance_batch(state)
            save_state(state)
            continue

        print(f"\n{'#'*60}")
        print(f"  BATCH {batch_number}  —  German MMLU  —  offset={offset}  (~{n // N_SUBJECTS} q/subject)")
        print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Models  : {[m for m, _ in pending_models]}")
        print(f"{'#'*60}")

        files_before = {m: count_files(m) for m, _ in MODEL_SCRIPTS}

        for model_name, script_name in pending_models:
            success = run_model(script_name, offset, n)
            if success:
                if model_name not in state["models_done_this_batch"]:
                    state["models_done_this_batch"].append(model_name)
                save_state(state)
            else:
                print(f"  Stopping — {model_name} failed. Fix the issue and run again.")
                sys.exit(1)

        if all(m in state["models_done_this_batch"] for m in all_models):
            active_models = [m for m, _ in MODEL_SCRIPTS]
            total_written = sum(count_new_records(m, files_before[m]) for m in active_models)

            print(f"\n{'*'*60}")
            print(f"  ALL MODELS DONE — batch offset={offset}")
            print(f"  Questions written this batch: {total_written}")
            print(f"{'*'*60}\n")

            state = advance_batch(state)
            save_state(state)

            if total_written == 0:
                print(f"\n{'='*60}")
                print(f"  DATASET FULLY COVERED — no more German questions available.")
                print(f"  Run: python3 ../analysis/combine_results.py --lang de --save")
                print(f"{'='*60}\n")
                print_status(state)
                return

            if args.once or args.only:
                print_status(state)
                return

            print(f"  Next batch starts at offset={state['batch_offset']} — continuing…\n")

        else:
            remaining = [m for m in all_models if m not in state["models_done_this_batch"]]
            print(f"\nCompleted so far: {state['models_done_this_batch']}")
            print(f"Still pending   : {remaining}")
            if args.only:
                return
            print(f"Continuing with: {remaining[0]}\n")


if __name__ == "__main__":
    main()
