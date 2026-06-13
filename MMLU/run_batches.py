"""
run_batches.py
--------------
Runs English MMLU evaluations for Gemma-4 (FP8), tracking progress
so the same questions are never repeated across batches.

By default, runs ALL remaining batches automatically until the full
dataset is covered. Stop any time with Ctrl+C — state is saved after
each model finishes, so it always resumes from where it left off.

Usage:
  caffeinate -i python3 run_batches.py          # run ALL remaining batches (recommended)
  python3 run_batches.py --once                 # run only the next single batch, then stop
  python3 run_batches.py --status               # show current state, don't run anything
  python3 run_batches.py --next-batch           # force-advance to the next batch offset

Batch size: 1000 questions (~17 per subject × 57 subjects)
Model: gemma4 (RedHatAI/gemma-4-31B-it-FP8-Dynamic)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
STATE_FILE  = os.path.join(SCRIPT_DIR, "batch_state.json")

BATCH_N        = 1000          # questions per batch
N_SUBJECTS     = 57
PER_SUBJECT    = BATCH_N // N_SUBJECTS   # = 17

MODEL_SCRIPTS = [
    ("gemma4",  "mmlu_english_gemma4.py"),
]

# ── State helpers ──────────────────────────────────────────────────────────────

def load_state() -> dict:
    """Load state file, initialising to offset=20 if it doesn't exist yet."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    # Batches 0 and 10 are already done for all models → start at 20
    return {
        "batch_offset"  : 20,
        "batch_n"       : BATCH_N,
        "per_subject"   : PER_SUBJECT,
        "models_done_this_batch": [],
        "history": [],
    }

def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def advance_batch(state: dict) -> dict:
    """Move to the next batch offset once all 3 models are done."""
    completed_offset = state["batch_offset"]
    state["history"].append({
        "offset"    : completed_offset,
        "per_subject": state["per_subject"],
        "completed" : state["models_done_this_batch"],
        "finished"  : datetime.now().isoformat(timespec="seconds"),
    })
    state["batch_offset"]           = completed_offset + state["per_subject"]
    state["models_done_this_batch"] = []
    return state

# ── Runner ─────────────────────────────────────────────────────────────────────

def run_model(script_name: str, offset: int, n: int) -> bool:
    """Run one model script with the given offset/n via env vars. Returns True on success."""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    env = {**os.environ, "MMLU_OFFSET": str(offset), "MMLU_N": str(n)}

    print(f"\n{'='*60}")
    print(f"  Running : {script_name}")
    print(f"  Offset  : {offset}  (questions {offset+1}–{offset+n//N_SUBJECTS} per subject)")
    print(f"  N       : {n} questions total  (~{n//N_SUBJECTS} per subject)")
    print(f"  Started : {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        env=env,
        cwd=SCRIPT_DIR,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    if result.returncode == 0:
        print(f"  DONE    : {script_name}")
        print(f"  Time    : {elapsed/60:.1f} minutes")
        print(f"  Finished: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")
        return True
    else:
        print(f"  ERROR   : {script_name} exited with code {result.returncode}")
        print(f"{'='*60}\n")
        return False

def print_status(state: dict) -> None:
    print(f"\n{'─'*50}")
    print(f"  batch_state.json")
    print(f"{'─'*50}")
    print(f"  Current batch offset  : {state['batch_offset']}")
    print(f"  Questions per batch   : {state['batch_n']}  (~{state['per_subject']} per subject)")
    print(f"  Models done this batch: {state['models_done_this_batch'] or '(none yet)'}")
    all_models = [m for m, _ in MODEL_SCRIPTS]
    pending = [m for m in all_models if m not in state["models_done_this_batch"]]
    print(f"  Models still pending  : {pending or '(all done — run --next-batch)'}")
    if state.get("history"):
        print(f"\n  Completed batches:")
        for h in state["history"]:
            print(f"    offset={h['offset']:>4}  per_subject={h['per_subject']}  "
                  f"models={h['completed']}  finished={h['finished']}")
    print(f"{'─'*50}\n")

# ── Exhaustion detection ───────────────────────────────────────────────────────

def count_files(model_key: str) -> int:
    """Return the number of existing JSONL files for a model."""
    pattern = os.path.join(SCRIPT_DIR, f"mmlu_en_{model_key}_v*.jsonl")
    files = [f for f in glob.glob(pattern) if not os.path.basename(f).startswith("pilot_")]
    return len(files)


def count_new_records(model_key: str, files_before: int) -> int:
    """Count records only in files created since the files_before snapshot."""
    pattern = os.path.join(SCRIPT_DIR, f"mmlu_en_{model_key}_v*.jsonl")
    files = sorted(f for f in glob.glob(pattern) if not os.path.basename(f).startswith("pilot_"))
    new_files = files[files_before:]
    total = 0
    for f in new_files:
        with open(f) as fh:
            total += sum(1 for line in fh if line.strip())
    return total


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run MMLU batches sequentially")
    parser.add_argument("--only",       type=str, default=None,
                        help="Run only this model for the current batch (gptoss | gemma4 | qwen3.5)")
    parser.add_argument("--once",       action="store_true",
                        help="Run only one batch then stop (default: loop until dataset exhausted)")
    parser.add_argument("--status",     action="store_true",
                        help="Show current state and exit")
    parser.add_argument("--next-batch", action="store_true",
                        help="Force-advance to next batch offset (use if all 3 are done)")
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

        # Which models still need to run in this batch?
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
            print(f"\nAll models already completed for batch offset={offset}.")
            if args.once:
                print_status(state)
                return
            # advance and continue looping
            state = advance_batch(state)
            save_state(state)
            continue

        print(f"\n{'#'*60}")
        print(f"  BATCH {batch_number}  —  offset={offset}  (~{n//N_SUBJECTS} questions/subject)")
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

        # Check if all models are done for this batch
        if all(m in state["models_done_this_batch"] for m in all_models):
            total_written = sum(count_new_records(m, files_before[m]) for m in all_models)

            print(f"\n{'*'*60}")
            print(f"  ALL MODELS DONE — batch offset={offset}")
            print(f"  Questions written this batch: {total_written}")
            print(f"{'*'*60}\n")

            state = advance_batch(state)
            save_state(state)

            if total_written == 0:
                print(f"\n{'='*60}")
                print(f"  DATASET FULLY COVERED — no more questions available.")
                print(f"  All batches complete. Run combine_results.py for final stats.")
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
            # if --only was used, stop here; otherwise this shouldn't happen
            if args.only:
                return
            print(f"Continuing with: {remaining[0]}\n")


if __name__ == "__main__":
    main()
