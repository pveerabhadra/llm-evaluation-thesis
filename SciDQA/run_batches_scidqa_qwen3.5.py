"""
Batch runner for SciDQA Qwen 3.5 evaluation.
Runs scidqa_qwen3.5.py in 500-question batches across all 2,937 questions.

Stops immediately if any batch exits with an error.
Resume: re-run this script — it reads batch_state_gemma4.json and continues
        from where it left off.

Usage:
  caffeinate python3 run_batches_scidqa_qwen3.5.py
"""

import json
import os
import subprocess
import sys
import time

# ── Config ─────────────────────────────────────────────────────────────────────
TOTAL_QUESTIONS = 2937
BATCH_SIZE      = 500
SCRIPT          = os.path.join(os.path.dirname(__file__), "scidqa_qwen3.5.py")
STATE_FILE      = os.path.join(os.path.dirname(__file__), "batch_state_scidqa_qwen3.5.json")

# ── State helpers ──────────────────────────────────────────────────────────────
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"next_offset": 0, "batches_done": 0}

def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# ── Batch plan ─────────────────────────────────────────────────────────────────
def build_batches() -> list[tuple[int, int]]:
    """Returns list of (offset, n) pairs."""
    batches = []
    offset = 0
    while offset < TOTAL_QUESTIONS:
        n = min(BATCH_SIZE, TOTAL_QUESTIONS - offset)
        batches.append((offset, n))
        offset += n
    return batches

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    batches = build_batches()
    total_batches = len(batches)
    state = load_state()
    next_offset = state["next_offset"]

    # Find which batch index to start from
    start_idx = next((i for i, (off, _) in enumerate(batches) if off == next_offset), None)

    if start_idx is None:
        if next_offset >= TOTAL_QUESTIONS:
            print("All batches already completed. Nothing to do.")
            print(f"Delete {STATE_FILE} to start fresh.")
            sys.exit(0)
        else:
            print(f"[ERROR] State offset {next_offset} not found in batch plan. "
                  f"Delete {STATE_FILE} to reset.")
            sys.exit(1)

    remaining = total_batches - start_idx
    print("=" * 60)
    print("SciDQA Batch Runner — Qwen 3.5")
    print(f"Total questions : {TOTAL_QUESTIONS}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Total batches   : {total_batches}")
    print(f"Starting at     : batch {start_idx + 1}/{total_batches} (offset {next_offset})")
    print(f"Remaining       : {remaining} batch(es)")
    print("=" * 60)

    for batch_idx in range(start_idx, total_batches):
        offset, n = batches[batch_idx]
        batch_num = batch_idx + 1

        print(f"\n{'─' * 60}")
        print(f"Batch {batch_num}/{total_batches}  —  offset={offset}  n={n}")
        print(f"Questions {offset + 1}–{offset + n} of {TOTAL_QUESTIONS}")
        print(f"{'─' * 60}")

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, SCRIPT, "--offset", str(offset), "--n", str(n)],
            check=False,
        )
        elapsed = round(time.time() - t0, 1)

        if result.returncode != 0:
            print(f"\n[FATAL] Batch {batch_num} failed with exit code {result.returncode}.")
            print(f"        Stopping. Fix the error and re-run to resume from offset {offset}.")
            # Don't advance state — will resume from this batch
            sys.exit(1)

        # Advance state
        next_completed_offset = offset + n
        state["next_offset"]  = next_completed_offset
        state["batches_done"] = batch_idx + 1
        save_state(state)

        print(f"\n  Batch {batch_num} done in {elapsed}s. "
              f"({batch_idx + 1}/{total_batches} batches complete, "
              f"{next_completed_offset}/{TOTAL_QUESTIONS} questions done)")

    print("\n" + "=" * 60)
    print("ALL BATCHES DONE — Qwen 3.5 SciDQA evaluation complete.")
    print(f"Total questions processed: {TOTAL_QUESTIONS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
