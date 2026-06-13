"""
Master SciDQA Runner — all 3 models, fully automatic.
======================================================
Runs each model batch-by-batch (500 questions), in order:
  1. Gemma-4   (resumes from where it left off if already started)
  2. GPT-OSS   (resumes from where it left off if already started)
  3. Qwen 3.5  (resumes from where it left off if already started)

Each model stops at error and the whole pipeline stops.
Re-run at any time to resume from the last completed batch.

Usage:
  caffeinate python3 run_all_scidqa.py
"""

import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TOTAL_QUESTIONS = 2937
BATCH_SIZE      = 500

# ── Model pipeline definition ──────────────────────────────────────────────────
MODELS = [
    {
        "name"      : "Gemma-4",
        "script"    : os.path.join(SCRIPT_DIR, "scidqa_gemma4.py"),
        "state_file": os.path.join(SCRIPT_DIR, "batch_state_scidqa_gemma4.json"),
    },
    {
        "name"      : "GPT-OSS",
        "script"    : os.path.join(SCRIPT_DIR, "scidqa_gptoss.py"),
        "state_file": os.path.join(SCRIPT_DIR, "batch_state_scidqa_gptoss.json"),
    },
    {
        "name"      : "Qwen 3.5",
        "script"    : os.path.join(SCRIPT_DIR, "scidqa_qwen3.5.py"),
        "state_file": os.path.join(SCRIPT_DIR, "batch_state_scidqa_qwen3.5.json"),
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_state(state_file: str) -> dict:
    if os.path.exists(state_file):
        with open(state_file) as f:
            return json.load(f)
    return {"next_offset": 0, "batches_done": 0}

def save_state(state_file: str, state: dict):
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

def build_batches() -> list[tuple[int, int]]:
    batches = []
    offset = 0
    while offset < TOTAL_QUESTIONS:
        n = min(BATCH_SIZE, TOTAL_QUESTIONS - offset)
        batches.append((offset, n))
        offset += n
    return batches

def run_model(model: dict, batches: list[tuple[int, int]]) -> bool:
    """
    Runs all remaining batches for one model.
    Returns True if all batches completed, False on error.
    """
    name       = model["name"]
    script     = model["script"]
    state_file = model["state_file"]
    total_batches = len(batches)

    state = load_state(state_file)
    next_offset = state["next_offset"]

    if next_offset >= TOTAL_QUESTIONS:
        print(f"\n  {name}: already complete — skipping.")
        return True

    start_idx = next((i for i, (off, _) in enumerate(batches) if off == next_offset), None)
    if start_idx is None:
        print(f"\n[ERROR] {name}: state offset {next_offset} not in batch plan.")
        print(f"        Delete {state_file} to reset.")
        return False

    remaining = total_batches - start_idx
    print(f"\n  Starting {remaining} batch(es) for {name} "
          f"(from offset {next_offset})...")

    for batch_idx in range(start_idx, total_batches):
        offset, n = batches[batch_idx]
        batch_num = batch_idx + 1

        print(f"\n  [{name}] Batch {batch_num}/{total_batches}  "
              f"questions {offset + 1}–{offset + n}")

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, script, "--offset", str(offset), "--n", str(n)],
            check=False,
        )
        elapsed = round(time.time() - t0, 1)

        if result.returncode != 0:
            print(f"\n[FATAL] {name} batch {batch_num} failed "
                  f"(exit code {result.returncode}).")
            print(f"        Pipeline stopped. Re-run to resume from offset {offset}.")
            return False

        next_completed = offset + n
        state["next_offset"]  = next_completed
        state["batches_done"] = batch_idx + 1
        save_state(state_file, state)

        print(f"  [{name}] Batch {batch_num} done in {elapsed}s  "
              f"({next_completed}/{TOTAL_QUESTIONS} questions complete)")

    return True

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    batches = build_batches()

    print("=" * 60)
    print("SciDQA Master Runner — 3 models × 2,937 questions")
    print(f"Batch size : {BATCH_SIZE}  |  Total batches per model: {len(batches)}")
    print(f"Models     : {' → '.join(m['name'] for m in MODELS)}")
    print("=" * 60)

    for model in MODELS:
        print(f"\n{'═' * 60}")
        print(f"  MODEL: {model['name']}")
        print(f"{'═' * 60}")

        success = run_model(model, batches)
        if not success:
            sys.exit(1)

        print(f"\n  ✓ {model['name']} — all {TOTAL_QUESTIONS} questions done.")

    print("\n" + "=" * 60)
    print("ALL 3 MODELS COMPLETE.")
    print(f"  Gemma-4  → scidqa_gemma4_offset*_n*.jsonl")
    print(f"  GPT-OSS  → scidqa_gptoss_v*.jsonl")
    print(f"  Qwen 3.5 → scidqa_qwen3.5_v*.jsonl")
    print("=" * 60)


if __name__ == "__main__":
    main()
