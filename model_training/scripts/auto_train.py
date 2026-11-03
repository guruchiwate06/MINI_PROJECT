"""Automated training pipeline: seamlessly waits for vision extraction and executes full GPU training."""
from __future__ import annotations
import sys
import time
import subprocess
from pathlib import Path

def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def main():
    root = Path(__file__).resolve().parent.parent
    python_exe = sys.executable
    vision_cache_dir = root / "artifacts" / "railvivid_v1" / "vision" / "278cf8c9dd03df3b88ae"
    ready_file = vision_cache_dir / "READY.json"

    log(f"Auto-train orchestrator started. Python: {python_exe}")
    log(f"Monitoring vision cache READY at: {ready_file}")

    while not ready_file.exists():
        # Check if INCOMPLETE or failure happened
        fail_file = vision_cache_dir / "FAILURE.json"
        if fail_file.exists():
            log(f"FATAL: Vision cache failure detected: {fail_file.read_text()}")
            sys.exit(1)
        time.sleep(20)

    log("SUCCESS: Vision cache READY.json confirmed!")
    time.sleep(3)

    # 1. Doctor check
    log("STAGE 1/5: Running track_ot doctor...")
    subprocess.run([python_exe, "-m", "track_ot", "doctor", "--config", "configs/track_ot_plus_plus.json"], cwd=root, check=True)

    # 2. Main TRACK-OT++ training (Pilot)
    log("STAGE 2/3: Launching TRACK-OT++ Model Training on RTX 5070 GPU (--on-training-device)...")
    res = subprocess.run([python_exe, "-m", "track_ot", "train", "--config", "configs/track_ot_plus_plus.json", "--on-training-device"], cwd=root)
    if res.returncode != 0:
        log(f"Training failed with code {res.returncode}")
        sys.exit(res.returncode)
    log("STAGE 2/3: TRACK-OT++ model training finished successfully!")

    # 3. Model Evaluation on Test Partition
    log("STAGE 3/3: Running Test Evaluation on GPU...")
    res_eval = subprocess.run([python_exe, "-m", "track_ot", "evaluate", "--config", "configs/track_ot_plus_plus.json", "--partition", "test", "--device", "cuda"], cwd=root)
    if res_eval.returncode != 0:
        log(f"Evaluation failed with code {res_eval.returncode}")
        sys.exit(res_eval.returncode)

    log("ALL DONE! TRACK-OT++ is fully trained, evaluated, and checkpointed on GPU.")

    # 4. Results ready for user review (Auto-publish disabled per audit protocol)
    log("STAGE 4: Milestone 1 training and evaluation complete. Results preserved in model_training/artifacts/track_ot/.")
    log("Awaiting user review. Automatic remote publishing is disabled.")

if __name__ == "__main__":
    main()

