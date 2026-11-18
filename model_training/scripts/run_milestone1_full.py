"""Complete Milestone 1 Pipeline on RTX 5070 GPU.
1. Verifies/extracts vision cache from E:\Rail-VIVID using DINOv2 on GPU.
2. Trains remaining epochs of TRACK-OT++ with validation early stopping.
3. Emits calibrated artifacts and comprehensive progress reporting.
"""
from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Ensure model_training is on path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from railvivid.common import read_json, utc_now
from railvivid.vision import vision_path, extract_vision
from track_ot.config import load_config
from track_ot.runner import train


def main():
    print(f"[{utc_now()}] Starting Milestone 1 Full GPU Pipeline on RTX 5070...")
    config_path = root / "configs" / "track_ot_full_recovered_fp32.json"
    c = load_config(config_path)

    # 1. Vision Cache Verification
    target_cache = vision_path(c["_base"])
    ready_file = target_cache / "READY.json"
    
    if ready_file.exists():
        print(f"[{utc_now()}] Vision cache verified intact at: {target_cache}")
    else:
        print(f"[{utc_now()}] Vision cache not found at {target_cache}. Extracting DINOv2 features from {c['_base']['data']['root']} on GPU...")
        t0 = time.perf_counter()
        extract_vision(c["_base"], device="cuda", allow_download=False)
        elapsed = time.perf_counter() - t0
        print(f"[{utc_now()}] Vision feature extraction complete in {elapsed/60:.2f} minutes.")

    # 2. Start Full GPU Training
    warm_start_run = root / "artifacts" / "track_ot" / "track_ot_pp_fold1_recovery_fp32_s42"
    use_warm_start = (warm_start_run / "best.pt").is_file() and (warm_start_run / "bags.json").is_file()
    print(f"[{utc_now()}] Launching TRACK-OT++ training run: {c['experiment']['name']}")
    if use_warm_start:
        print(f"  Warm-start checkpoint: {warm_start_run / 'best.pt'}")
    else:
        print("  Starting fresh full training run from Epoch 1 on RTX 5070.")
    print(f"  Budget: up to {c['training']['epochs']} epochs (early stopping patience: {c['training']['patience']})")
    print(f"  Device: {c['training']['device']} (FP32 precision, strict FP64 grad clipping)")
    
    result = train(c, on_training_device=True, warm_start=str(warm_start_run) if use_warm_start else None)
    print(f"[{utc_now()}] Training stage finished successfully: {result}")


if __name__ == "__main__":
    main()
