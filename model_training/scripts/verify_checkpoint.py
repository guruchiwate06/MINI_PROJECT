import torch
from pathlib import Path
import json

def verify():
    ckpt_path = Path("model_training/artifacts/track_ot/track_ot_pp_fold1_full_s42/best.pt")
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} not found")
        return

    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print("Checkpoint loaded successfully!")
    if isinstance(data, dict):
        print("Keys:", list(data.keys()))
        for k in ["epoch", "best_ap", "validation_ap", "step", "loss"]:
            if k in data:
                print(f"  {k}: {data[k]}")
        if "model_state_dict" in data:
            tensors = data["model_state_dict"]
            print(f"  Total model tensors: {len(tensors)}")
            sample = list(tensors.keys())[:8]
            for s in sample:
                print(f"    {s}: {tensors[s].shape} ({tensors[s].dtype})")
    
    # Also verify CNN checkpoint
    cnn_path = Path("model_training/artifacts/experiments/fold1_cnn_s42/best.pt")
    if cnn_path.exists():
        cnn_data = torch.load(cnn_path, map_location="cpu", weights_only=False)
        print("\nCNN Checkpoint loaded successfully!")
        if isinstance(cnn_data, dict):
            print("CNN Keys:", list(cnn_data.keys()))
            if "model_state_dict" in cnn_data:
                print(f"  CNN tensors: {len(cnn_data['model_state_dict'])}")

if __name__ == "__main__":
    verify()

