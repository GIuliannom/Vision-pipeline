"""Smoke test: run pretrained slow_r50 on webcam with Kinetics-400 labels."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import urllib.request

# Download Kinetics-400 class labels (one-time, ~10 KB)
labels_path = Path("configs/kinetics_400_labels.txt")
if not labels_path.exists():
    url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
    import json
    with urllib.request.urlopen(url) as r:
        data = json.load(r)
    # Format: {"label_name (id)": id, ...} — invert and sort by id
    by_id = {v: k.split(" (")[0] for k, v in data.items()}
    with open(labels_path, "w") as f:
        for i in range(len(by_id)):
            f.write(by_id[i].replace('"', "") + "\n")
    print(f"Saved {len(by_id)} Kinetics labels to {labels_path}")

# Load pretrained model directly (no training, no checkpoint)
model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
model.eval()

# Save it in our checkpoint format so the inference backend can load it
ckpt = {
    "model_state": model.state_dict(),
    "config": {
        "model_name": "slow_r50",
        "pretrained": True,
        "num_classes": 400,
        "num_frames": 8,
        "sampling_rate": 8,
        "fps": 30,
        "crop_size": 224,
        "side_size": 256,
    },
    "val_acc": 0.0,
    "epoch": 0,
}
out = Path("runs/video/best.pt")
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(ckpt, out)
print(f"Saved pretrained checkpoint to {out}")
print("Now update configs/inference.yaml -> video_labels_file: configs/kinetics_400_labels.txt")