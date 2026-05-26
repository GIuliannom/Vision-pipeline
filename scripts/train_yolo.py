"""Entry point: train the YOLO model."""
import sys
from pathlib import Path

# Allow `python scripts/train_yolo.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.yolo.train import train  # noqa: E402

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/yolo.yaml")
    args = p.parse_args()
    train(args.config)
