"""Entry point: train the PyTorchVideo action recognizer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.video.train import train  # noqa: E402

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/video.yaml")
    args = p.parse_args()
    train(args.config)
