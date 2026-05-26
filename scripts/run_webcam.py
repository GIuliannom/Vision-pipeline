"""Entry point: run real-time webcam inference."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.webcam import run  # noqa: E402

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/inference.yaml")
    p.add_argument(
        "--backend",
        choices=["yolo", "video", "both"],
        default=None,
        help="Override default backend from config.",
    )
    args = p.parse_args()
    run(args.config, args.backend)
