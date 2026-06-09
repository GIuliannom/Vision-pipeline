"""
RT-DETR trainer.

Ultralytics supports RT-DETR with the same API as YOLO — same dataset
format, same training loop, different architecture under the hood
(transformer-based detection instead of anchor-based CNN).

Because the dataset format is identical, you can train RT-DETR on the
exact same data you trained YOLO on, ensuring both detectors share a
class list (required for the ensemble fusion to work cleanly).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import RTDETR


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train(config_path: str) -> Path:
    cfg = load_config(config_path)

    model = RTDETR(cfg["base_model"])
    results = model.train(
        data=cfg["data"],
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg.get("device", 0),
        workers=cfg.get("workers", 8),
        patience=cfg.get("patience", 30),
        optimizer=cfg.get("optimizer", "AdamW"),
        lr0=cfg.get("lr0", 0.0001),
        lrf=cfg.get("lrf", 0.01),
        project=cfg.get("project", "runs/rtdetr"),
        name=cfg.get("name", "train"),
        exist_ok=cfg.get("exist_ok", True),
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n✓ RT-DETR training complete. Best weights: {best}")
    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/rtdetr.yaml")
    args = p.parse_args()
    train(args.config)
