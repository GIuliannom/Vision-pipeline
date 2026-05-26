"""
YOLO trainer.

Thin wrapper around Ultralytics' training API that reads our config
format and writes results to a known location for the export step.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train(config_path: str) -> Path:
    cfg = load_config(config_path)

    model = YOLO(cfg["base_model"])
    results = model.train(
        data=cfg["data"],
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg.get("device", 0),
        workers=cfg.get("workers", 8),
        patience=cfg.get("patience", 30),
        optimizer=cfg.get("optimizer", "auto"),
        lr0=cfg.get("lr0", 0.01),
        lrf=cfg.get("lrf", 0.01),
        momentum=cfg.get("momentum", 0.937),
        weight_decay=cfg.get("weight_decay", 0.0005),
        mosaic=cfg.get("mosaic", 1.0),
        mixup=cfg.get("mixup", 0.0),
        hsv_h=cfg.get("hsv_h", 0.015),
        hsv_s=cfg.get("hsv_s", 0.7),
        hsv_v=cfg.get("hsv_v", 0.4),
        project=cfg.get("project", "runs/yolo"),
        name=cfg.get("name", "train"),
        exist_ok=cfg.get("exist_ok", False),
    )

    # Ultralytics writes best.pt to <project>/<name>/weights/best.pt
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n✓ YOLO training complete. Best weights: {best}")
    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/yolo.yaml")
    args = p.parse_args()
    train(args.config)
