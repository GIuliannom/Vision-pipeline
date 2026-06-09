"""
Export trained models to ONNX so they share one inference runtime.

YOLO: Ultralytics has a built-in .export(format='onnx') call.
Video: standard torch.onnx.export with a representative clip tensor.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def export_yolo(yolo_cfg_path: str) -> Path:
    """Use Ultralytics' export to produce best.onnx next to best.pt."""
    from ultralytics import YOLO

    cfg = load_config(yolo_cfg_path)
    weights_dir = Path(cfg.get("project", "runs/yolo")) / cfg.get("name", "train") / "weights"
    pt_path = weights_dir / "best.pt"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"YOLO weights not found at {pt_path}. Train first."
        )

    model = YOLO(str(pt_path))
    onnx_path = model.export(
        format="onnx",
        imgsz=cfg["imgsz"],
        dynamic=True,    # dynamic batch + spatial dims
        simplify=True,
        opset=17,
    )
    print(f"✓ YOLO exported to ONNX: {onnx_path}")
    return Path(onnx_path)


def export_video(video_cfg_path: str) -> Path:
    """Export the trained video classifier with a dummy clip."""
    from models.video.train import build_model

    cfg = load_config(video_cfg_path)
    ckpt_path = Path(cfg["output_dir"]) / cfg["checkpoint_name"]
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Video checkpoint not found at {ckpt_path}. Train first."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Dummy input matches the format the model expects:
    #   (batch, channels, time, height, width)
    dummy = torch.randn(
        1, 3, cfg["num_frames"], cfg["crop_size"], cfg["crop_size"]
    )

    onnx_path = ckpt_path.with_suffix(".onnx")
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["clip"],
        output_names=["logits"],
        dynamic_axes={"clip": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"✓ Video model exported to ONNX: {onnx_path}")
    return onnx_path


def export_rtdetr(rtdetr_cfg_path: str) -> Path:
    """Use Ultralytics' export to produce best.onnx next to best.pt."""
    from ultralytics import RTDETR

    cfg = load_config(rtdetr_cfg_path)
    weights_dir = Path(cfg.get("project", "runs/rtdetr")) / cfg.get("name", "train") / "weights"
    pt_path = weights_dir / "best.pt"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"RT-DETR weights not found at {pt_path}. Train first."
        )

    model = RTDETR(str(pt_path))
    onnx_path = model.export(
        format="onnx",
        imgsz=cfg["imgsz"],
        dynamic=True,
        simplify=True,
        opset=17,
    )
    print(f"✓ RT-DETR exported to ONNX: {onnx_path}")
    return Path(onnx_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--target",
        choices=["yolo", "rtdetr", "video", "all"],
        default="all",
        help="Which model to export.",
    )
    p.add_argument("--yolo-config", default="configs/yolo.yaml")
    p.add_argument("--rtdetr-config", default="configs/rtdetr.yaml")
    p.add_argument("--video-config", default="configs/video.yaml")
    args = p.parse_args()

    if args.target in ("yolo", "all"):
        export_yolo(args.yolo_config)
    if args.target in ("rtdetr", "all"):
        export_rtdetr(args.rtdetr_config)
    if args.target in ("video", "all"):
        export_video(args.video_config)
