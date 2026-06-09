"""
Train the fusion head that combines YOLO features + video features.

Prerequisites:
  1. Trained YOLO model:    runs/yolo/train/weights/best.pt
  2. Trained video model:   runs/video/best.pt
  3. Video training data:   data/video/{train,val}/<class>/*.mp4

What this script does:
  - Freezes YOLO and the video backbone.
  - Trains ONLY the fusion head (small MLP) to predict actions from
    (video_features ⊕ pooled_yolo_features).
  - Saves the fusion head to runs/video/fusion_head.pt

This is much faster than retraining either base model and gives you the
'shared learning' effect at inference — the action classifier learns
to use YOLO's understanding of objects as additional context.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.video.train import (  # noqa: E402
    build_dataset, build_model, load_config
)
from shared_features.fused_classifier import FeatureFusionHead  # noqa: E402
from shared_features.yolo_features import YoloFeatureExtractor  # noqa: E402


def get_yolo_feat_dim(extractor: YoloFeatureExtractor, sample_frame) -> int:
    """Run a dummy forward to discover the YOLO feature channel count."""
    _, feats = extractor.predict_with_features(sample_frame)
    if feats is None:
        raise RuntimeError("YOLO hook did not capture features.")
    return feats.shape[1]


def clip_to_middle_frame(clip: torch.Tensor) -> "np.ndarray":
    """
    Take the center frame of a (C, T, H, W) clip and convert to BGR uint8
    for YOLO. clip is normalized (Kinetics stats); we de-normalize.
    """
    import numpy as np
    MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1)
    STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1)
    center_t = clip.shape[1] // 2
    frame = clip[:, center_t]                       # (C, H, W) normalized
    frame = frame * STD + MEAN
    frame = (frame.clamp(0, 1) * 255).byte()
    frame = frame.permute(1, 2, 0).cpu().numpy()    # (H, W, C) RGB
    return frame[..., ::-1].copy()                  # -> BGR


def main(cfg_path: str, yolo_weights: str, video_ckpt: str, out_path: str, epochs: int):
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- YOLO feature extractor (frozen) -----
    yolo_extractor = YoloFeatureExtractor(yolo_weights)
    # Freeze YOLO
    for p in yolo_extractor.model.model.parameters():
        p.requires_grad_(False)

    # ----- Video backbone (frozen) -----
    ckpt = torch.load(video_ckpt, map_location="cpu")
    vcfg = ckpt["config"]
    video_model = build_model(vcfg)
    video_model.load_state_dict(ckpt["model_state"])
    video_model.eval().to(device)
    for p in video_model.parameters():
        p.requires_grad_(False)

    # Capture penultimate features via pre-hook on the final proj
    feat_buffer = {}
    final_proj = video_model.blocks[-1].proj
    video_feat_dim = final_proj.in_features

    def cap(_module, inputs):
        x = inputs[0]
        if x.dim() > 2:
            x = x.mean(dim=tuple(range(2, x.dim())))
        feat_buffer["video"] = x

    final_proj.register_forward_pre_hook(cap)

    # ----- Probe YOLO feature dimensionality -----
    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    train_loader = DataLoader(train_ds, batch_size=1, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    # Use one sample to probe yolo_dim
    sample = next(iter(train_loader))
    sample_frame = clip_to_middle_frame(sample["video"][0])
    yolo_dim = get_yolo_feat_dim(yolo_extractor, sample_frame)
    print(f"YOLO feature channels: {yolo_dim}, video feature dim: {video_feat_dim}")

    # ----- Build trainable fusion head -----
    num_classes = cfg["num_classes"]
    head = FeatureFusionHead(
        video_dim=video_feat_dim,
        yolo_dim=yolo_dim,
        num_classes=num_classes,
    ).to(device)

    optim = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ---- train ----
        head.train()
        tr_total, tr_correct, tr_loss = 0, 0, 0.0
        for batch in train_loader:
            clip = batch["video"].to(device)              # (1, C, T, H, W)
            label = batch["label"].to(device)

            # Center frame -> YOLO -> feature map
            yolo_frame = clip_to_middle_frame(batch["video"][0])
            _, yolo_feats = yolo_extractor.predict_with_features(yolo_frame)
            yolo_feats = yolo_feats.to(device)

            # Video forward (hook captures penultimate)
            with torch.no_grad():
                _ = video_model(clip)
            video_feats = feat_buffer["video"]

            optim.zero_grad()
            logits = head(video_feats, yolo_feats)
            loss = criterion(logits, label)
            loss.backward()
            optim.step()

            tr_loss += loss.item()
            tr_correct += (logits.argmax(1) == label).sum().item()
            tr_total += label.size(0)

        # ---- val ----
        head.eval()
        va_total, va_correct = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                clip = batch["video"].to(device)
                label = batch["label"].to(device)
                yolo_frame = clip_to_middle_frame(batch["video"][0])
                _, yolo_feats = yolo_extractor.predict_with_features(yolo_frame)
                yolo_feats = yolo_feats.to(device)
                _ = video_model(clip)
                video_feats = feat_buffer["video"]
                logits = head(video_feats, yolo_feats)
                va_correct += (logits.argmax(1) == label).sum().item()
                va_total += label.size(0)

        tr_acc = tr_correct / max(tr_total, 1)
        va_acc = va_correct / max(va_total, 1)
        print(f"epoch {epoch:3d}/{epochs} | train loss={tr_loss/max(tr_total,1):.4f} "
              f"acc={tr_acc:.3f} | val acc={va_acc:.3f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(
                {
                    "state": head.state_dict(),
                    "yolo_dim": yolo_dim,
                    "video_dim": video_feat_dim,
                    "num_classes": num_classes,
                    "val_acc": va_acc,
                },
                out_path,
            )
            print(f"  ↳ saved fusion head ({va_acc:.3f}) to {out_path}")

    yolo_extractor.close()
    print(f"\n✓ Fusion head training complete. Best val acc: {best_acc:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/video.yaml")
    p.add_argument("--yolo-weights", default="runs/yolo/train/weights/best.pt")
    p.add_argument("--video-ckpt", default="runs/video/best.pt")
    p.add_argument("--out", default="runs/video/fusion_head.pt")
    p.add_argument("--epochs", type=int, default=20)
    args = p.parse_args()
    main(args.config, args.yolo_weights, args.video_ckpt, args.out, args.epochs)
