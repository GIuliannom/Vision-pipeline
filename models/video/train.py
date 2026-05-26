"""
PyTorchVideo action-recognition trainer.

Builds a labeled-video dataset from data/video/{train,val}/<class>/*.mp4,
loads a pretrained backbone (default: slow_r50, Kinetics-400 init), swaps
the classification head to match num_classes, and runs a standard
train/val loop with checkpointing.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)


# ImageNet/Kinetics normalization stats used by SlowFast family
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_transforms(cfg: dict, train: bool) -> Compose:
    """Standard PyTorchVideo transform pipeline."""
    base = [
        UniformTemporalSubsample(cfg["num_frames"]),
        Lambda(lambda x: x / 255.0),
        Normalize(MEAN, STD),
    ]
    if train:
        spatial = [
            RandomShortSideScale(min_size=cfg["side_size"], max_size=cfg["side_size"] + 64),
            RandomCrop(cfg["crop_size"]),
            RandomHorizontalFlip(p=0.5),
        ]
    else:
        spatial = [
            ShortSideScale(size=cfg["side_size"]),
            CenterCrop(cfg["crop_size"]),
        ]
    return ApplyTransformToKey(key="video", transform=Compose(base + spatial))


def build_dataset(cfg: dict, split: str) -> LabeledVideoDataset:
    root = Path(cfg["data_root"]) / split
    if not root.exists():
        raise FileNotFoundError(f"Expected dataset folder at {root}")

    labeled_paths = LabeledVideoPaths.from_path(str(root))
    clip_duration = cfg["num_frames"] * cfg["sampling_rate"] / cfg["fps"]
    sampler = make_clip_sampler(
        "random" if split == "train" else "uniform",
        clip_duration,
    )
    return LabeledVideoDataset(
        labeled_video_paths=labeled_paths,
        clip_sampler=sampler,
        transform=build_transforms(cfg, train=(split == "train")),
        decode_audio=False,
    )


def build_model(cfg: dict) -> nn.Module:
    """Load a PyTorchVideo backbone and swap its head for our num_classes."""
    model = torch.hub.load(
        "facebookresearch/pytorchvideo",
        cfg["model_name"],
        pretrained=cfg.get("pretrained", True),
    )

    # Replace the final classifier. The attribute name varies by model;
    # most PyTorchVideo models expose `.blocks[-1].proj` as the head.
    try:
        in_features = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Linear(in_features, cfg["num_classes"])
    except AttributeError:
        # Some models (mvit) use a different head structure
        if hasattr(model, "head"):
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, cfg["num_classes"])
        else:
            raise RuntimeError(
                f"Don't know how to replace head on {cfg['model_name']}. "
                "Inspect the model and patch build_model()."
            )
    return model


def train_one_epoch(model, loader, optim, criterion, device) -> tuple[float, float]:
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        videos = batch["video"].to(device)        # (B, C, T, H, W)
        labels = batch["label"].to(device)

        optim.zero_grad()
        logits = model(videos)
        loss = criterion(logits, labels)
        loss.backward()
        optim.step()

        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def validate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        videos = batch["video"].to(device)
        labels = batch["label"].to(device)
        logits = model(videos)
        loss = criterion(logits, labels)

        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train(config_path: str) -> Path:
    cfg = load_config(config_path)
    device = torch.device(cfg["device"] if torch.cuda.is_available() or cfg["device"] != "cuda" else "cpu")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"], pin_memory=True,
    )

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg["epochs"])

    best_acc = 0.0
    ckpt_path = Path(cfg["output_dir"]) / cfg["checkpoint_name"]

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optim, criterion, device)
        va_loss, va_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(
            f"epoch {epoch:3d}/{cfg['epochs']} | "
            f"train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"val loss={va_loss:.4f} acc={va_acc:.3f}"
        )
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "val_acc": va_acc,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"  ↳ saved new best ({va_acc:.3f}) to {ckpt_path}")

    print(f"\n✓ Video training complete. Best val acc: {best_acc:.3f}. Weights: {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/video.yaml")
    args = p.parse_args()
    train(args.config)
