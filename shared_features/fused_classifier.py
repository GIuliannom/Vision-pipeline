"""
Action classifier with optional YOLO feature fusion.

This is a small wrapper around the existing PyTorchVideo classifier that
allows feature maps from YOLO's backbone to be concatenated with the
video model's penultimate features before classification.

Design:
- The original video classifier produces a (batch, feat_dim) penultimate
  vector. We extract that with a hook.
- YOLO produces a (1, C, H, W) feature map. We global-average-pool it to
  (1, C) and pass through a small projection layer to (1, feat_dim).
- The two vectors are concatenated and passed through a new head.

This module is built lazily on first use because the feature dimensions
depend on which YOLO model + video model the user trained.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class FeatureFusionHead(nn.Module):
    """Combines video penultimate features + YOLO pooled features → class logits."""

    def __init__(self, video_dim: int, yolo_dim: int, num_classes: int, hidden: int = 512):
        super().__init__()
        self.yolo_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(yolo_dim, hidden),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(video_dim + hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, video_feats: torch.Tensor, yolo_feats: torch.Tensor) -> torch.Tensor:
        # video_feats: (B, video_dim); yolo_feats: (B, yolo_C, H, W)
        y = self.yolo_proj(yolo_feats)             # (B, hidden)
        combined = torch.cat([video_feats, y], dim=1)
        return self.classifier(combined)


class FusedVideoClassifier:
    """
    Inference-only wrapper that fuses YOLO features into video classification.

    The fusion head is trained jointly (see scripts/train_fusion_head.py) but
    here we just load and run it.
    """

    def __init__(
        self,
        video_ckpt_path: str,
        fusion_head_path: Optional[str],
        labels: list[str],
        device: str = "cuda",
    ):
        from models.video.train import build_model
        self.labels = labels
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(video_ckpt_path, map_location="cpu")
        cfg = ckpt["config"]
        self.cfg = cfg

        # Load base video model
        self.video_model = build_model(cfg)
        self.video_model.load_state_dict(ckpt["model_state"])
        self.video_model.eval().to(self.device)

        # Hook to grab the penultimate features (input to the final projection)
        self._video_feat: Optional[torch.Tensor] = None
        final_proj = self.video_model.blocks[-1].proj
        final_proj.register_forward_pre_hook(self._capture_video_feat)
        self.video_feat_dim = final_proj.in_features

        # Lazy: fusion head loaded if path provided
        self.fusion_head: Optional[FeatureFusionHead] = None
        if fusion_head_path:
            fh_ckpt = torch.load(fusion_head_path, map_location="cpu")
            self.fusion_head = FeatureFusionHead(
                video_dim=self.video_feat_dim,
                yolo_dim=fh_ckpt["yolo_dim"],
                num_classes=len(labels),
            )
            self.fusion_head.load_state_dict(fh_ckpt["state"])
            self.fusion_head.eval().to(self.device)

    def _capture_video_feat(self, module, inputs) -> None:
        # `inputs` is a tuple; the first element is the tensor going into the head
        x = inputs[0]
        # Flatten spatial/temporal dims if present so we have (B, C)
        if x.dim() > 2:
            x = x.mean(dim=tuple(range(2, x.dim())))
        self._video_feat = x.detach()

    @torch.no_grad()
    def predict(
        self,
        clip_tensor: torch.Tensor,            # (1, C, T, H, W)
        yolo_features: Optional[torch.Tensor] # (1, yolo_C, H', W') or None
    ) -> list[dict]:
        self._video_feat = None
        clip_tensor = clip_tensor.to(self.device)
        base_logits = self.video_model(clip_tensor)

        if self.fusion_head is not None and yolo_features is not None:
            yolo_features = yolo_features.to(self.device)
            # Resize/match batch dim if needed
            if yolo_features.shape[0] != self._video_feat.shape[0]:
                yolo_features = yolo_features.expand(self._video_feat.shape[0], -1, -1, -1)
            logits = self.fusion_head(self._video_feat, yolo_features)
        else:
            logits = base_logits

        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        top_idx = probs.argsort()[-3:][::-1]
        return [
            {"label": self.labels[i] if i < len(self.labels) else str(i),
             "conf": float(probs[i])}
            for i in top_idx
        ]
