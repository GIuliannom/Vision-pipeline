"""
Extract intermediate feature maps from a trained YOLO model's backbone.

This is the 'shared features at inference' approach: YOLO does its normal
detection, AND we tap one of its backbone layers to get a feature map that
the video classifier can use as an extra input signal.

We use PyTorch forward hooks on a specific layer of the Ultralytics model.
Default layer is the deepest backbone stage (index 9 in YOLOv8/11), which
gives a small, semantically rich feature map (~20x20x512 at imgsz=640).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


class YoloFeatureExtractor:
    """
    Wraps an Ultralytics YOLO model and captures features from a chosen layer.

    Usage:
        extractor = YoloFeatureExtractor("runs/yolo/train/weights/best.pt")
        detections, features = extractor.predict_with_features(frame_bgr)
        # features: torch.Tensor of shape (1, C, H, W)
    """

    def __init__(
        self,
        weights_path: str | Path,
        feature_layer_idx: int = 9,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ):
        from ultralytics import YOLO
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.feature_layer_idx = feature_layer_idx

        self.model = YOLO(str(weights_path))
        self.names = self.model.names

        # Buffer to hold the most recent captured feature map
        self._feature_buffer: Optional[torch.Tensor] = None

        # Register a forward hook on the chosen layer of the underlying nn.Module
        # Ultralytics nests the actual nn.Module at .model.model
        try:
            target = self.model.model.model[feature_layer_idx]
        except (AttributeError, IndexError) as e:
            raise RuntimeError(
                f"Could not access YOLO layer {feature_layer_idx}. "
                f"Inspect with: list(yolo.model.model.children())"
            ) from e

        self._hook_handle = target.register_forward_hook(self._capture_hook)

    def _capture_hook(self, module, inputs, output) -> None:
        # YOLO modules sometimes return tuples; take the first tensor
        if isinstance(output, tuple):
            output = output[0]
        self._feature_buffer = output.detach()

    def predict_with_features(
        self, frame_bgr: np.ndarray
    ) -> tuple[list[dict], Optional[torch.Tensor]]:
        """Run YOLO; return detections and the captured backbone feature map."""
        self._feature_buffer = None
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )

        # Convert detections to our normalized format
        dets: list[dict] = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                dets.append({
                    "label": self.names.get(int(k), str(k)),
                    "conf": float(c),
                    "box": (float(x1), float(y1), float(x2), float(y2)),
                })

        return dets, self._feature_buffer

    def close(self) -> None:
        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
