"""
Unified detector backend.

Wraps either YOLO or RT-DETR (both from Ultralytics) behind a single
interface. Critically, detections now include the FULL per-class
probability vector, not just the top class. This is what the ensemble
needs to combine scores.

Each detection looks like:
    {
      "label": "car",            # top-1 class (for display)
      "conf": 0.50,              # top-1 confidence
      "box": (x1, y1, x2, y2),   # pixel coords in original frame
      "class_probs": np.ndarray  # shape (num_classes,), full distribution
    }

NOTE on `class_probs`: Ultralytics' default output gives you only the
top-class score per box (the model already applied argmax). To get the
full vector we use `embed=False, save=False, verbose=False` and read
from `boxes.cls` / `boxes.conf` — but those only give the top class.
To get true full distributions we'd need to hook the model's raw output
before NMS. For most use cases, building a synthetic distribution from
the top-1 score (giving the rest of the probability mass to "other"
classes) works fine for combining two detectors' confidence on the
top class. We do BOTH: keep the top-1 score from each model, and for
matched detections combine those scores per class.

If a detection from Model A has class "car" with conf 0.5 and a matched
detection from Model B has class "truck" with conf 0.6, we treat each
as a vector that's confident only in its top class and combine.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np


class DetectorBackend:
    """
    Wraps a single Ultralytics model (YOLO or RT-DETR).

    Args:
        weights_path: Path to .pt or .onnx weights.
        kind: "yolo" or "rtdetr". Determines which Ultralytics class to use.
        conf: Confidence threshold for keeping a detection.
        iou:  NMS IoU threshold.
        imgsz: Inference image size.
    """

    def __init__(
        self,
        weights_path: str | Path,
        kind: str = "yolo",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ):
        self.kind = kind
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.weights_path = Path(weights_path)

        if kind == "yolo":
            from ultralytics import YOLO
            self.model = YOLO(str(self.weights_path))
        elif kind == "rtdetr":
            from ultralytics import RTDETR
            self.model = RTDETR(str(self.weights_path))
        else:
            raise ValueError(f"Unknown detector kind: {kind!r}")

        # names: dict[int, str], e.g. {0: 'person', 1: 'bicycle', ...}
        self.names: Dict[int, str] = self.model.names
        self.num_classes = len(self.names)

    def predict(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference; return detections with full class probability vectors."""
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )

        out: List[Dict[str, Any]] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                # Build a "one-hot-with-confidence" class probability vector.
                # The top class gets the model's confidence, the rest get
                # the remaining probability mass spread uniformly. This
                # gives the fusion layer something to work with even
                # though the underlying model doesn't expose its raw
                # pre-NMS softmax.
                probs = np.full(self.num_classes, (1.0 - c) / max(self.num_classes - 1, 1),
                                dtype=np.float32)
                probs[int(k)] = float(c)
                out.append({
                    "label": self.names.get(int(k), str(k)),
                    "conf": float(c),
                    "box": (float(x1), float(y1), float(x2), float(y2)),
                    "class_probs": probs,
                    "class_id": int(k),
                })
        return out


# Backwards-compat alias so existing v2 code keeps working
class YoloDetector(DetectorBackend):
    def __init__(self, weights_path, conf=0.25, iou=0.45, imgsz=640, prefer_onnx=True):
        super().__init__(weights_path, kind="yolo", conf=conf, iou=iou, imgsz=imgsz)
