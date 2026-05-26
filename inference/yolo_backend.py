"""
YOLO inference backend.

Supports two runtimes:
- 'native'  — Ultralytics' .pt loaded via the YOLO class (easiest)
- 'onnx'    — onnxruntime, lighter dependency footprint at inference

Both return a normalized list of detections:
    [{"label": str, "conf": float, "box": (x1, y1, x2, y2)}]
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import numpy as np


class YoloDetector:
    def __init__(
        self,
        weights_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        prefer_onnx: bool = True,
    ):
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.weights_path = Path(weights_path)

        # Decide runtime based on file extension
        if self.weights_path.suffix == ".onnx" or prefer_onnx and self.weights_path.with_suffix(".onnx").exists():
            self._init_onnx()
        else:
            self._init_native()

    # ---------- native (.pt) ----------
    def _init_native(self) -> None:
        from ultralytics import YOLO
        self.runtime = "native"
        self.model = YOLO(str(self.weights_path))
        self.names = self.model.names

    def _predict_native(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        out: List[Dict[str, Any]] = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                out.append({
                    "label": self.names.get(int(k), str(k)),
                    "conf": float(c),
                    "box": (float(x1), float(y1), float(x2), float(y2)),
                })
        return out

    # ---------- ONNX ----------
    def _init_onnx(self) -> None:
        import onnxruntime as ort
        self.runtime = "onnx"

        onnx_path = (
            self.weights_path if self.weights_path.suffix == ".onnx"
            else self.weights_path.with_suffix(".onnx")
        )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # Class names are stored as metadata by Ultralytics' exporter
        meta = self.session.get_modelmeta().custom_metadata_map
        names_str = meta.get("names", "")
        if names_str:
            # Format is like "{0: 'person', 1: 'bicycle', ...}"
            import ast
            try:
                self.names = ast.literal_eval(names_str)
            except Exception:
                self.names = {}
        else:
            self.names = {}

    def _preprocess_onnx(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Letterbox resize to imgsz, return tensor + scale + pad for un-projecting boxes."""
        h0, w0 = frame_bgr.shape[:2]
        r = self.imgsz / max(h0, w0)
        new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
        import cv2
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # pad to square
        pad_w = self.imgsz - new_w
        pad_h = self.imgsz - new_h
        top, left = pad_h // 2, pad_w // 2
        padded = cv2.copyMakeBorder(
            resized, top, pad_h - top, left, pad_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        img = padded[..., ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img[None], r, (left, top)

    def _postprocess_onnx(
        self, output: np.ndarray, scale: float, pad: tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Ultralytics ONNX output: (1, 4+nc, N). Apply confidence + class NMS."""
        import cv2
        preds = output[0]            # (4+nc, N)
        preds = preds.T              # (N, 4+nc)
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        confs = class_scores.max(axis=1)

        mask = confs > self.conf
        boxes_xywh = boxes_xywh[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]
        if len(boxes_xywh) == 0:
            return []

        # xywh -> xyxy
        xyxy = np.empty_like(boxes_xywh)
        xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        # NMS via OpenCV (per-class)
        keep_idxs: list[int] = []
        for cid in np.unique(class_ids):
            mask_c = class_ids == cid
            idxs = cv2.dnn.NMSBoxes(
                bboxes=xyxy[mask_c].tolist(),
                scores=confs[mask_c].tolist(),
                score_threshold=self.conf,
                nms_threshold=self.iou,
            )
            if len(idxs) > 0:
                global_idxs = np.where(mask_c)[0]
                # cv2.dnn.NMSBoxes returns shape (K,) in recent versions
                idxs = np.array(idxs).flatten()
                keep_idxs.extend(global_idxs[idxs].tolist())

        # Un-project boxes back to original frame
        pad_l, pad_t = pad
        out: List[Dict[str, Any]] = []
        for i in keep_idxs:
            x1, y1, x2, y2 = xyxy[i]
            x1 = (x1 - pad_l) / scale
            y1 = (y1 - pad_t) / scale
            x2 = (x2 - pad_l) / scale
            y2 = (y2 - pad_t) / scale
            out.append({
                "label": self.names.get(int(class_ids[i]), str(int(class_ids[i]))),
                "conf": float(confs[i]),
                "box": (float(x1), float(y1), float(x2), float(y2)),
            })
        return out

    # ---------- public ----------
    def predict(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if self.runtime == "native":
            return self._predict_native(frame_bgr)
        img, scale, pad = self._preprocess_onnx(frame_bgr)
        output = self.session.run(None, {self.input_name: img})[0]
        return self._postprocess_onnx(output, scale, pad)
