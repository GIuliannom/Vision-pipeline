"""
PyTorchVideo (action recognition) inference backend.

Maintains a rolling buffer of recent webcam frames; when full, runs the
classifier and returns the top-k action predictions.

Supports both native (.pt) and ONNX runtimes.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import List, Dict, Any

import numpy as np


MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
STD = np.array([0.225, 0.225, 0.225], dtype=np.float32)


def _short_side_scale(frame_bgr: np.ndarray, side: int) -> np.ndarray:
    import cv2
    h, w = frame_bgr.shape[:2]
    if h < w:
        new_h, new_w = side, int(round(w * side / h))
    else:
        new_h, new_w = int(round(h * side / w)), side
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _center_crop(frame: np.ndarray, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return frame[top:top + size, left:left + size]


def _preprocess_frame(frame_bgr: np.ndarray, side_size: int, crop_size: int) -> np.ndarray:
    """BGR uint8 frame -> (3, H, W) float32 normalized."""
    f = _short_side_scale(frame_bgr, side_size)
    f = _center_crop(f, crop_size)
    f = f[..., ::-1].astype(np.float32) / 255.0  # BGR->RGB, [0,1]
    f = (f - MEAN) / STD
    return f.transpose(2, 0, 1)                  # (3, H, W)


class VideoClassifier:
    def __init__(
        self,
        weights_path: str | Path,
        labels: List[str],
        clip_len: int = 8,
        sampling_rate: int = 8,
        side_size: int = 256,
        crop_size: int = 224,
        topk: int = 3,
        prefer_onnx: bool = True,
        device: str = "cuda",
    ):
        self.labels = labels
        self.clip_len = clip_len
        self.sampling_rate = sampling_rate
        self.side_size = side_size
        self.crop_size = crop_size
        self.topk = topk
        self.weights_path = Path(weights_path)

        # Rolling buffer holds clip_len * sampling_rate raw frames; we
        # subsample to clip_len at inference time.
        self.buffer: deque[np.ndarray] = deque(maxlen=clip_len * sampling_rate)

        onnx_candidate = self.weights_path.with_suffix(".onnx")
        if (self.weights_path.suffix == ".onnx"
                or (prefer_onnx and onnx_candidate.exists())):
            self._init_onnx(self.weights_path if self.weights_path.suffix == ".onnx" else onnx_candidate)
        else:
            self._init_native(device)

    # ---------- native ----------
    def _init_native(self, device: str) -> None:
        import torch
        from models.video.train import build_model
        self.runtime = "native"
        self.torch = torch

        ckpt = torch.load(self.weights_path, map_location="cpu")
        cfg = ckpt["config"]
        self.model = build_model(cfg)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # ---------- ONNX ----------
    def _init_onnx(self, path: Path) -> None:
        import onnxruntime as ort
        self.runtime = "onnx"
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    # ---------- public ----------
    def update(self, frame_bgr: np.ndarray) -> None:
        """Push a new frame into the rolling buffer."""
        self.buffer.append(frame_bgr)

    def ready(self) -> bool:
        return len(self.buffer) == self.buffer.maxlen

    def predict(self) -> List[Dict[str, Any]]:
        """Run inference on the current buffer; returns [] if not ready yet."""
        if not self.ready():
            return []

        # Uniformly subsample clip_len frames from the buffer
        frames_full = list(self.buffer)
        idxs = np.linspace(0, len(frames_full) - 1, self.clip_len).astype(int)
        clip = [
            _preprocess_frame(frames_full[i], self.side_size, self.crop_size)
            for i in idxs
        ]
        # (T, C, H, W) -> (C, T, H, W) -> add batch
        clip_arr = np.stack(clip, axis=1)[None]   # (1, C, T, H, W)

        if self.runtime == "native":
            tensor = self.torch.from_numpy(clip_arr).to(self.device)
            with self.torch.no_grad():
                logits = self.model(tensor).cpu().numpy()[0]
        else:
            logits = self.session.run(None, {self.input_name: clip_arr.astype(np.float32)})[0][0]

        # Softmax
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        top_idx = probs.argsort()[-self.topk:][::-1]
        return [
            {
                "label": self.labels[i] if i < len(self.labels) else str(i),
                "conf": float(probs[i]),
            }
            for i in top_idx
        ]
