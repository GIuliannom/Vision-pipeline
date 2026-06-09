"""Drawing overlays for the webcam window."""
from __future__ import annotations

from typing import List, Dict, Any

import cv2
import numpy as np


# Deterministic color per label (so 'person' is always the same color)
def _color_for_label(label: str) -> tuple[int, int, int]:
    h = hash(label) & 0xFFFFFF
    return (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)


def draw_detections(frame: np.ndarray, dets: List[Dict[str, Any]]) -> None:
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        color = _color_for_label(d["label"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Optional source tag from the ensemble: "both" / "a_only" / "b_only"
        src = d.get("source")
        tag = ""
        if src == "both":
            tag = " ✓"          # both detectors agreed
        elif src == "a_only":
            tag = " [A]"
        elif src == "b_only":
            tag = " [B]"
        text = f"{d['label']} {d['conf']:.2f}{tag}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
        )


def draw_actions(frame: np.ndarray, actions: List[Dict[str, Any]]) -> None:
    if not actions:
        return
    x, y = 12, 30
    cv2.putText(frame, "Action (top-k):", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    for i, a in enumerate(actions):
        cv2.putText(
            frame, f"  {a['label']}: {a['conf']:.2f}",
            (x, y + 28 * (i + 1)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )


def draw_hud(
    frame: np.ndarray,
    backend: str,
    fps: float,
    summary: str = "",
    show_help: bool = True,
) -> None:
    h, w = frame.shape[:2]
    bar_h = 60 if show_help else 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(
        frame, f"backend: {backend} | {fps:.1f} fps",
        (12, h - bar_h + 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )
    if show_help:
        cv2.putText(
            frame, "1:yolo  2:rtdetr  3:ensemble  4:video  5:all  q:quit",
            (12, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2,
        )
    if summary:
        cv2.putText(
            frame, summary[:120], (12, h - bar_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
        )
