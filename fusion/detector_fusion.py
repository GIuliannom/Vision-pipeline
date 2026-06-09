"""
Detector ensemble fusion.

Combines detections from two (or more) detectors into a single, more
confident set of detections. Process:

  1. Run each detector independently on the same frame.
  2. Match boxes across detectors using IoU > threshold.
  3. For matched boxes, combine per-class probability vectors using
     one of: average, weighted average, multiplication, max.
  4. For unmatched boxes, keep them but optionally apply a confidence
     penalty (since only one detector saw them).
  5. Return a unified list of detections with combined scores.

The result has the same shape as a single detector's output, so the
rest of the pipeline (drawing, MCP, alerts) doesn't change.
"""
from __future__ import annotations

from typing import List, Dict, Any

import numpy as np


def _iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_boxes(
    dets_a: List[Dict[str, Any]],
    dets_b: List[Dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Greedy IoU matching between two lists of detections.

    Returns:
        matches:    list of (i, j) pairs where dets_a[i] matched dets_b[j]
        unmatched_a: indices in dets_a with no match in b
        unmatched_b: indices in dets_b with no match in a
    """
    if not dets_a or not dets_b:
        return [], list(range(len(dets_a))), list(range(len(dets_b)))

    # Build IoU matrix
    iou_mat = np.zeros((len(dets_a), len(dets_b)), dtype=np.float32)
    for i, da in enumerate(dets_a):
        for j, db in enumerate(dets_b):
            iou_mat[i, j] = _iou(da["box"], db["box"])

    # Greedy: repeatedly pick the highest-IoU pair until none remain above threshold
    matches: list[tuple[int, int]] = []
    used_a, used_b = set(), set()
    while True:
        # Mask out used rows/cols
        masked = iou_mat.copy()
        for i in used_a:
            masked[i, :] = -1
        for j in used_b:
            masked[:, j] = -1
        if masked.max() < iou_threshold:
            break
        i, j = np.unravel_index(masked.argmax(), masked.shape)
        matches.append((int(i), int(j)))
        used_a.add(int(i))
        used_b.add(int(j))

    unmatched_a = [i for i in range(len(dets_a)) if i not in used_a]
    unmatched_b = [j for j in range(len(dets_b)) if j not in used_b]
    return matches, unmatched_a, unmatched_b


def _combine_probs(
    p_a: np.ndarray,
    p_b: np.ndarray,
    method: str,
    weight_a: float,
    weight_b: float,
) -> np.ndarray:
    if method == "average":
        out = (p_a + p_b) / 2.0
    elif method == "weighted":
        total = weight_a + weight_b
        out = (weight_a * p_a + weight_b * p_b) / total
    elif method == "multiply":
        out = p_a * p_b
        s = out.sum()
        if s > 0:
            out = out / s     # renormalize
    elif method == "max":
        out = np.maximum(p_a, p_b)
    else:
        raise ValueError(f"Unknown fusion method: {method!r}")
    return out


def _average_box(box_a: tuple, box_b: tuple, w_a: float, w_b: float) -> tuple:
    total = w_a + w_b
    return tuple(
        (w_a * a + w_b * b) / total for a, b in zip(box_a, box_b)
    )


def fuse_detectors(
    dets_a: List[Dict[str, Any]],
    dets_b: List[Dict[str, Any]],
    names: Dict[int, str],
    iou_threshold: float = 0.5,
    method: str = "average",
    weight_a: float = 1.0,
    weight_b: float = 1.0,
    unmatched_penalty: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Combine two lists of detections (both indexing the same class list).

    Args:
        dets_a, dets_b: detections from each detector, must have
            'box', 'class_probs', 'class_id', 'label', 'conf'.
        names: shared class id -> name map.
        iou_threshold: minimum IoU to consider two boxes the same object.
        method: 'average' | 'weighted' | 'multiply' | 'max'
        weight_a, weight_b: only used when method='weighted'.
        unmatched_penalty: multiply the conf of detections seen by only
            one detector by this factor (e.g. 0.7 = mild distrust).

    Returns:
        Unified list of detections with combined class_probs and recomputed
        label/conf as the argmax of the combined distribution.
    """
    matches, only_a, only_b = _match_boxes(dets_a, dets_b, iou_threshold)
    fused: List[Dict[str, Any]] = []

    # Matched boxes
    for i, j in matches:
        da, db = dets_a[i], dets_b[j]
        combined_probs = _combine_probs(
            da["class_probs"], db["class_probs"], method, weight_a, weight_b
        )
        top_id = int(combined_probs.argmax())
        top_conf = float(combined_probs[top_id])
        # When two independent detectors agree on a box, we GAIN confidence
        # — not lose it. The averaged top-class conf is the right number.
        # (No agreement bonus by default; method='multiply' already gives it.)
        fused.append({
            "label": names.get(top_id, str(top_id)),
            "conf": top_conf,
            "box": _average_box(da["box"], db["box"], weight_a, weight_b),
            "class_probs": combined_probs,
            "class_id": top_id,
            "source": "both",
            "source_labels": {
                "a": {"label": da["label"], "conf": da["conf"]},
                "b": {"label": db["label"], "conf": db["conf"]},
            },
        })

    # Detections seen only by A
    for i in only_a:
        d = dict(dets_a[i])
        d["conf"] = float(d["conf"]) * unmatched_penalty
        d["class_probs"] = d["class_probs"] * unmatched_penalty
        d["source"] = "a_only"
        fused.append(d)

    # Detections seen only by B
    for j in only_b:
        d = dict(dets_b[j])
        d["conf"] = float(d["conf"]) * unmatched_penalty
        d["class_probs"] = d["class_probs"] * unmatched_penalty
        d["source"] = "b_only"
        fused.append(d)

    return fused
