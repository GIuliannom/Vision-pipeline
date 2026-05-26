"""
Fusion layer: combine YOLO detections + video action predictions into one
structured result that downstream code (alerting, MCP, logging) can consume.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional


def fuse(
    detections: Optional[List[Dict[str, Any]]],
    actions: Optional[List[Dict[str, Any]]],
    mode: str = "concat",
) -> Dict[str, Any]:
    """
    Args:
        detections: list of {label, conf, box} or None if YOLO disabled.
        actions:    list of {label, conf}      or None if video disabled.
        mode:       'concat' or 'weighted'.

    Returns a structured dict:
        {
          "objects": [{"label": str, "conf": float, "box": [...]}],
          "action":  {"label": str, "conf": float} | None,
          "scene_summary": str,
        }
    """
    result: Dict[str, Any] = {
        "objects": detections or [],
        "action": (actions[0] if actions else None),
        "action_topk": actions or [],
        "scene_summary": "",
    }

    obj_part = ""
    if result["objects"]:
        # Count occurrences per label
        counts: Dict[str, int] = {}
        for d in result["objects"]:
            counts[d["label"]] = counts.get(d["label"], 0) + 1
        obj_part = ", ".join(f"{n} {k}" for k, n in counts.items())

    act_part = ""
    if result["action"]:
        act_part = f"{result['action']['label']} ({result['action']['conf']:.2f})"

    if mode == "weighted" and result["objects"] and result["action"]:
        # Simple gating: only mention the action if its confidence is decent
        if result["action"]["conf"] < 0.4:
            act_part = ""

    if obj_part and act_part:
        result["scene_summary"] = f"Seeing {obj_part}; action: {act_part}"
    elif obj_part:
        result["scene_summary"] = f"Seeing {obj_part}"
    elif act_part:
        result["scene_summary"] = f"Action: {act_part}"
    else:
        result["scene_summary"] = "Nothing detected"

    return result
