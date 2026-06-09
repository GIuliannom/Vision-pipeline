"""
Shared scene state.

The webcam loop writes the latest fused result here. The MCP server and
the alert watcher both read from here. Uses a simple threading lock; the
state is small (a few hundred bytes) so contention is not a concern.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SceneSnapshot:
    timestamp: float
    objects: list[dict] = field(default_factory=list)   # YOLO detections
    action: dict | None = None                          # top action
    action_topk: list[dict] = field(default_factory=list)
    summary: str = ""
    backend: str = ""
    fps: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "objects": self.objects,
            "action": self.action,
            "action_topk": self.action_topk,
            "summary": self.summary,
            "backend": self.backend,
            "fps": self.fps,
        }


class SceneState:
    """Thread-safe latest snapshot + ring buffer of recent history."""

    def __init__(self, history_size: int = 60):
        self._lock = threading.Lock()
        self._latest: SceneSnapshot | None = None
        self._history: deque[SceneSnapshot] = deque(maxlen=history_size)

    def update(self, snapshot: SceneSnapshot) -> None:
        with self._lock:
            self._latest = snapshot
            self._history.append(snapshot)

    def latest(self) -> SceneSnapshot | None:
        with self._lock:
            return self._latest

    def history(self, last_n: int | None = None) -> list[SceneSnapshot]:
        with self._lock:
            snaps = list(self._history)
        return snaps[-last_n:] if last_n else snaps

    def age_seconds(self) -> float:
        latest = self.latest()
        if latest is None:
            return float("inf")
        return time.time() - latest.timestamp


# Module-level singleton — both the webcam loop and MCP server import this
SCENE = SceneState()
