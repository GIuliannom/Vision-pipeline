"""
Background alert system.

Polls the SceneState; when configured trigger rules match, fires an LLM
prompt and stores the resulting alert in a ring buffer that the MCP
server exposes.

Trigger rules are declared in configs/alerts.yaml. Supported types:
  - object_present:   fires when a labeled object is detected
  - object_absent:    fires when a labeled object disappears for N seconds
  - action_matches:   fires when an action label is in the top-1 prediction
  - object_count:     fires when count of a label >=, <=, == threshold
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from shared_features.scene_state import SCENE, SceneSnapshot


@dataclass
class Alert:
    timestamp: float
    rule_name: str
    trigger_summary: str
    llm_message: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "rule_name": self.rule_name,
            "trigger_summary": self.trigger_summary,
            "llm_message": self.llm_message,
        }


class AlertBuffer:
    def __init__(self, max_alerts: int = 50):
        self._lock = threading.Lock()
        self._alerts: deque[Alert] = deque(maxlen=max_alerts)

    def add(self, alert: Alert) -> None:
        with self._lock:
            self._alerts.append(alert)

    def recent(self, n: int = 10) -> list[Alert]:
        with self._lock:
            return list(self._alerts)[-n:]


ALERTS = AlertBuffer()


def _check_object_present(snap: SceneSnapshot, label: str, min_conf: float) -> str | None:
    matches = [o for o in snap.objects
               if o["label"] == label and o["conf"] >= min_conf]
    if matches:
        return f"Detected {len(matches)} '{label}' (max conf {max(m['conf'] for m in matches):.2f})"
    return None


def _check_action_matches(snap: SceneSnapshot, label: str, min_conf: float) -> str | None:
    if snap.action and snap.action["label"] == label and snap.action["conf"] >= min_conf:
        return f"Action '{label}' detected (conf {snap.action['conf']:.2f})"
    return None


def _check_object_count(snap: SceneSnapshot, label: str, op: str, value: int) -> str | None:
    count = sum(1 for o in snap.objects if o["label"] == label)
    ops = {">=": count >= value, "<=": count <= value,
           "==": count == value, ">": count > value, "<": count < value}
    if ops.get(op, False):
        return f"Count of '{label}' = {count} ({op} {value})"
    return None


def _check_object_absent(
    label: str, duration_seconds: float, history: list[SceneSnapshot]
) -> str | None:
    if not history:
        return None
    cutoff = time.time() - duration_seconds
    relevant = [s for s in history if s.timestamp >= cutoff]
    if not relevant:
        return None
    # Span must cover the duration
    if relevant[-1].timestamp - relevant[0].timestamp < duration_seconds * 0.8:
        return None
    if all(not any(o["label"] == label for o in s.objects) for s in relevant):
        return f"'{label}' absent for {duration_seconds}s"
    return None


def evaluate_rules(rules: list[dict]) -> list[tuple[str, str]]:
    """Returns list of (rule_name, trigger_summary) for fired rules."""
    snap = SCENE.latest()
    if snap is None:
        return []
    history = SCENE.history()
    fired: list[tuple[str, str]] = []

    for rule in rules:
        rtype = rule.get("type")
        name = rule.get("name", rtype)
        msg: str | None = None
        try:
            if rtype == "object_present":
                msg = _check_object_present(snap, rule["label"], rule.get("min_conf", 0.5))
            elif rtype == "action_matches":
                msg = _check_action_matches(snap, rule["label"], rule.get("min_conf", 0.5))
            elif rtype == "object_count":
                msg = _check_object_count(snap, rule["label"], rule["op"], rule["value"])
            elif rtype == "object_absent":
                msg = _check_object_absent(rule["label"], rule.get("duration_seconds", 30), history)
        except KeyError:
            continue

        if msg:
            fired.append((name, msg))

    return fired


class AlertWatcher(threading.Thread):
    """Background thread polling for triggered rules and asking the LLM."""

    def __init__(
        self,
        rules: list[dict],
        ask_llm: Callable[[str], str],
        poll_interval: float = 2.0,
        cooldown_seconds: float = 30.0,
    ):
        super().__init__(daemon=True)
        self.rules = rules
        self.ask_llm = ask_llm
        self.poll_interval = poll_interval
        self.cooldown_seconds = cooldown_seconds
        self._stop = threading.Event()
        self._last_fire: dict[str, float] = {}

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                fired = evaluate_rules(self.rules)
                now = time.time()
                for rule_name, summary in fired:
                    last = self._last_fire.get(rule_name, 0.0)
                    if now - last < self.cooldown_seconds:
                        continue
                    self._last_fire[rule_name] = now
                    snap = SCENE.latest()
                    prompt = (
                        f"Alert rule '{rule_name}' triggered: {summary}.\n"
                        f"Current scene: {snap.summary if snap else 'unknown'}.\n"
                        "Give a brief 1-2 sentence assessment, and if it's concerning, "
                        "a clear warning. Otherwise just acknowledge."
                    )
                    try:
                        msg = self.ask_llm(prompt)
                    except Exception as e:
                        msg = f"(LLM error: {e})"
                    ALERTS.add(Alert(
                        timestamp=now,
                        rule_name=rule_name,
                        trigger_summary=summary,
                        llm_message=msg,
                    ))
                    print(f"\n[ALERT] {rule_name}: {summary}\n        LLM: {msg}\n")
            except Exception as e:
                print(f"[alert watcher error] {e}")
            self._stop.wait(self.poll_interval)
