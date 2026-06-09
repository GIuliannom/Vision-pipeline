"""
Interactive Q&A: ask the LLM about what the webcam is seeing.

This is the simplest integration: instead of going through MCP, it reads
the scene state directly (same Python process) and feeds it to Ollama as
context for each user question.

For a full MCP roundtrip (where the LLM decides when to call tools), see
docs/mcp_setup.md — that requires an MCP-aware client like Claude Desktop
or a custom Ollama+MCP harness, which is more setup than most people need
for testing.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared_features.scene_state import SCENE
from alerts.watcher import ALERTS
from llm_client.ollama_client import OllamaClient


def build_context() -> str:
    """Compose a compact text snapshot of the current scene for the LLM."""
    snap = SCENE.latest()
    if snap is None:
        return "No scene data available — webcam pipeline may not be running."

    parts = [f"Timestamp: {time.strftime('%H:%M:%S', time.localtime(snap.timestamp))}"]
    parts.append(f"Scene summary: {snap.summary}")

    if snap.objects:
        # Group by label
        counts: dict[str, list[float]] = {}
        for o in snap.objects:
            counts.setdefault(o["label"], []).append(o["conf"])
        obj_lines = [
            f"  - {n} x {label} (max conf {max(confs):.2f})"
            for label, confs in counts.items()
            for n in [len(confs)]
        ]
        parts.append("Objects detected:\n" + "\n".join(obj_lines))
    else:
        parts.append("Objects detected: none")

    if snap.action_topk:
        action_lines = "\n".join(
            f"  - {a['label']} ({a['conf']:.2f})" for a in snap.action_topk
        )
        parts.append(f"Action top-{len(snap.action_topk)}:\n{action_lines}")
    else:
        parts.append("Action: none")

    recent = ALERTS.recent(3)
    if recent:
        alert_lines = "\n".join(
            f"  - [{time.strftime('%H:%M:%S', time.localtime(a.timestamp))}] "
            f"{a.rule_name}: {a.trigger_summary}"
            for a in recent
        )
        parts.append(f"Recent alerts:\n{alert_lines}")

    return "\n".join(parts)


def interactive_loop(model: str = "llama3.2:3b") -> None:
    client = OllamaClient(model=model)

    if not client.health_check():
        print("❌ Ollama is not reachable at http://localhost:11434")
        print("   Start it with:  ollama serve")
        print(f"   And pull the model: ollama pull {model}")
        return

    print(f"✓ Connected to Ollama (model: {model})")
    print("  Type a question about what's on screen, or:")
    print("    'state'   - dump raw scene state")
    print("    'alerts'  - show recent alerts")
    print("    'quit'    - exit\n")

    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break
        if q.lower() == "state":
            snap = SCENE.latest()
            print(json.dumps(snap.to_dict() if snap else {}, indent=2))
            continue
        if q.lower() == "alerts":
            for a in ALERTS.recent(10):
                print(f"  [{time.strftime('%H:%M:%S', time.localtime(a.timestamp))}] "
                      f"{a.rule_name}: {a.trigger_summary}\n    → {a.llm_message}")
            continue

        ctx = build_context()
        try:
            answer = client.chat(q, extra_context=ctx)
        except Exception as e:
            print(f"  (error: {e})")
            continue
        print(f"llm> {answer}\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama3.2:3b")
    args = p.parse_args()
    interactive_loop(args.model)
