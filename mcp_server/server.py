"""
MCP server exposing the live vision pipeline state as tools.

Tools exposed:
  - what_is_on_screen():       structured snapshot of latest detections + action
  - describe_scene():           natural-language summary
  - get_recent_alerts(n):       list recent alerts the watcher has fired
  - get_scene_history(seconds): scene snapshots over the last N seconds

Resources exposed:
  - scene://current             current scene state JSON

Usage:
    python -m mcp_server.server
or via the convenience script:
    python scripts/run_mcp.py

Transports: defaults to stdio so an MCP-aware LLM client can spawn it.
Pass --http to run as HTTP for testing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # pip install mcp

from shared_features.scene_state import SCENE
from alerts.watcher import ALERTS


mcp = FastMCP("vision-pipeline")


@mcp.tool()
def what_is_on_screen() -> dict:
    """
    Return the structured contents of the current webcam frame:
    detected objects (with bounding boxes and confidence), the current
    action classification, and a one-line scene summary.
    """
    snap = SCENE.latest()
    if snap is None:
        return {"status": "no_data", "message": "Webcam pipeline is not running or hasn't produced any frames yet."}
    return {"status": "ok", **snap.to_dict()}


@mcp.tool()
def describe_scene() -> str:
    """One-line natural-language description of the current scene."""
    snap = SCENE.latest()
    if snap is None:
        return "No scene data available — is the webcam pipeline running?"
    return snap.summary or "Scene state present but no summary."


@mcp.tool()
def get_recent_alerts(n: int = 10) -> list[dict]:
    """Return the last N alerts the background watcher has fired."""
    return [a.to_dict() for a in ALERTS.recent(n)]


@mcp.tool()
def get_scene_history(seconds: int = 10) -> list[dict]:
    """
    Return scene snapshots from the last N seconds. Useful for the LLM
    to reason about temporal changes ("did anything move recently?").
    """
    import time
    cutoff = time.time() - seconds
    return [s.to_dict() for s in SCENE.history() if s.timestamp >= cutoff]


@mcp.resource("scene://current")
def current_scene_resource() -> str:
    """The current scene state as a JSON resource."""
    snap = SCENE.latest()
    if snap is None:
        return json.dumps({"status": "no_data"})
    return json.dumps(snap.to_dict(), indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="Run as HTTP server (default: stdio).")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.http:
        # FastMCP supports SSE/HTTP transport
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run()  # stdio


if __name__ == "__main__":
    main()
