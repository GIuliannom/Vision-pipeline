"""
All-in-one runner.

Starts:
  1. Webcam inference loop (in a background thread)
  2. Alert watcher (in a background thread, asks Ollama on triggers)
  3. Interactive Q&A loop in the main thread (you ↔ Ollama)

The webcam window still appears; close it with 'q' or use --headless.

This is the easiest way to test the full v2 stack.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.webcam import run as run_webcam
from alerts.watcher import AlertWatcher
from llm_client.ollama_client import OllamaClient
from llm_client.interactive import interactive_loop, build_context  # noqa: F401
from shared_features.scene_state import SCENE


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inference-config", default="configs/inference.yaml")
    p.add_argument("--alerts-config", default="configs/alerts.yaml")
    p.add_argument("--backend", choices=["yolo", "video", "both"], default=None)
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--model", default="llama3.2:3b")
    p.add_argument("--headless", action="store_true",
                   help="Don't show the webcam window (still publishes state).")
    p.add_argument("--no-alerts", action="store_true",
                   help="Disable background alert watcher.")
    p.add_argument("--no-llm", action="store_true",
                   help="Disable interactive LLM loop (just run webcam + maybe alerts).")
    args = p.parse_args()

    # ---- start webcam thread ----
    webcam_thread = threading.Thread(
        target=run_webcam,
        kwargs=dict(
            config_path=args.inference_config,
            backend_override=args.backend,
            camera_override=args.camera,
            headless=args.headless,
        ),
        daemon=True,
    )
    webcam_thread.start()
    print("[runner] webcam thread started, waiting for first frame...")

    # Wait briefly for the scene state to populate
    for _ in range(30):
        if SCENE.latest() is not None:
            break
        time.sleep(0.2)
    if SCENE.latest() is None:
        print("[runner] warning: no scene data yet — webcam may have failed to open.")

    # ---- prepare LLM client (used by both alerts and interactive) ----
    llm = None
    if not args.no_llm or not args.no_alerts:
        llm = OllamaClient(model=args.model)
        if not llm.health_check():
            print(f"[runner] ⚠ Ollama not reachable. Run `ollama serve` and "
                  f"`ollama pull {args.model}` first.")
            llm = None

    # ---- start alert watcher ----
    watcher = None
    if not args.no_alerts and llm is not None:
        alerts_cfg = load_yaml(args.alerts_config)
        watcher = AlertWatcher(
            rules=alerts_cfg.get("rules", []),
            ask_llm=lambda prompt: llm.chat(prompt, extra_context=None),
            poll_interval=alerts_cfg.get("poll_interval_seconds", 2.0),
            cooldown_seconds=alerts_cfg.get("cooldown_seconds", 30.0),
        )
        watcher.start()
        print(f"[runner] alert watcher started ({len(alerts_cfg.get('rules', []))} rules).")

    # ---- main: interactive Q&A ----
    if args.no_llm or llm is None:
        print("[runner] LLM disabled — webcam will run until you close its window.")
        try:
            webcam_thread.join()
        except KeyboardInterrupt:
            pass
    else:
        try:
            interactive_loop(model=args.model)
        finally:
            if watcher:
                watcher.stop()
            print("[runner] shutting down.")


if __name__ == "__main__":
    main()
