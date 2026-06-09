"""
Thin Ollama client wrapper.

Talks to a local Ollama server (default: http://localhost:11434) using
the simple /api/chat endpoint. No external SDK required — just HTTP.
"""
from __future__ import annotations

import json
from typing import Optional

import urllib.request
import urllib.error


class OllamaClient:
    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
        system_prompt: Optional[str] = None,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.system_prompt = system_prompt or (
            "You are a concise vision assistant. Given structured scene data "
            "from a webcam (objects detected, actions classified), answer the "
            "user's question directly. Keep responses under 3 sentences. If "
            "something looks unsafe or unusual, say so clearly."
        )

    def chat(self, user_message: str, extra_context: Optional[str] = None) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        if extra_context:
            messages.append({"role": "system", "content": f"Scene data:\n{extra_context}"})
        messages.append({"role": "user", "content": user_message})

        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. "
                f"Is `ollama serve` running? Original error: {e}"
            ) from e

        return data.get("message", {}).get("content", "").strip()

    def health_check(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status == 200
        except Exception:
            return False
