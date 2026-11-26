"""Client wrapper for invoking Gemini 3.8 Flash orchestrator via `agy` CLI."""

from __future__ import annotations
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

DEFAULT_AGY_MODEL = "gemini-3.8-flash-high"

MODEL_ALIASES: Dict[str, str] = {
    "gemini-3.8-flash": "gemini-3.8-flash-high",
    "gemini-flash-3.8": "gemini-3.8-flash-high",
    "gemini-3.8-flash-high": "gemini-3.8-flash-high",
    "gemini-3.8-flash-medium": "gemini-3.8-flash-medium",
    "gemini-3.8-flash-low": "gemini-3.8-flash-low",
}


def normalize_agy_model(model_name: str) -> str:
    """Normalize shorthand model names (e.g. gemini-3.8-flash) to exact `agy models` IDs."""
    key = model_name.strip().lower()
    return MODEL_ALIASES.get(key, model_name.strip())


@dataclass
class AgyResponse:
    text: str
    model: str
    status: str
    conversation_id: Optional[str] = None
    duration_seconds: float = 0.0
    usage: Dict[str, Any] = field(default_factory=dict)
    raw_stdout: str = ""


class AgyCliClient:
    """Invokes `agy` CLI in non-interactive JSON print mode as the harness proposer orchestrator."""

    def __init__(
        self,
        agy_executable: str = "agy",
        default_model: str = DEFAULT_AGY_MODEL,
        timeout_sec: int = 120,
    ):
        self.agy_executable = agy_executable
        self.default_model = normalize_agy_model(default_model)
        self.timeout_sec = timeout_sec

    @staticmethod
    def _get_enhanced_env() -> Dict[str, str]:
        env = os.environ.copy()
        user_home = Path.home()
        extra_paths = [
            str(user_home / ".local" / "bin"),
            str(user_home / "bin"),
        ]
        current_path = env.get("PATH", "")
        env["PATH"] = ":".join(extra_paths) + (":" + current_path if current_path else "")
        return env

    @staticmethod
    def parse_json_envelope(raw_stdout: str) -> Dict[str, Any]:
        """Parse the JSON envelope produced by `agy --output-format json`."""
        text = raw_stdout.strip()
        if not text:
            raise RuntimeError("agy CLI returned empty stdout")

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # Fallback: scan lines from bottom up for the JSON object containing response/status
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict) and ("response" in data or "status" in data):
                    return data
            except json.JSONDecodeError:
                continue

        # Fallback: extract largest {...} slice
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        raise RuntimeError(f"Failed to parse JSON envelope from agy stdout: {text[:400]}")

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        timeout_sec: Optional[int] = None,
    ) -> AgyResponse:
        env = self._get_enhanced_env()
        agy_bin = shutil.which(self.agy_executable, path=env.get("PATH"))
        if not agy_bin:
            raise FileNotFoundError(
                f"agy CLI executable '{self.agy_executable}' not found in PATH."
            )

        resolved_model = normalize_agy_model(model or self.default_model)
        t = timeout_sec if timeout_sec is not None else self.timeout_sec

        cmd = [
            agy_bin,
            "--model",
            resolved_model,
            "--output-format",
            "json",
            "--disable-slash-commands",
            f"--print-timeout={t}s",
            f"-p={prompt}",
        ]

        # Run inside an isolated empty temp directory so orchestrator calls cannot mutate repo files
        with tempfile.TemporaryDirectory(prefix="agy_orch_") as isolated_cwd:
            proc = subprocess.run(
                cmd,
                cwd=isolated_cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=t + 30,
            )

        if proc.returncode != 0:
            raise RuntimeError(
                f"agy CLI failed (exit={proc.returncode}): {(proc.stderr or proc.stdout).strip()[:500]}"
            )

        envelope = self.parse_json_envelope(proc.stdout)
        status = str(envelope.get("status", ""))
        response_text = str(envelope.get("response", "") or "")
        if status != "SUCCESS" or not response_text.strip():
            raise RuntimeError(
                f"agy CLI returned non-success status '{status}' or empty response: {envelope}"
            )

        return AgyResponse(
            text=response_text,
            model=resolved_model,
            status=status,
            conversation_id=envelope.get("conversation_id"),
            duration_seconds=float(envelope.get("duration_seconds", 0.0) or 0.0),
            usage=envelope.get("usage", {}) or {},
            raw_stdout=proc.stdout,
        )

    def as_llm_caller(self) -> Callable[[str, str], str]:
        def _call(model_name: str, prompt: str) -> str:
            return self.generate(prompt=prompt, model=model_name).text

        return _call
