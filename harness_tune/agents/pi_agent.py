"""Pi Coding Agent adapter."""

from __future__ import annotations
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from harness_tune.agents.base import AbstractCodingAgent
from harness_tune.core.manifest import HarnessManifest, ModelProfile
from harness_tune.core.trace import ExecutionTrace, StepTrace, ToolCallRecord

VALID_PI_TOOLS = {"read", "write", "edit", "bash", "grep", "find", "ls"}
SAFE_COMPAT_KEYS = {
    "supportsStore",
    "supportsDeveloperRole",
    "supportsReasoningEffort",
    "supportsUsageInStreaming",
    "supportsFinishReason",
    "maxTokensField",
    "thinkingFormat",
    "reasoningEffortMap",
}


class PiNdjsonTraceParser:
    """Incremental parser for `pi --mode json` NDJSON event streams."""

    def __init__(
        self,
        max_steps: int = 25,
        max_consecutive_tool_errors: int = 2,
        early_stop_on_repeated_failure: bool = True,
    ):
        self.max_steps = max_steps
        self.max_consecutive_tool_errors = max_consecutive_tool_errors
        self.early_stop_on_repeated_failure = early_stop_on_repeated_failure

        self.steps: List[StepTrace] = []
        self._current_step: Optional[StepTrace] = None
        self._pending_tools: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        self.total_tokens: int = 0
        self.final_assistant_text: str = ""
        self.api_error: Optional[str] = None
        self.early_stop_reason: Optional[str] = None
        self._error_streak: int = 0
        self._last_error_sig: Optional[Tuple[str, str]] = None

    def _ensure_step(self) -> StepTrace:
        if self._current_step is None:
            self._current_step = StepTrace(step_number=len(self.steps) + 1)
        return self._current_step

    @staticmethod
    def _extract_tool_output(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                if parts:
                    return "\n".join(parts)
            return json.dumps(result, ensure_ascii=False)
        return str(result or "")

    def process_line(self, raw_line: str) -> Optional[str]:
        """Process one NDJSON line. Returns an early-stop reason if execution should abort."""
        line = raw_line.strip()
        if not line or not line.startswith("{"):
            return None

        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            return None

        if not isinstance(ev, dict):
            return None

        ev_type = ev.get("type")

        if ev_type == "turn_start":
            if self._current_step is not None:
                self.steps.append(self._current_step)
            self._current_step = StepTrace(step_number=len(self.steps) + 1)

        elif ev_type == "tool_execution_start":
            tool_id = str(ev.get("toolCallId") or "")
            tool_name = str(ev.get("toolName") or "unknown")
            args = ev.get("args") if isinstance(ev.get("args"), dict) else {}
            self._pending_tools[tool_id] = (tool_name, args)

        elif ev_type == "tool_execution_end":
            step = self._ensure_step()
            tool_id = str(ev.get("toolCallId") or "")
            tool_name = str(ev.get("toolName") or "unknown")
            args: Dict[str, Any] = {}
            if tool_id in self._pending_tools:
                pending_name, args = self._pending_tools.pop(tool_id)
                tool_name = tool_name or pending_name

            is_error = bool(ev.get("isError", False))
            output_text = self._extract_tool_output(ev.get("result"))
            duration_ms = float(ev.get("durationMs", 0.0) or 0.0)
            status = "error" if is_error else "success"

            step.tool_calls.append(
                ToolCallRecord(
                    tool_name=tool_name,
                    arguments=args,
                    output=output_text,
                    status=status,
                    duration_ms=duration_ms,
                )
            )
            step.duration_ms += duration_ms

            if is_error:
                sig = (tool_name, output_text[:100])
                if sig == self._last_error_sig:
                    self._error_streak += 1
                else:
                    self._error_streak = 1
                    self._last_error_sig = sig

                if (
                    self.early_stop_on_repeated_failure
                    and self.max_consecutive_tool_errors > 0
                    and self._error_streak >= self.max_consecutive_tool_errors
                ):
                    self.early_stop_reason = (
                        f"Early stopped: tool '{tool_name}' failed {self._error_streak} "
                        "times consecutively with the same error"
                    )
                    return self.early_stop_reason
            else:
                self._error_streak = 0
                self._last_error_sig = None

        elif ev_type == "message_end":
            msg = ev.get("message")
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                step = self._ensure_step()
                usage = msg.get("usage")
                if isinstance(usage, dict):
                    tok = usage.get("totalTokens")
                    if isinstance(tok, (int, float)) and tok > 0:
                        step.tokens_used += int(tok)
                        self.total_tokens += int(tok)
                    else:
                        in_t = int(usage.get("input", 0) or 0)
                        out_t = int(usage.get("output", 0) or 0)
                        step.tokens_used += in_t + out_t
                        self.total_tokens += in_t + out_t

                thinking_parts: List[str] = []
                text_parts: List[str] = []
                for item in msg.get("content", []):
                    if not isinstance(item, dict):
                        continue
                    ctype = item.get("type")
                    if ctype == "thinking" and isinstance(item.get("thinking"), str):
                        thinking_parts.append(item["thinking"])
                    elif ctype == "text" and isinstance(item.get("text"), str):
                        text_parts.append(item["text"])

                if text_parts:
                    joined_text = "\n".join(text_parts).strip()
                    self.final_assistant_text = joined_text
                    step.thought = joined_text
                elif thinking_parts:
                    step.thought = "\n".join(thinking_parts).strip()

                stop_reason = msg.get("stopReason")
                if stop_reason == "error" or msg.get("errorMessage"):
                    self.api_error = str(
                        msg.get("errorMessage") or f"Assistant stopped with reason '{stop_reason}'"
                    )

        elif ev_type == "turn_end":
            if self._current_step is not None:
                self.steps.append(self._current_step)
                self._current_step = None
            if self.max_steps > 0 and len(self.steps) >= self.max_steps:
                self.early_stop_reason = f"Early stopped: reached max_steps ({self.max_steps})"
                return self.early_stop_reason

        elif ev_type == "agent_end":
            if self._current_step is not None:
                self.steps.append(self._current_step)
                self._current_step = None

        return None

    def finalize(self) -> List[StepTrace]:
        if self._current_step is not None:
            self.steps.append(self._current_step)
            self._current_step = None
        return self.steps


class PiCodingAgent(AbstractCodingAgent):
    """Adapter for Pi Coding Agent (@earendil-works/pi-coding-agent)."""

    def __init__(
        self,
        pi_executable: Optional[str] = "pi",
        mode: str = "auto",  # "auto", "real", "simulated"
        ssh_host: Optional[str] = None,
        provider: Optional[str] = None,
        thinking_level: Optional[str] = None,
        use_user_profile: bool = False,
        user_skills_dir: Optional[str | Path] = None,
    ):
        self.pi_executable = pi_executable
        self.mode = mode
        self.ssh_host = ssh_host
        self.provider = provider
        self.thinking_level = thinking_level
        self.use_user_profile = use_user_profile
        self.user_skills_dir = Path(user_skills_dir).expanduser().resolve() if user_skills_dir else None

    @staticmethod
    def _get_enhanced_env() -> Dict[str, str]:
        env = os.environ.copy()
        user_home = Path.home()
        extra_paths = [
            str(user_home / ".local" / "bin"),
            str(user_home / "bin"),
        ]
        nvm_node_dir = user_home / ".nvm" / "versions" / "node"
        if nvm_node_dir.exists():
            for version_dir in sorted(nvm_node_dir.iterdir(), reverse=True):
                bin_dir = version_dir / "bin"
                if bin_dir.exists():
                    extra_paths.append(str(bin_dir))
                    break
        current_path = env.get("PATH", "")
        env["PATH"] = ":".join(extra_paths) + (":" + current_path if current_path else "")
        return env

    @staticmethod
    def resolve_host_pi_bundle() -> Tuple[Path, Path]:
        """Locate host Node binary and managed Pi Coding Agent release directory."""
        user_home = Path.home()
        node_candidates: List[Path] = [
            user_home / ".local" / "share" / "pi-node" / "current" / "bin" / "node",
        ]
        nvm_dir = user_home / ".nvm" / "versions" / "node"
        if nvm_dir.exists():
            for vdir in sorted(nvm_dir.iterdir(), reverse=True):
                node_candidates.append(vdir / "bin" / "node")

        which_node = shutil.which("node")
        if which_node:
            node_candidates.append(Path(which_node))

        node_bin: Optional[Path] = None
        for cand in node_candidates:
            if cand.exists() and cand.is_file():
                node_bin = cand.resolve()
                break
        if node_bin is None:
            raise FileNotFoundError("Host Node.js binary not found for Pi container bundle.")

        install_root = user_home / ".pi" / "agent" / "install"
        current_file = install_root / "current-version"
        release_dir: Optional[Path] = None
        if current_file.exists():
            version = current_file.read_text(encoding="utf-8").strip()
            cand_rel = install_root / "releases" / version
            if (cand_rel / "node_modules" / "@earendil-works" / "pi-coding-agent" / "dist" / "bundle" / "cli.js").exists():
                release_dir = cand_rel.resolve()

        if release_dir is None and (install_root / "releases").exists():
            for rdir in sorted((install_root / "releases").iterdir(), reverse=True):
                if (rdir / "node_modules" / "@earendil-works" / "pi-coding-agent" / "dist" / "bundle" / "cli.js").exists():
                    release_dir = rdir.resolve()
                    break

        if release_dir is None:
            raise FileNotFoundError(
                f"Managed Pi release directory not found under {install_root / 'releases'}."
            )
        return node_bin, release_dir

    @staticmethod
    def resolve_user_profile_paths(
        user_skills_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Resolve non-secret personal Pi profile paths (packages, git, npm, skills)."""
        user_home = Path.home()
        pi_agent_dir = user_home / ".pi" / "agent"

        git_dir = pi_agent_dir / "git"
        npm_dir = pi_agent_dir / "npm"
        skills_cand = (
            Path(user_skills_dir).expanduser().resolve()
            if user_skills_dir
            else (user_home / ".agents" / "skills")
        )

        packages: List[str] = []
        settings_path = pi_agent_dir / "settings.json"
        if settings_path.exists():
            try:
                raw_settings = json.loads(settings_path.read_text(encoding="utf-8"))
                raw_pkgs = raw_settings.get("packages", [])
                if isinstance(raw_pkgs, list):
                    packages = [str(p) for p in raw_pkgs if isinstance(p, str) and p.strip()]
            except Exception:
                packages = []

        return {
            "git_dir": git_dir.resolve() if git_dir.exists() else None,
            "npm_dir": npm_dir.resolve() if npm_dir.exists() else None,
            "skills_dir": skills_cand.resolve() if skills_cand.exists() else None,
            "packages": packages,
        }

    @staticmethod
    def resolve_provider_compat(provider_name: str) -> Optional[Dict[str, Any]]:
        """Read non-secret provider compat flags (e.g. thinkingFormat) from ~/.pi/agent/models.json."""
        if not provider_name or provider_name == "unknown":
            return None
        models_path = Path.home() / ".pi" / "agent" / "models.json"
        if not models_path.exists():
            return None
        try:
            data = json.loads(models_path.read_text(encoding="utf-8"))
            prov_cfg = data.get("providers", {}).get(provider_name, {})
            compat = prov_cfg.get("compat")
            if not isinstance(compat, dict):
                return None
            filtered = {k: v for k, v in compat.items() if k in SAFE_COMPAT_KEYS}
            return filtered or None
        except Exception:
            return None

    @staticmethod
    def write_isolated_container_models_json(
        target_dir: Path,
        base_url: str,
        model_profile: ModelProfile,
        packages: Optional[List[str]] = None,
        compat: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write a secret-free models.json and settings.json into target_dir for isolated container runs."""
        target_dir.mkdir(parents=True, exist_ok=True)
        provider_name = model_profile.provider if model_profile.provider not in ("", "unknown") else "local"
        provider_entry: Dict[str, Any] = {
            "baseUrl": base_url,
            "api": "openai-completions",
            "apiKey": "local-unused",
            "models": [
                {
                    "id": model_profile.model_name,
                    "name": model_profile.model_name,
                    "contextWindow": model_profile.context_window,
                    "maxTokens": 16384,
                    "reasoning": True,
                    "input": ["text"],
                }
            ],
        }
        if compat:
            provider_entry["compat"] = dict(compat)

        models_payload = {"providers": {provider_name: provider_entry}}
        (target_dir / "models.json").write_text(
            json.dumps(models_payload, indent=2), encoding="utf-8"
        )
        settings_payload: Dict[str, Any] = {
            "defaultProvider": provider_name,
            "defaultModel": model_profile.model_name,
            "defaultThinkingLevel": model_profile.thinking_level or "medium",
        }
        if packages:
            settings_payload["packages"] = list(packages)
        (target_dir / "settings.json").write_text(
            json.dumps(settings_payload, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _load_yaml_surface(harness: HarnessManifest, surface_key: str) -> Dict[str, Any]:
        rel = harness.surfaces.get(surface_key)
        if not rel or not harness.base_dir:
            return {}
        rel_path = rel.get("path") if isinstance(rel, dict) else str(rel)
        target = harness.base_dir / rel_path
        if not target.exists():
            return {}
        try:
            loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _load_system_prompt(harness: HarnessManifest) -> str:
        rel = harness.surfaces.get("system_prompt")
        if not rel or not harness.base_dir:
            return ""
        rel_path = rel.get("path") if isinstance(rel, dict) else str(rel)
        p_path = harness.base_dir / rel_path
        if p_path.exists():
            return p_path.read_text(encoding="utf-8")
        return ""

    @staticmethod
    def _resolve_skill_rel_paths(harness: HarnessManifest) -> List[str]:
        if not harness.base_dir:
            return []
        skills_val = harness.surfaces.get("skills")
        entries: List[str] = []
        if isinstance(skills_val, str):
            entries.append(skills_val)
        elif isinstance(skills_val, list):
            entries.extend(str(item) for item in skills_val)
        elif isinstance(skills_val, dict):
            for item in skills_val.get("active_skills", []):
                entries.append(str(item))

        valid_rels: List[str] = []
        for entry in entries:
            p = (harness.base_dir / entry).resolve()
            if p.exists() and p.is_file():
                valid_rels.append(entry)
        return valid_rels

    @staticmethod
    def _resolve_skill_paths(harness: HarnessManifest) -> List[Path]:
        if not harness.base_dir:
            return []
        return [
            (harness.base_dir / rel).resolve()
            for rel in PiCodingAgent._resolve_skill_rel_paths(harness)
        ]

    def _build_pi_flags(
        self,
        instruction: str,
        harness: HarnessManifest,
        system_prompt_content: str,
        runtime_config: Dict[str, Any],
        tool_config: Dict[str, Any],
        skill_args: List[str],
        use_user_profile: bool = False,
        skill_dirs: Optional[List[str]] = None,
    ) -> List[str]:
        flags: List[str] = [
            "-p",
            "--mode",
            "json",
            "--no-session",
            "--offline",
            "--no-mcp",
            "--no-context-files",
            "--no-prompt-templates",
            "--no-themes",
        ]
        if not use_user_profile:
            flags.extend(["--no-extensions", "--no-skills"])

        flags.extend(["--model", harness.model_profile.model_name])

        effective_provider = self.provider or harness.model_profile.provider
        if effective_provider and effective_provider not in ("unknown", ""):
            flags.extend(["--provider", effective_provider])

        effective_thinking = (
            self.thinking_level
            or runtime_config.get("thinking_level")
            or harness.model_profile.thinking_level
        )
        if effective_thinking:
            flags.extend(["--thinking", str(effective_thinking)])

        use_default_tools = bool(tool_config.get("use_default_tools", False))
        core_tools = tool_config.get("core_tools")
        if not use_default_tools and isinstance(core_tools, list):
            valid_tools = [str(t) for t in core_tools if str(t) in VALID_PI_TOOLS]
            if valid_tools:
                flags.extend(["--tools", ",".join(valid_tools)])

        if system_prompt_content.strip():
            flags.extend(["--system-prompt", system_prompt_content])

        for skill_dir in skill_dirs or []:
            flags.extend(["--skill", skill_dir])

        for skill_arg in skill_args:
            flags.extend(["--append-system-prompt", skill_arg])

        flags.extend(["--", instruction])
        return flags

    def build_command(
        self,
        pi_bin: str,
        instruction: str,
        harness: HarnessManifest,
        system_prompt_content: str,
        runtime_config: Dict[str, Any],
        tool_config: Dict[str, Any],
        skill_paths: List[Path],
        use_user_profile: bool = False,
        skill_dirs: Optional[List[str]] = None,
    ) -> List[str]:
        return [
            pi_bin,
            *self._build_pi_flags(
                instruction=instruction,
                harness=harness,
                system_prompt_content=system_prompt_content,
                runtime_config=runtime_config,
                tool_config=tool_config,
                skill_args=[str(p) for p in skill_paths],
                use_user_profile=use_user_profile,
                skill_dirs=skill_dirs,
            ),
        ]

    def _run_real_pi_json(
        self,
        cmd: List[str],
        workspace: Optional[Path],
        env: Optional[Dict[str, str]],
        timeout_sec: int,
        max_steps: int,
        max_tool_errors: int,
        early_stop_on_repeated_failure: bool,
    ) -> Tuple[List[StepTrace], bool, int, Optional[str], Optional[str]]:
        parser = PiNdjsonTraceParser(
            max_steps=max_steps,
            max_consecutive_tool_errors=max_tool_errors,
            early_stop_on_repeated_failure=early_stop_on_repeated_failure,
        )

        proc = subprocess.Popen(
            cmd,
            cwd=str(workspace) if workspace else None,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        line_queue: queue.Queue[Optional[str]] = queue.Queue()
        stderr_chunks: List[str] = []

        def _read_stdout() -> None:
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    line_queue.put(line)
            finally:
                line_queue.put(None)

        def _read_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_chunks.append(line)

        t_out = threading.Thread(target=_read_stdout, daemon=True)
        t_err = threading.Thread(target=_read_stderr, daemon=True)
        t_out.start()
        t_err.start()

        deadline = time.monotonic() + timeout_sec
        timed_out = False
        early_stopped = False

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                proc.kill()
                break

            try:
                item = line_queue.get(timeout=min(0.25, remaining))
            except queue.Empty:
                if proc.poll() is not None and not t_out.is_alive():
                    break
                continue

            if item is None:
                break

            stop_reason = parser.process_line(item)
            if stop_reason:
                early_stopped = True
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)

        t_out.join(timeout=1.0)
        t_err.join(timeout=1.0)

        steps = parser.finalize()
        stderr_text = "".join(stderr_chunks).strip()

        error_summary: Optional[str] = None
        if timed_out:
            error_summary = f"Pi agent execution timed out after {timeout_sec}s"
        elif early_stopped:
            error_summary = parser.early_stop_reason
        elif parser.api_error:
            error_summary = parser.api_error
        elif proc.returncode != 0:
            error_summary = (
                f"pi CLI exited with code {proc.returncode}: "
                f"{(stderr_text or parser.final_assistant_text)[:300]}"
            )

        passed = (
            proc.returncode == 0
            and not timed_out
            and not early_stopped
            and not parser.api_error
        )
        verifier_output = parser.final_assistant_text or stderr_text or None
        return steps, passed, parser.total_tokens, verifier_output, error_summary

    def run_task_in_container(
        self,
        task_id: str,
        instruction: str,
        container_id: str,
        harness: HarnessManifest,
        container_pi_root: str = "/opt/harness_pi",
        container_harness_dir: str = "/opt/harness_pi/harness",
        timeout_sec: int = 300,
    ) -> ExecutionTrace:
        """Execute Pi Coding Agent inside an already-running Docker container."""
        start_time = time.time()
        system_prompt_content = self._load_system_prompt(harness)
        runtime_config = self._load_yaml_surface(harness, "runtime_policy")
        tool_config = self._load_yaml_surface(harness, "tool_policy")
        skill_rels = self._resolve_skill_rel_paths(harness)
        skill_args = [
            f"{container_harness_dir.rstrip('/')}/{rel.lstrip('./')}"
            for rel in skill_rels
        ]

        max_steps = int(runtime_config.get("max_steps", 25))
        max_tool_errors = int(runtime_config.get("max_consecutive_same_tool_errors", 2))
        early_stop = bool(runtime_config.get("early_stop_on_repeated_failure", True))
        effective_user_profile = self.use_user_profile or bool(
            runtime_config.get("use_user_profile", False)
        )
        pi_root = container_pi_root.rstrip("/")

        bundled_node = f"{container_pi_root.rstrip('/')}/bin/node"
        probe_bundled = subprocess.run(
            ["docker", "exec", container_id, bundled_node, "--version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
        if probe_bundled.returncode == 0:
            node_exec = bundled_node
        else:
            probe_sys = subprocess.run(
                ["docker", "exec", container_id, "sh", "-c", "command -v node"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )
            sys_node = probe_sys.stdout.strip()
            if probe_sys.returncode == 0 and sys_node:
                node_exec = sys_node
            else:
                duration = (time.time() - start_time) * 1000.0
                return ExecutionTrace(
                    task_id=task_id,
                    harness_id=harness.harness_id,
                    model_name=harness.model_profile.model_name,
                    passed=False,
                    steps=[],
                    total_tokens=0,
                    total_latency_ms=duration,
                    error_summary=(
                        f"Bundled node failed ({probe_bundled.stderr.strip()}) and no system node found in container."
                    ),
                )

        # Copy read-only agent config into writable /tmp/pi_agent inside container
        prep_cmd = (
            f"rm -rf /tmp/pi_agent && mkdir -p /tmp/pi_agent && "
            f"cp -r {pi_root}/agent_ro/. /tmp/pi_agent/"
        )
        if effective_user_profile:
            prep_cmd += (
                f" && ([ ! -d {pi_root}/user_git ] || ln -sfn {pi_root}/user_git /tmp/pi_agent/git)"
                f" && ([ ! -d {pi_root}/user_npm ] || ln -sfn {pi_root}/user_npm /tmp/pi_agent/npm)"
            )
        prep = subprocess.run(
            [
                "docker",
                "exec",
                container_id,
                "sh",
                "-c",
                prep_cmd,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
        if prep.returncode != 0:
            duration = (time.time() - start_time) * 1000.0
            return ExecutionTrace(
                task_id=task_id,
                harness_id=harness.harness_id,
                model_name=harness.model_profile.model_name,
                passed=False,
                steps=[],
                total_tokens=0,
                total_latency_ms=duration,
                error_summary=f"Failed to prepare container /tmp/pi_agent: {prep.stderr.strip()}",
            )

        skill_dirs: List[str] = []
        if effective_user_profile:
            chk_skills = subprocess.run(
                ["docker", "exec", container_id, "test", "-d", f"{pi_root}/user_skills"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            if chk_skills.returncode == 0:
                skill_dirs.append(f"{pi_root}/user_skills")

        pi_cli_js = (
            f"{pi_root}/release/node_modules/"
            "@earendil-works/pi-coding-agent/dist/bundle/cli.js"
        )
        pi_flags = self._build_pi_flags(
            instruction=instruction,
            harness=harness,
            system_prompt_content=system_prompt_content,
            runtime_config=runtime_config,
            tool_config=tool_config,
            skill_args=skill_args,
            use_user_profile=effective_user_profile,
            skill_dirs=skill_dirs,
        )

        cmd = [
            "docker",
            "exec",
            "-e",
            "PI_CODING_AGENT_DIR=/tmp/pi_agent",
            "-e",
            "PI_OFFLINE=1",
            container_id,
            node_exec,
            pi_cli_js,
            *pi_flags,
        ]

        try:
            steps, passed, total_tokens, verifier_output, error_summary = self._run_real_pi_json(
                cmd=cmd,
                workspace=None,
                env=None,
                timeout_sec=timeout_sec,
                max_steps=max_steps,
                max_tool_errors=max_tool_errors,
                early_stop_on_repeated_failure=early_stop,
            )
        except Exception as ex:
            steps, passed, total_tokens, verifier_output, error_summary = (
                [],
                False,
                0,
                None,
                str(ex),
            )

        duration = (time.time() - start_time) * 1000.0
        return ExecutionTrace(
            task_id=task_id,
            harness_id=harness.harness_id,
            model_name=harness.model_profile.model_name,
            passed=passed,
            steps=steps,
            total_tokens=total_tokens,
            total_latency_ms=duration,
            error_summary=error_summary,
            verifier_output=verifier_output,
        )

    def run_task(
        self,
        task_id: str,
        instruction: str,
        workspace_dir: str,
        harness: HarnessManifest,
        timeout_sec: int = 120,
    ) -> ExecutionTrace:
        start_time = time.time()
        steps: List[StepTrace] = []
        passed = False
        total_tokens = 0
        error_summary: Optional[str] = None
        verifier_output: Optional[str] = None

        ws = Path(workspace_dir).resolve()
        ws.mkdir(parents=True, exist_ok=True)

        system_prompt_content = self._load_system_prompt(harness)
        runtime_config = self._load_yaml_surface(harness, "runtime_policy")
        tool_config = self._load_yaml_surface(harness, "tool_policy")
        skill_paths = self._resolve_skill_paths(harness)

        max_steps = int(runtime_config.get("max_steps", 25))
        max_tool_errors = int(runtime_config.get("max_consecutive_same_tool_errors", 2))
        early_stop = bool(runtime_config.get("early_stop_on_repeated_failure", True))
        effective_user_profile = self.use_user_profile or bool(
            runtime_config.get("use_user_profile", False)
        )
        skill_dirs: List[str] = []
        if effective_user_profile:
            profile_paths = self.resolve_user_profile_paths(self.user_skills_dir)
            if profile_paths.get("skills_dir"):
                skill_dirs.append(str(profile_paths["skills_dir"]))

        env = self._get_enhanced_env()
        pi_bin = (
            shutil.which(self.pi_executable, path=env.get("PATH"))
            if (self.pi_executable and self.mode != "simulated")
            else None
        )

        if self.mode == "real" and not pi_bin:
            duration = (time.time() - start_time) * 1000.0
            return ExecutionTrace(
                task_id=task_id,
                harness_id=harness.harness_id,
                model_name=harness.model_profile.model_name,
                passed=False,
                steps=[],
                total_tokens=0,
                total_latency_ms=duration,
                error_summary=f"pi CLI executable '{self.pi_executable}' not found in PATH",
                verifier_output=None,
            )

        if pi_bin and self.mode != "simulated":
            cmd = self.build_command(
                pi_bin=pi_bin,
                instruction=instruction,
                harness=harness,
                system_prompt_content=system_prompt_content,
                runtime_config=runtime_config,
                tool_config=tool_config,
                skill_paths=skill_paths,
                use_user_profile=effective_user_profile,
                skill_dirs=skill_dirs,
            )
            try:
                steps, passed, total_tokens, verifier_output, error_summary = self._run_real_pi_json(
                    cmd=cmd,
                    workspace=ws,
                    env=env,
                    timeout_sec=timeout_sec,
                    max_steps=max_steps,
                    max_tool_errors=max_tool_errors,
                    early_stop_on_repeated_failure=early_stop,
                )
            except Exception as ex:
                error_summary = str(ex)
                passed = False
        else:
            steps, passed, verifier_output, error_summary = self._simulate_pi_execution(
                instruction=instruction,
                workspace=ws,
                system_prompt=system_prompt_content,
                max_steps=max_steps,
                max_consecutive_tool_errors=max_tool_errors,
                model_name=harness.model_profile.model_name,
            )
            total_tokens = sum(s.tokens_used for s in steps) or 1500

        duration = (time.time() - start_time) * 1000.0
        return ExecutionTrace(
            task_id=task_id,
            harness_id=harness.harness_id,
            model_name=harness.model_profile.model_name,
            passed=passed,
            steps=steps,
            total_tokens=total_tokens,
            total_latency_ms=duration,
            error_summary=error_summary,
            verifier_output=verifier_output,
        )

    def _simulate_pi_execution(
        self,
        instruction: str,
        workspace: Path,
        system_prompt: str,
        max_steps: int,
        max_consecutive_tool_errors: int,
        model_name: str,
    ) -> tuple[List[StepTrace], bool, Optional[str], Optional[str]]:
        steps: List[StepTrace] = []

        has_env_discipline = "verify tool dependencies" in system_prompt.lower()
        has_self_verify = "compile or test checks" in system_prompt.lower()

        steps.append(
            StepTrace(
                step_number=1,
                thought="Exploring files with Pi 'read' and 'bash' tools",
                tool_calls=[
                    ToolCallRecord(
                        tool_name="read",
                        arguments={"path": "solution.py"},
                        output="File not found or empty",
                        status="warning",
                    )
                ],
                tokens_used=400,
            )
        )

        solution_file = workspace / "solution.py"

        if not has_env_discipline and "dependency" in instruction.lower():
            steps.append(
                StepTrace(
                    step_number=2,
                    thought="Attempting to run unverified command",
                    tool_calls=[
                        ToolCallRecord(
                            tool_name="bash",
                            arguments={"command": "uninstalled_tool --run"},
                            output="bash: uninstalled_tool: command not found",
                            status="error",
                        )
                    ],
                    tokens_used=350,
                )
            )
            return steps, False, "Execution failed due to missing dependency", "bash: uninstalled_tool: command not found"

        solution_content = "# Solved by Pi Agent\ndef solve():\n    return 42\n"
        solution_file.write_text(solution_content, encoding="utf-8")
        steps.append(
            StepTrace(
                step_number=len(steps) + 1,
                thought="Writing code with Pi 'write' tool",
                tool_calls=[
                    ToolCallRecord(
                        tool_name="write",
                        arguments={"path": "solution.py", "content": solution_content},
                        output="Wrote solution.py",
                        status="success",
                    )
                ],
                tokens_used=500,
            )
        )

        if has_self_verify:
            steps.append(
                StepTrace(
                    step_number=len(steps) + 1,
                    thought="Running deterministic verification before completion",
                    tool_calls=[
                        ToolCallRecord(
                            tool_name="bash",
                            arguments={"command": "python3 -m py_compile solution.py"},
                            output="Compiled successfully",
                            status="success",
                        )
                    ],
                    tokens_used=300,
                )
            )
            return steps, True, "All tests passed", None

        return steps, True, "Executed without explicit self-verification", None
