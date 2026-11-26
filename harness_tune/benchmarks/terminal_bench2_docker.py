"""Official Terminal-Bench 2.0 Docker benchmark adapter with checkpoint/resume."""

from __future__ import annotations
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from harness_tune.agents.base import AbstractCodingAgent
from harness_tune.agents.pi_agent import PiCodingAgent
from harness_tune.benchmarks.base import AbstractBenchmark, BenchmarkSplit, BenchmarkTask
from harness_tune.core.manifest import HarnessManifest
from harness_tune.core.fault_localizer import extract_verifier_signal
from harness_tune.core.trace import ExecutionTrace, StepTrace, ToolCallRecord

DEFAULT_TB2_REPO = "https://github.com/laude-institute/terminal-bench-2.git"
DEFAULT_TB2_DIR = Path("datasets") / "terminal-bench-2"


def parse_simple_toml(text: str) -> Dict[str, Any]:
    """Parse task.toml using Python 3.11+ tomllib when available, or stdlib fallback for 3.10."""
    try:
        import tomllib  # type: ignore[import-not-found]
        return tomllib.loads(text)
    except ModuleNotFoundError:
        pass

    root: Dict[str, Any] = {}
    current: Dict[str, Any] = root
    lines = text.splitlines()
    idx = 0

    def _strip_comment(s: str) -> str:
        in_str: Optional[str] = None
        escaped = False
        for pos, ch in enumerate(s):
            if escaped:
                escaped = False
                continue
            if ch == "\\" and in_str == '"':
                escaped = True
                continue
            if ch in ('"', "'"):
                if in_str is None:
                    in_str = ch
                elif in_str == ch:
                    in_str = None
            elif ch == "#" and in_str is None:
                return s[:pos]
        return s

    def _parse_val(raw_val: str) -> Any:
        v = raw_val.strip()
        if not v:
            return ""
        if v == "true":
            return True
        if v == "false":
            return False
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            if v.startswith('"'):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    return v[1:-1]
            return v[1:-1]
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            if not inner:
                return []
            items: List[Any] = []
            buf: List[str] = []
            in_str: Optional[str] = None
            escaped = False
            for ch in inner:
                if escaped:
                    buf.append(ch)
                    escaped = False
                    continue
                if ch == "\\" and in_str == '"':
                    buf.append(ch)
                    escaped = True
                    continue
                if ch in ('"', "'"):
                    if in_str is None:
                        in_str = ch
                    elif in_str == ch:
                        in_str = None
                    buf.append(ch)
                elif ch == "," and in_str is None:
                    item_s = "".join(buf).strip()
                    if item_s:
                        items.append(_parse_val(item_s))
                    buf = []
                else:
                    buf.append(ch)
            tail = "".join(buf).strip()
            if tail:
                items.append(_parse_val(tail))
            return items
        try:
            if "." in v or "e" in v.lower():
                return float(v)
            return int(v)
        except ValueError:
            return v

    while idx < len(lines):
        raw_line = _strip_comment(lines[idx]).strip()
        idx += 1
        if not raw_line:
            continue

        if raw_line.startswith("[[") and raw_line.endswith("]]"):
            path_parts = [p.strip() for p in raw_line[2:-2].split(".") if p.strip()]
            target: Any = root
            for part in path_parts[:-1]:
                if part not in target or not isinstance(target[part], dict):
                    target[part] = {}
                target = target[part]
            arr_key = path_parts[-1]
            if arr_key not in target or not isinstance(target[arr_key], list):
                target[arr_key] = []
            new_obj: Dict[str, Any] = {}
            target[arr_key].append(new_obj)
            current = new_obj
            continue

        if raw_line.startswith("[") and raw_line.endswith("]"):
            path_parts = [p.strip() for p in raw_line[1:-1].split(".") if p.strip()]
            target = root
            for part in path_parts:
                if part not in target or not isinstance(target[part], dict):
                    target[part] = {}
                target = target[part]
            current = target
            continue

        if "=" in raw_line:
            k, v = raw_line.split("=", 1)
            key = k.strip().strip('"').strip("'")
            val_str = v.strip()
            while val_str.startswith("[") and not val_str.endswith("]") and idx < len(lines):
                next_part = _strip_comment(lines[idx]).strip()
                idx += 1
                val_str += " " + next_part
            current[key] = _parse_val(val_str)

    return root


def parse_reward_output(reward_text: Optional[str], verify_returncode: int) -> bool:
    """Interpret Harbor/TB 2.0 `/logs/verifier/reward.txt` with return-code fallback."""
    if reward_text is not None:
        cleaned = reward_text.strip()
        if cleaned:
            try:
                return float(cleaned.splitlines()[-1].strip()) >= 1.0
            except ValueError:
                pass
    return verify_returncode == 0


def trace_from_dict(data: Dict[str, Any]) -> ExecutionTrace:
    """Reconstruct an ExecutionTrace from serialized checkpoint JSON."""
    steps: List[StepTrace] = []
    for s in data.get("steps", []):
        tcs = [
            ToolCallRecord(
                tool_name=tc.get("tool_name", "unknown"),
                arguments=tc.get("arguments", {}),
                output=tc.get("output", ""),
                status=tc.get("status", "success"),
                duration_ms=float(tc.get("duration_ms", 0.0)),
            )
            for tc in s.get("tool_calls", [])
        ]
        steps.append(
            StepTrace(
                step_number=int(s.get("step_number", len(steps) + 1)),
                thought=s.get("thought"),
                tool_calls=tcs,
                tokens_used=int(s.get("tokens_used", 0)),
                duration_ms=float(s.get("duration_ms", 0.0)),
            )
        )
    return ExecutionTrace(
        task_id=data["task_id"],
        harness_id=data.get("harness_id", "unknown"),
        model_name=data.get("model_name", "unknown"),
        passed=bool(data.get("passed", False)),
        steps=steps,
        total_tokens=int(data.get("total_tokens", 0)),
        total_latency_ms=float(data.get("total_latency_ms", 0.0)),
        error_summary=data.get("error_summary"),
        verifier_output=data.get("verifier_output"),
    )


class TerminalBench2DockerAdapter(AbstractBenchmark):
    """Runs official Terminal-Bench 2.0 tasks (89 Docker environments) via Docker + Pi Agent."""

    def __init__(
        self,
        dataset_dir: Optional[str | Path] = None,
        timeout_sec: int = 300,
        docker_network: str = "bridge",
        limit: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
        min_free_gb: float = 15.0,
        auto_clone: bool = True,
        repo_url: str = DEFAULT_TB2_REPO,
        local_port: int = 8080,
    ):
        self.dataset_dir = Path(dataset_dir or DEFAULT_TB2_DIR).resolve()
        self.timeout_sec = timeout_sec
        self.docker_network = docker_network
        self.limit = limit
        self.task_ids = [t.strip() for t in task_ids if t.strip()] if task_ids else None
        self.min_free_gb = min_free_gb
        self.auto_clone = auto_clone
        self.repo_url = repo_url
        self.local_port = local_port

    def ensure_dataset(self) -> Path:
        if not self.dataset_dir.exists() or not any(self.dataset_dir.iterdir()):
            if not self.auto_clone:
                raise FileNotFoundError(f"Terminal-Bench 2.0 dataset not found: {self.dataset_dir}")
            self.dataset_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, str(self.dataset_dir)],
                check=True,
                stdin=subprocess.DEVNULL,
            )
        return self.dataset_dir

    def discover_tasks(self) -> List[BenchmarkTask]:
        root = self.ensure_dataset()
        search_root = root / "tasks" if (root / "tasks").is_dir() else root
        candidates = sorted(search_root.glob("*/task.toml"), key=lambda p: p.parent.name)

        tasks: List[BenchmarkTask] = []
        for toml_path in candidates:
            task_dir = toml_path.parent
            instr_path = task_dir / "instruction.md"
            test_sh = task_dir / "tests" / "test.sh"
            if not instr_path.exists() or not test_sh.exists():
                continue

            parsed = parse_simple_toml(toml_path.read_text(encoding="utf-8"))
            meta_sec = parsed.get("metadata", {}) if isinstance(parsed.get("metadata"), dict) else {}
            env_sec = parsed.get("environment", {}) if isinstance(parsed.get("environment"), dict) else {}
            ver_sec = parsed.get("verifier", {}) if isinstance(parsed.get("verifier"), dict) else {}
            agent_sec = parsed.get("agent", {}) if isinstance(parsed.get("agent"), dict) else {}

            task_id = task_dir.name
            if self.task_ids is not None and task_id not in self.task_ids:
                continue

            tasks.append(
                BenchmarkTask(
                    task_id=task_id,
                    dataset_name="terminal_bench_2_full",
                    instruction=instr_path.read_text(encoding="utf-8").strip(),
                    timeout_sec=self.timeout_sec,
                    metadata={
                        "task_dir": str(task_dir.resolve()),
                        "docker_image": env_sec.get("docker_image"),
                        "memory_mb": int(env_sec.get("memory_mb", 2048) or 2048),
                        "cpus": env_sec.get("cpus", 1),
                        "category": str(meta_sec.get("category", "general")),
                        "difficulty": str(meta_sec.get("difficulty", "medium")),
                        "tags": meta_sec.get("tags", []),
                        "verifier_timeout_sec": float(ver_sec.get("timeout_sec", 300.0) or 300.0),
                        "agent_timeout_sec": float(agent_sec.get("timeout_sec", self.timeout_sec) or self.timeout_sec),
                    },
                )
            )

        if self.limit is not None and self.limit >= 0:
            tasks = tasks[: self.limit]
        return tasks

    def get_split(self) -> BenchmarkSplit:
        tasks = self.discover_tasks()
        # Deterministic stratified split by (category, difficulty, task_id)
        ordered = sorted(
            tasks,
            key=lambda t: (
                str(t.metadata.get("category", "")),
                str(t.metadata.get("difficulty", "")),
                t.task_id,
            ),
        )
        if len(ordered) <= 1:
            return BenchmarkSplit(held_in=ordered, held_out=[])
        held_in = [t for idx, t in enumerate(ordered) if idx % 2 == 1]
        held_out = [t for idx, t in enumerate(ordered) if idx % 2 == 0]
        return BenchmarkSplit(held_in=held_in, held_out=held_out)

    @staticmethod
    def _load_checkpoint(checkpoint_file: Path) -> Dict[str, ExecutionTrace]:
        completed: Dict[str, ExecutionTrace] = {}
        if not checkpoint_file.exists():
            return completed
        for line in checkpoint_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                trace = trace_from_dict(data)
                completed[trace.task_id] = trace
            except Exception:
                continue
        return completed

    @staticmethod
    def _append_checkpoint(checkpoint_file: Path, trace: ExecutionTrace) -> None:
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")
            f.flush()

    def _ensure_docker_image(self, task: BenchmarkTask) -> Tuple[str, bool]:
        """Return (image_ref, pulled_or_built_for_task)."""
        image = task.metadata.get("docker_image")
        task_dir = Path(task.metadata["task_dir"])

        if image:
            inspect = subprocess.run(
                ["docker", "image", "inspect", str(image)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
            if inspect.returncode == 0:
                return str(image), False

            pull = subprocess.run(
                ["docker", "pull", str(image)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=600,
            )
            if pull.returncode == 0:
                return str(image), True

        env_dir = task_dir / "environment"
        if env_dir.is_dir():
            local_tag = f"tb2_local_{task.task_id}:latest"
            subprocess.run(
                ["docker", "build", "-t", local_tag, str(env_dir)],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=600,
            )
            return local_tag, True

        raise RuntimeError(f"Could not pull '{image}' or build Dockerfile for task '{task.task_id}'")

    def _start_container(
        self,
        task: BenchmarkTask,
        image: str,
        node_bin: Path,
        pi_release_dir: Path,
        tmp_agent_dir: Path,
        harness_base_dir: Path,
        user_profile_paths: Optional[Dict[str, Any]] = None,
    ) -> str:
        cid_name = f"harness_tb2_{task.task_id[:32]}_{uuid.uuid4().hex[:8]}"
        mem_mb = min(max(512, int(task.metadata.get("memory_mb", 2048))), 8192)
        net_args = (
            ["--network", "host"]
            if self.docker_network == "host"
            else ["--add-host=host.docker.internal:host-gateway"]
        )

        base_flags = [
            "docker",
            "run",
            "-d",
            "--name",
            cid_name,
            *net_args,
            "--memory",
            f"{mem_mb}m",
            "-v",
            f"{node_bin}:/opt/harness_pi/bin/node:ro",
            "-v",
            f"{pi_release_dir}:/opt/harness_pi/release:ro",
            "-v",
            f"{tmp_agent_dir}:/opt/harness_pi/agent_ro:ro",
            "-v",
            f"{harness_base_dir}:/opt/harness_pi/harness:ro",
        ]
        if user_profile_paths:
            if user_profile_paths.get("git_dir"):
                base_flags.extend(["-v", f"{user_profile_paths['git_dir']}:/opt/harness_pi/user_git:ro"])
            if user_profile_paths.get("npm_dir"):
                base_flags.extend(["-v", f"{user_profile_paths['npm_dir']}:/opt/harness_pi/user_npm:ro"])
            if user_profile_paths.get("skills_dir"):
                base_flags.extend(["-v", f"{user_profile_paths['skills_dir']}:/opt/harness_pi/user_skills:ro"])

        proc = subprocess.run(
            [*base_flags, image, "sh", "-c", "sleep infinity"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            # Retry overriding any custom image ENTRYPOINT
            subprocess.run(
                ["docker", "rm", "-f", cid_name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc = subprocess.run(
                [*base_flags, "--entrypoint", "/bin/sh", image, "-c", "sleep infinity"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
            )
        if proc.returncode != 0:
            raise RuntimeError(f"docker run failed for {task.task_id}: {proc.stderr.strip()}")
        return cid_name

    @staticmethod
    def _run_container_verifier(
        cid: str,
        tests_dir: Path,
        verifier_timeout_sec: int,
    ) -> Tuple[bool, str]:
        prep = subprocess.run(
            ["docker", "exec", cid, "mkdir", "-p", "/tests", "/logs/verifier"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if prep.returncode != 0:
            return False, f"Verifier mkdir failed: {prep.stderr.strip()}"

        cp_proc = subprocess.run(
            ["docker", "cp", f"{str(tests_dir).rstrip('/')}/.", f"{cid}:/tests/"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        if cp_proc.returncode != 0:
            return False, f"Verifier docker cp failed: {cp_proc.stderr.strip()}"

        try:
            ver_proc = subprocess.run(
                ["docker", "exec", "-e", "TEST_DIR=/tests", cid, "bash", "/tests/test.sh"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=verifier_timeout_sec,
            )
            ver_rc = ver_proc.returncode
            ver_out = (ver_proc.stdout + "\n" + ver_proc.stderr).strip()
        except subprocess.TimeoutExpired:
            return False, f"Verifier script /tests/test.sh timed out after {verifier_timeout_sec}s"

        reward_proc = subprocess.run(
            ["docker", "exec", cid, "cat", "/logs/verifier/reward.txt"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
        reward_text = reward_proc.stdout if reward_proc.returncode == 0 else None
        passed = parse_reward_output(reward_text, ver_rc)
        return passed, ver_out[-2500:]

    def _maybe_prune_image(self, image: str, created_for_task: bool) -> None:
        if not created_for_task:
            return
        free_gb = shutil.disk_usage("/").free / (1024**3)
        if free_gb < self.min_free_gb:
            subprocess.run(
                ["docker", "rmi", "-f", image],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def run_evaluation(
        self,
        agent: AbstractCodingAgent,
        harness: HarnessManifest,
        tasks: List[BenchmarkTask],
        work_base_dir: str,
        checkpoint_file: Optional[Path | str] = None,
    ) -> List[ExecutionTrace]:
        if not isinstance(agent, PiCodingAgent):
            raise TypeError("TerminalBench2DockerAdapter requires PiCodingAgent with container support.")

        base = Path(work_base_dir).resolve()
        base.mkdir(parents=True, exist_ok=True)
        ckpt_path = Path(checkpoint_file).resolve() if checkpoint_file else (base / "checkpoint.jsonl")

        completed_map = self._load_checkpoint(ckpt_path)
        total = len(tasks)
        if completed_map:
            resumed = sum(1 for t in tasks if t.task_id in completed_map)
            if resumed:
                print(
                    f"   ♻️  Resuming from checkpoint ({resumed}/{total} tasks already completed)",
                    flush=True,
                )

        node_bin, pi_release_dir = PiCodingAgent.resolve_host_pi_bundle()
        base_url = (
            f"http://127.0.0.1:{self.local_port}/v1"
            if self.docker_network == "host"
            else f"http://host.docker.internal:{self.local_port}/v1"
        )
        runtime_cfg = PiCodingAgent._load_yaml_surface(harness, "runtime_policy")
        use_user_profile = agent.use_user_profile or bool(runtime_cfg.get("use_user_profile", False))
        user_profile_paths = (
            PiCodingAgent.resolve_user_profile_paths(agent.user_skills_dir)
            if use_user_profile
            else None
        )
        provider_compat = PiCodingAgent.resolve_provider_compat(harness.model_profile.provider)

        passed_so_far = sum(1 for t in tasks if t.task_id in completed_map and completed_map[t.task_id].passed)
        done_so_far = sum(1 for t in tasks if t.task_id in completed_map)

        for idx, task in enumerate(tasks, start=1):
            if task.task_id in completed_map:
                continue

            task_start = time.time()
            task_dir = Path(task.metadata["task_dir"])
            tests_dir = task_dir / "tests"
            cid: Optional[str] = None
            image: Optional[str] = None
            created_for_task = False
            tmp_agent_dir = Path(tempfile.mkdtemp(prefix=f"tb2_agent_{task.task_id[:20]}_"))

            try:
                PiCodingAgent.write_isolated_container_models_json(
                    target_dir=tmp_agent_dir,
                    base_url=base_url,
                    model_profile=harness.model_profile,
                    packages=user_profile_paths.get("packages") if user_profile_paths else None,
                    compat=provider_compat,
                )
                image, created_for_task = self._ensure_docker_image(task)
                assert harness.base_dir is not None
                cid = self._start_container(
                    task=task,
                    image=image,
                    node_bin=node_bin,
                    pi_release_dir=pi_release_dir,
                    tmp_agent_dir=tmp_agent_dir,
                    harness_base_dir=harness.base_dir.resolve(),
                    user_profile_paths=user_profile_paths,
                )

                trace = agent.run_task_in_container(
                    task_id=task.task_id,
                    instruction=task.instruction,
                    container_id=cid,
                    harness=harness,
                    timeout_sec=task.timeout_sec or self.timeout_sec,
                )

                ver_timeout = min(max(300, int(task.metadata.get("verifier_timeout_sec", 300))), 900)
                ver_passed, ver_out = self._run_container_verifier(
                    cid=cid,
                    tests_dir=tests_dir,
                    verifier_timeout_sec=ver_timeout,
                )
                trace.passed = ver_passed
                if ver_out:
                    trace.verifier_output = ver_out
                if not ver_passed and not trace.error_summary:
                    sig = extract_verifier_signal(ver_out, max_chars=200)
                    tail = sig.splitlines()[0] if sig else (ver_out.splitlines()[-1] if ver_out else "Verifier failed")
                    trace.error_summary = f"TB2 verifier failed: {tail[:200]}"
            except Exception as ex:
                trace = ExecutionTrace(
                    task_id=task.task_id,
                    harness_id=harness.harness_id,
                    model_name=harness.model_profile.model_name,
                    passed=False,
                    steps=[],
                    total_tokens=0,
                    total_latency_ms=(time.time() - task_start) * 1000.0,
                    error_summary=f"Docker task execution error: {ex}",
                    verifier_output=None,
                )
            finally:
                if cid:
                    subprocess.run(
                        ["docker", "rm", "-f", cid],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                shutil.rmtree(tmp_agent_dir, ignore_errors=True)
                if image:
                    self._maybe_prune_image(image, created_for_task)

            self._append_checkpoint(ckpt_path, trace)
            completed_map[task.task_id] = trace
            done_so_far += 1
            if trace.passed:
                passed_so_far += 1

            status_lbl = "PASS" if trace.passed else "FAIL"
            pct = (passed_so_far / done_so_far) * 100.0
            elapsed = time.time() - task_start
            print(
                f"   [{done_so_far}/{total}] {task.task_id} -> {status_lbl} "
                f"(running: {passed_so_far}/{done_so_far} = {pct:.1f}%, "
                f"steps={len(trace.steps)}, tokens={trace.total_tokens}, time={elapsed:.1f}s)",
                flush=True,
            )

        return [completed_map[t.task_id] for t in tasks]
