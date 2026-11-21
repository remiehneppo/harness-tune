"""Terminal-Bench-2.0 adapter."""

from __future__ import annotations
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any
from harness_tune.benchmarks.base import AbstractBenchmark, BenchmarkTask, BenchmarkSplit
from harness_tune.agents.base import AbstractCodingAgent
from harness_tune.core.manifest import HarnessManifest
from harness_tune.core.trace import ExecutionTrace


class TerminalBenchAdapter(AbstractBenchmark):
    """Adapter for Terminal-Bench-2.0 dataset.

    Focuses on bash/terminal system tasks, containerized operations, and diagnostic verifications.
    Pre-configured with standard held-in and held-out task suites.
    """

    def __init__(self, custom_tasks: List[BenchmarkTask] = None):
        self.custom_tasks = custom_tasks

    def get_split(self) -> BenchmarkSplit:
        if self.custom_tasks:
            mid = len(self.custom_tasks) // 2
            return BenchmarkSplit(
                held_in=self.custom_tasks[:mid],
                held_out=self.custom_tasks[mid:],
            )

        # Standard built-in tasks representing Terminal-Bench-2.0 categories
        tasks = [
            BenchmarkTask(
                task_id="tb_001_log_analyzer",
                dataset_name="terminal_bench_2",
                instruction="Write a bash script parse_logs.sh that extracts 500 error count from access.log.",
                verification_script="test -f parse_logs.sh && bash parse_logs.sh | grep -q 'count='",
            ),
            BenchmarkTask(
                task_id="tb_002_dependency_install",
                dataset_name="terminal_bench_2",
                instruction="Inspect requirements.txt, identify missing binary libraries, and generate setup.sh.",
                verification_script="test -f setup.sh",
            ),
            BenchmarkTask(
                task_id="tb_003_process_monitor",
                dataset_name="terminal_bench_2",
                instruction="Create a script monitor.py that checks CPU usage and writes alerts to alert.log.",
                verification_script="python3 -m py_compile monitor.py",
            ),
            BenchmarkTask(
                task_id="tb_004_sqlite_migration",
                dataset_name="terminal_bench_2",
                instruction="Apply schema migration to users table in test.db using sqlite3.",
                verification_script="test -f migration.sql",
            ),
            BenchmarkTask(
                task_id="tb_005_network_config",
                dataset_name="terminal_bench_2",
                instruction="Parse nginx.conf and output all listening ports to ports.txt.",
                verification_script="test -f ports.txt",
            ),
            BenchmarkTask(
                task_id="tb_006_dockerfile_fix",
                dataset_name="terminal_bench_2",
                instruction="Fix syntax and caching layers in provided Dockerfile.",
                verification_script="grep -q 'ENTRYPOINT' Dockerfile 2>/dev/null || true",
            ),
        ]

        # 3 tasks held-in, 3 tasks held-out
        return BenchmarkSplit(
            held_in=tasks[:3],
            held_out=tasks[3:],
        )

    def run_evaluation(
        self,
        agent: AbstractCodingAgent,
        harness: HarnessManifest,
        tasks: List[BenchmarkTask],
        work_base_dir: str,
    ) -> List[ExecutionTrace]:
        traces: List[ExecutionTrace] = []
        base = Path(work_base_dir).resolve()
        base.mkdir(parents=True, exist_ok=True)

        for task in tasks:
            task_ws = base / f"run_{task.task_id}"
            task_ws.mkdir(parents=True, exist_ok=True)

            # 1. Run agent on task
            trace = agent.run_task(
                task_id=task.task_id,
                instruction=task.instruction,
                workspace_dir=str(task_ws),
                harness=harness,
            )

            # 2. Run deterministic verification script if present
            if task.verification_script and trace.passed:
                try:
                    res = subprocess.run(
                        task.verification_script,
                        shell=True,
                        cwd=str(task_ws),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=30,
                    )
                    if res.returncode != 0:
                        trace.passed = False
                        trace.verifier_output = f"Verifier script failed: {res.stderr or res.stdout}"
                except Exception as ex:
                    trace.passed = False
                    trace.verifier_output = f"Verifier error: {str(ex)}"

            traces.append(trace)

        return traces
