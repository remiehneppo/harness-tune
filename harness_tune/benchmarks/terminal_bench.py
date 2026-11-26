"""Terminal-Bench-2.0 adapter with real workspace fixtures and deterministic verifiers."""

from __future__ import annotations
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional
from harness_tune.benchmarks.base import AbstractBenchmark, BenchmarkTask, BenchmarkSplit
from harness_tune.agents.base import AbstractCodingAgent
from harness_tune.core.manifest import HarnessManifest
from harness_tune.core.trace import ExecutionTrace


class TerminalBenchAdapter(AbstractBenchmark):
    """Adapter for Terminal-Bench-2.0 dataset.

    Seeds isolated per-task workspace fixtures (logs, CSVs, SQLite DBs, Nginx configs,
    Dockerfiles) and evaluates agent outputs with strict deterministic verifiers.
    """

    def __init__(
        self,
        custom_tasks: Optional[List[BenchmarkTask]] = None,
        tasks_file: Optional[str | Path] = None,
        timeout_sec: int = 180,
    ):
        self.custom_tasks = custom_tasks
        self.tasks_file = Path(tasks_file).resolve() if tasks_file else None
        self.timeout_sec = timeout_sec

    def _load_tasks_from_file(self) -> List[BenchmarkTask]:
        assert self.tasks_file is not None
        raw = json.loads(self.tasks_file.read_text(encoding="utf-8"))
        items = raw.get("tasks", raw) if isinstance(raw, dict) else raw
        tasks: List[BenchmarkTask] = []
        for item in items:
            tasks.append(
                BenchmarkTask(
                    task_id=item["task_id"],
                    dataset_name=item.get("dataset_name", "terminal_bench_2"),
                    instruction=item["instruction"],
                    workspace_template=item.get("workspace_template"),
                    verification_script=item.get("verification_script"),
                    setup_files=item.get("setup_files", {}),
                    setup_script=item.get("setup_script"),
                    timeout_sec=int(item.get("timeout_sec", self.timeout_sec)),
                    metadata=item.get("metadata", {}),
                )
            )
        return tasks

    def get_split(self) -> BenchmarkSplit:
        if self.custom_tasks is not None:
            mid = len(self.custom_tasks) // 2
            return BenchmarkSplit(
                held_in=self.custom_tasks[:mid],
                held_out=self.custom_tasks[mid:],
            )

        if self.tasks_file is not None:
            loaded = self._load_tasks_from_file()
            held_in = [t for t in loaded if t.metadata.get("split") == "held_in"]
            held_out = [t for t in loaded if t.metadata.get("split") == "held_out"]
            if held_in or held_out:
                return BenchmarkSplit(held_in=held_in, held_out=held_out)
            mid = len(loaded) // 2
            return BenchmarkSplit(held_in=loaded[:mid], held_out=loaded[mid:])

        tasks = self._default_tasks()
        return BenchmarkSplit(
            held_in=tasks[:3],
            held_out=tasks[3:],
        )

    def _default_tasks(self) -> List[BenchmarkTask]:
        return [
            BenchmarkTask(
                task_id="tb_001_log_analyzer",
                dataset_name="terminal_bench_2",
                instruction=(
                    "Inspect access.log in the current workspace. Write an executable bash script "
                    "`parse_logs.sh` that accepts an optional log file path as `$1` (defaulting to `access.log`) "
                    "and prints `count=<N>` to stdout where `<N>` is the number of requests with HTTP status code 500 "
                    "(match the HTTP status field only — do not count `/items/500` in URLs or 502/503 responses, and skip malformed lines)."
                ),
                setup_files={
                    "access.log": (
                        '127.0.0.1 - - [21/Nov/2026:10:00:01 +0000] "GET /api/orders HTTP/1.1" 500 512\n'
                        '127.0.0.1 - - [21/Nov/2026:10:00:02 +0000] "GET /items/500 HTTP/1.1" 200 128\n'
                        '10.0.0.5 - - [21/Nov/2026:10:00:03 +0000] "POST /api/pay HTTP/1.1" 500 256\n'
                        '10.0.0.6 - - [21/Nov/2026:10:00:04 +0000] "GET /health HTTP/1.1" 200 64\n'
                        '10.0.0.7 - - [21/Nov/2026:10:00:05 +0000] "GET /old HTTP/1.1" 301 0\n'
                        '10.0.0.8 - - [21/Nov/2026:10:00:06 +0000] "GET /missing HTTP/1.1" 404 99\n'
                        '10.0.0.9 - - [21/Nov/2026:10:00:07 +0000] "PUT /api/users HTTP/1.1" 500 401\n'
                        '10.0.0.10 - - [21/Nov/2026:10:00:08 +0000] "GET /upstream HTTP/1.1" 502 120\n'
                        '10.0.0.11 - - [21/Nov/2026:10:00:09 +0000] "GET /busy HTTP/1.1" 503 120\n'
                        'MALFORMED_LINE_WITHOUT_HTTP_FIELDS 500\n'
                        '10.0.0.12 - - [21/Nov/2026:10:00:10 +0000] "DELETE /api/item/5 HTTP/1.1" 500 88\n'
                    )
                },
                verification_script=(
                    "test -f parse_logs.sh && bash -n parse_logs.sh && "
                    "python3 -c \""
                    "import subprocess, tempfile; from pathlib import Path; "
                    "out1 = subprocess.check_output(['bash', 'parse_logs.sh'], text=True).strip(); "
                    "assert out1 == 'count=4', f'Default access.log expected count=4, got {out1!r}'; "
                    "p = Path('tmp_verify.log'); "
                    "p.write_text('1.1.1.1 - - [x] \\\"GET /500 HTTP/1.1\\\" 200 10\\n"
                    "1.1.1.1 - - [x] \\\"GET /a HTTP/1.1\\\" 500 10\\n"
                    "1.1.1.1 - - [x] \\\"POST /b HTTP/1.1\\\" 500 10\\n'); "
                    "out2 = subprocess.check_output(['bash', 'parse_logs.sh', str(p)], text=True).strip(); "
                    "p.unlink(); "
                    "assert out2 == 'count=2', f'Custom file expected count=2, got {out2!r}'"
                    "\""
                ),
                timeout_sec=self.timeout_sec,
            ),
            BenchmarkTask(
                task_id="tb_002_dependency_install",
                dataset_name="terminal_bench_2",
                instruction=(
                    "Inspect `requirements.txt` in the current workspace. Write a bash script `setup.sh` "
                    "(using `set -euo pipefail`) that parses tool names listed after `# system:` in `requirements.txt`, "
                    "checks whether each tool exists via `command -v`, writes all missing tool names sorted alphabetically "
                    "one-per-line to `missing_bins.txt` (without exiting non-zero if tools are missing), and exits 0. "
                    "Run `setup.sh` to generate `missing_bins.txt`."
                ),
                setup_files={
                    "requirements.txt": (
                        "# system: bash, nonexistent_tool_beta, nonexistent_tool_alpha\n"
                        "requests==2.31.0\n"
                        "pyyaml==6.0.1\n"
                    )
                },
                verification_script=(
                    "test -f setup.sh && bash -n setup.sh && rm -f missing_bins.txt && bash setup.sh && "
                    "python3 -c \""
                    "from pathlib import Path; "
                    "lines = Path('missing_bins.txt').read_text(encoding='utf-8').splitlines(); "
                    "expected = ['nonexistent_tool_alpha', 'nonexistent_tool_beta']; "
                    "assert lines == expected, f'Expected {expected}, got {lines}'"
                    "\""
                ),
                timeout_sec=self.timeout_sec,
            ),
            BenchmarkTask(
                task_id="tb_003_process_monitor",
                dataset_name="terminal_bench_2",
                instruction=(
                    "Inspect `metrics_sample.csv` in the current workspace. Write `monitor.py` to read `metrics_sample.csv`, "
                    "skip rows with non-numeric `cpu_pct`, select rows strictly matching `cpu_pct > 80.0`, "
                    "sort matching rows by `timestamp` ascending, and write lines formatted as "
                    "`<timestamp> <host> HIGH_CPU <cpu_pct:.1f>` to `alert.log`. Run `monitor.py`."
                ),
                setup_files={
                    "metrics_sample.csv": (
                        "timestamp,host,cpu_pct,mem_pct\n"
                        "2026-11-21T10:02:00Z,web-02,88.5,64.0\n"
                        "2026-11-21T10:00:00Z,web-01,80.0,50.0\n"
                        "2026-11-21T10:01:00Z,db-01,N/A,70.0\n"
                        "2026-11-21T10:00:30Z,api-01,91.2,72.1\n"
                        "2026-11-21T10:03:00Z,cache-01,45.0,30.0\n"
                        "2026-11-21T10:01:30Z,worker-01,80.1,55.0\n"
                    )
                },
                verification_script=(
                    "test -f monitor.py && python3 -m py_compile monitor.py && rm -f alert.log && python3 monitor.py && "
                    "python3 -c \""
                    "from pathlib import Path; "
                    "expected = ["
                    "'2026-11-21T10:00:30Z api-01 HIGH_CPU 91.2', "
                    "'2026-11-21T10:01:30Z worker-01 HIGH_CPU 80.1', "
                    "'2026-11-21T10:02:00Z web-02 HIGH_CPU 88.5'"
                    "]; "
                    "got = Path('alert.log').read_text(encoding='utf-8').splitlines(); "
                    "assert got == expected, f'Expected {expected}, got {got}'"
                    "\""
                ),
                timeout_sec=self.timeout_sec,
            ),
            BenchmarkTask(
                task_id="tb_004_sqlite_migration",
                dataset_name="terminal_bench_2",
                instruction=(
                    "Inspect `test.db` in the current workspace (SQLite database with a `users` table). "
                    "Write `migration.sql` that alters `users` to add column `status TEXT NOT NULL DEFAULT 'active'` "
                    "and updates the row with `id = 2` to `status = 'suspended'`. Apply `migration.sql` to `test.db`."
                ),
                setup_script=(
                    "python3 -c \""
                    "import sqlite3; "
                    "conn = sqlite3.connect('test.db'); "
                    "conn.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL)'); "
                    "conn.executemany('INSERT INTO users (id, name, email) VALUES (?, ?, ?)', ["
                    "(1, 'Alice', 'alice@example.com'), "
                    "(2, 'Bob', 'bob@example.com'), "
                    "(3, 'Carol', 'carol@example.com')"
                    "]); "
                    "conn.commit(); conn.close()"
                    "\""
                ),
                verification_script=(
                    "test -f migration.sql && python3 -c \""
                    "import sqlite3; "
                    "conn = sqlite3.connect('test.db'); "
                    "cols = [r[1] for r in conn.execute('PRAGMA table_info(users)').fetchall()]; "
                    "assert 'status' in cols, f'Missing status column: {cols}'; "
                    "rows = conn.execute('SELECT id, name, status FROM users ORDER BY id').fetchall(); "
                    "expected = [(1, 'Alice', 'active'), (2, 'Bob', 'suspended'), (3, 'Carol', 'active')]; "
                    "assert rows == expected, f'Expected {expected}, got {rows}'"
                    "\""
                ),
                timeout_sec=self.timeout_sec,
            ),
            BenchmarkTask(
                task_id="tb_005_network_config",
                dataset_name="terminal_bench_2",
                instruction=(
                    "Inspect `nginx.conf` in the current workspace. Extract all unique numeric listening ports "
                    "from active (uncommented) `listen` directives (ignoring commented lines and `proxy_pass` ports), "
                    "sort them numerically ascending, and write them one-per-line to `ports.txt`."
                ),
                setup_files={
                    "nginx.conf": (
                        "server {\n"
                        "    listen 80 default_server;\n"
                        "    listen [::]:443 ssl;\n"
                        "    # listen 9999;\n"
                        "    server_name example.com;\n"
                        "    location / {\n"
                        "        proxy_pass http://127.0.0.1:9000;\n"
                        "    }\n"
                        "}\n"
                        "server {\n"
                        "    listen 127.0.0.1:8080;\n"
                        "    listen 80;\n"
                        "    # listen 7777 default_server;\n"
                        "}\n"
                    )
                },
                verification_script=(
                    "test -f ports.txt && python3 -c \""
                    "from pathlib import Path; "
                    "got = [line.strip() for line in Path('ports.txt').read_text(encoding='utf-8').splitlines() if line.strip()]; "
                    "assert got == ['80', '443', '8080'], f'Expected [\"80\", \"443\", \"8080\"], got {got}'"
                    "\""
                ),
                timeout_sec=self.timeout_sec,
            ),
            BenchmarkTask(
                task_id="tb_006_dockerfile_fix",
                dataset_name="terminal_bench_2",
                instruction=(
                    "Fix `Dockerfile` in the current workspace: "
                    "(1) fix the `RUn` typo to `RUN`, "
                    "(2) optimize layer caching by copying `requirements.txt` (`COPY requirements.txt ...`) and running "
                    "`RUN pip install --no-cache-dir -r requirements.txt` BEFORE `COPY . /app`, and "
                    "(3) fix `ENTRYPOINT` to valid double-quoted JSON exec form `ENTRYPOINT [\"python3\", \"app.py\"]`."
                ),
                setup_files={
                    "Dockerfile": (
                        "FROM python:3.11-slim\n"
                        "WORKDIR /app\n"
                        "COPY . /app\n"
                        "RUn pip install -r requirements.txt\n"
                        "EXPOSE 8000\n"
                        "ENTRYPOINT ['python3', 'app.py']\n"
                    )
                },
                verification_script=(
                    "test -f Dockerfile && python3 -c \""
                    "import json; from pathlib import Path; "
                    "lines = [l.strip() for l in Path('Dockerfile').read_text(encoding='utf-8').splitlines() if l.strip() and not l.strip().startswith('#')]; "
                    "assert not any(l.startswith('RUn ') for l in lines), 'Typo RUn still present'; "
                    "copy_req_idx = next((i for i, l in enumerate(lines) if l.upper().startswith('COPY') and 'requirements.txt' in l), -1); "
                    "pip_idx = next((i for i, l in enumerate(lines) if l.startswith('RUN ') and 'pip install' in l and '--no-cache-dir' in l and 'requirements.txt' in l), -1); "
                    "copy_all_idx = next((i for i, l in enumerate(lines) if l.upper().startswith('COPY . ')), -1); "
                    "assert copy_req_idx != -1, 'Missing COPY requirements.txt'; "
                    "assert pip_idx != -1, 'Missing RUN pip install --no-cache-dir -r requirements.txt'; "
                    "assert copy_all_idx != -1, 'Missing COPY . /app'; "
                    "assert copy_req_idx < pip_idx < copy_all_idx, f'Invalid layer order: {copy_req_idx}, {pip_idx}, {copy_all_idx}'; "
                    "ep_lines = [l for l in lines if l.upper().startswith('ENTRYPOINT ')]; "
                    "assert len(ep_lines) == 1, f'Expected single ENTRYPOINT, got {ep_lines}'; "
                    "ep_json = json.loads(ep_lines[0].split(None, 1)[1]); "
                    "assert ep_json == ['python3', 'app.py'], f'Unexpected ENTRYPOINT {ep_json}'"
                    "\""
                ),
                timeout_sec=self.timeout_sec,
            ),
        ]

    @staticmethod
    def _seed_workspace(task_ws: Path, task: BenchmarkTask) -> None:
        if task_ws.exists():
            shutil.rmtree(task_ws)
        task_ws.mkdir(parents=True, exist_ok=True)

        if task.workspace_template:
            tpl_dir = Path(task.workspace_template)
            if tpl_dir.exists() and tpl_dir.is_dir():
                shutil.copytree(tpl_dir, task_ws, dirs_exist_ok=True)

        for rel_path, content in task.setup_files.items():
            dest = (task_ws / rel_path).resolve()
            dest.relative_to(task_ws.resolve())
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        if task.setup_script:
            subprocess.run(
                task.setup_script,
                shell=True,
                cwd=str(task_ws),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
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
            self._seed_workspace(task_ws, task)

            # 1. Run agent on seeded task workspace
            trace = agent.run_task(
                task_id=task.task_id,
                instruction=task.instruction,
                workspace_dir=str(task_ws),
                harness=harness,
                timeout_sec=task.timeout_sec or self.timeout_sec,
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
                        detail = (res.stderr or res.stdout or f"exit code {res.returncode}").strip()
                        trace.passed = False
                        trace.verifier_output = f"Verifier script failed: {detail}"
                        if not trace.error_summary:
                            trace.error_summary = f"Verifier failed: {detail[:200]}"
                except Exception as ex:
                    trace.passed = False
                    trace.verifier_output = f"Verifier error: {str(ex)}"
                    if not trace.error_summary:
                        trace.error_summary = f"Verifier error: {str(ex)[:200]}"

            traces.append(trace)

        return traces
