"""Unit tests for TerminalBenchAdapter fixtures and strict deterministic verifiers."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from harness_tune.agents.base import AbstractCodingAgent
from harness_tune.benchmarks.terminal_bench import TerminalBenchAdapter
from harness_tune.core.manifest import HarnessManifest, ModelProfile
from harness_tune.core.trace import ExecutionTrace


class _NoOpAgent(AbstractCodingAgent):
    def run_task(self, task_id, instruction, workspace_dir, harness, timeout_sec=120):
        return ExecutionTrace(task_id=task_id, harness_id=harness.harness_id, model_name="test", passed=True)


class _CorrectReferenceAgent(AbstractCodingAgent):
    def run_task(self, task_id, instruction, workspace_dir, harness, timeout_sec=120):
        ws = Path(workspace_dir)
        if task_id == "tb_001_log_analyzer":
            (ws / "parse_logs.sh").write_text(
                "#!/usr/bin/env bash\n"
                "LOG=\"${1:-access.log}\"\n"
                "c=$(awk 'match($0, /\" [0-9][0-9][0-9] /) { code=substr($0, RSTART+2, 3); if (code == \"500\") n++ } END { print n+0 }' \"$LOG\")\n"
                "echo \"count=$c\"\n",
                encoding="utf-8",
            )
        elif task_id == "tb_002_dependency_install":
            (ws / "setup.sh").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "line=$(grep '^# system:' requirements.txt | sed 's/^# system://')\n"
                "IFS=',' read -ra tools <<< \"$line\"\n"
                "missing=()\n"
                "for t in \"${tools[@]}\"; do\n"
                "  t=$(echo \"$t\" | xargs)\n"
                "  if ! command -v \"$t\" >/dev/null 2>&1; then missing+=(\"$t\"); fi\n"
                "done\n"
                "printf '%s\\n' \"${missing[@]}\" | sort > missing_bins.txt\n",
                encoding="utf-8",
            )
        elif task_id == "tb_003_process_monitor":
            (ws / "monitor.py").write_text(
                "import csv\n"
                "rows = []\n"
                "with open('metrics_sample.csv') as f:\n"
                "    for r in csv.DictReader(f):\n"
                "        try:\n"
                "            cpu = float(r['cpu_pct'])\n"
                "        except ValueError:\n"
                "            continue\n"
                "        if cpu > 80.0:\n"
                "            rows.append((r['timestamp'], r['host'], cpu))\n"
                "rows.sort(key=lambda x: x[0])\n"
                "with open('alert.log', 'w') as out:\n"
                "    for ts, host, cpu in rows:\n"
                "        out.write(f'{ts} {host} HIGH_CPU {cpu:.1f}\\n')\n",
                encoding="utf-8",
            )
        elif task_id == "tb_004_sqlite_migration":
            sql = (
                "ALTER TABLE users ADD COLUMN status TEXT NOT NULL DEFAULT 'active';\n"
                "UPDATE users SET status = 'suspended' WHERE id = 2;\n"
            )
            (ws / "migration.sql").write_text(sql, encoding="utf-8")
            conn = sqlite3.connect(ws / "test.db")
            conn.executescript(sql)
            conn.commit()
            conn.close()
        elif task_id == "tb_005_network_config":
            (ws / "ports.txt").write_text("80\n443\n8080\n", encoding="utf-8")
        elif task_id == "tb_006_dockerfile_fix":
            (ws / "Dockerfile").write_text(
                "FROM python:3.11-slim\n"
                "WORKDIR /app\n"
                "COPY requirements.txt /app/requirements.txt\n"
                "RUN pip install --no-cache-dir -r requirements.txt\n"
                "COPY . /app\n"
                "EXPOSE 8000\n"
                'ENTRYPOINT ["python3", "app.py"]\n',
                encoding="utf-8",
            )
        return ExecutionTrace(task_id=task_id, harness_id=harness.harness_id, model_name="test", passed=True)


class TestTerminalBenchFixtures(unittest.TestCase):
    def setUp(self):
        self.adapter = TerminalBenchAdapter(timeout_sec=30)
        self.split = self.adapter.get_split()
        self.manifest = HarnessManifest(
            harness_id="tb-test",
            version="1.0.0",
            model_profile=ModelProfile("local", "arex-2"),
            surfaces={},
        )

    def test_split_has_3_held_in_and_3_held_out(self):
        self.assertEqual(len(self.split.held_in), 3)
        self.assertEqual(len(self.split.held_out), 3)

    def test_noop_agent_fails_all_verifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            all_tasks = self.split.held_in + self.split.held_out
            traces = self.adapter.run_evaluation(_NoOpAgent(), self.manifest, all_tasks, tmp)
            self.assertTrue(all(not t.passed for t in traces))
            # Fixtures should be seeded into task workspaces
            self.assertTrue((Path(tmp) / "run_tb_001_log_analyzer" / "access.log").exists())
            self.assertTrue((Path(tmp) / "run_tb_004_sqlite_migration" / "test.db").exists())

    def test_reference_agent_passes_all_verifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            all_tasks = self.split.held_in + self.split.held_out
            traces = self.adapter.run_evaluation(_CorrectReferenceAgent(), self.manifest, all_tasks, tmp)
            for t in traces:
                self.assertTrue(t.passed, f"{t.task_id} failed: {t.verifier_output}")


if __name__ == "__main__":
    unittest.main()
