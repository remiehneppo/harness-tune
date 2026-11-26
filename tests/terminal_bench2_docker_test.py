"""Unit tests for TerminalBench2DockerAdapter, checkpointing, TOML parsing, and baseline reuse."""

import json
import tempfile
import unittest
from pathlib import Path

from harness_tune.agents.base import AbstractCodingAgent
from harness_tune.agents.pi_agent import PiCodingAgent
from harness_tune.benchmarks.base import AbstractBenchmark, BenchmarkSplit, BenchmarkTask
from harness_tune.benchmarks.terminal_bench2_docker import (
    TerminalBench2DockerAdapter,
    parse_reward_output,
    parse_simple_toml,
)
from harness_tune.core.manifest import HarnessManifest, ModelProfile
from harness_tune.core.proposer import DedicatedProposer
from harness_tune.core.trace import ExecutionTrace
from harness_tune.engine import HarnessEvolutionEngine


SAMPLE_TASK_TOML = """
schema_version = "1.1"
artifacts = []

[task]
name = "terminal-bench/sample-task"
keywords = ["python", "cli"]
[[task.authors]]
name = "alice"
email = "alice@example.com"

[metadata]
difficulty = "hard"
category = "software-engineering"
tags = ["debugging", "tests"]

[verifier]
timeout_sec = 900.0

[agent]
timeout_sec = 600.0

[environment]
docker_image = "alexgshaw/sample-task:20251031"
cpus = 1
memory_mb = 4096
allow_internet = true
"""


class TestTerminalBench2Docker(unittest.TestCase):
    def test_parse_simple_toml_with_array_tables(self):
        parsed = parse_simple_toml(SAMPLE_TASK_TOML)
        self.assertEqual(parsed["schema_version"], "1.1")
        self.assertEqual(parsed["task"]["name"], "terminal-bench/sample-task")
        self.assertEqual(parsed["task"]["authors"][0]["name"], "alice")
        self.assertEqual(parsed["metadata"]["difficulty"], "hard")
        self.assertEqual(parsed["metadata"]["category"], "software-engineering")
        self.assertEqual(parsed["metadata"]["tags"], ["debugging", "tests"])
        self.assertEqual(parsed["environment"]["docker_image"], "alexgshaw/sample-task:20251031")
        self.assertEqual(parsed["environment"]["memory_mb"], 4096)
        self.assertEqual(parsed["environment"]["allow_internet"], True)

    def test_parse_reward_output(self):
        self.assertTrue(parse_reward_output("1\n", 1))
        self.assertTrue(parse_reward_output("1.0", 0))
        self.assertFalse(parse_reward_output("0\n", 0))
        self.assertFalse(parse_reward_output("0.5", 0))
        self.assertTrue(parse_reward_output(None, 0))
        self.assertFalse(parse_reward_output(None, 1))
        self.assertTrue(parse_reward_output("", 0))

    def test_discover_and_stratified_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for idx in range(6):
                tdir = root / f"task_{idx:02d}"
                (tdir / "tests").mkdir(parents=True)
                (tdir / "instruction.md").write_text(f"Solve task {idx}", encoding="utf-8")
                (tdir / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n", encoding="utf-8")
                cat = "debugging" if idx < 4 else "security"
                diff = "easy" if idx % 2 == 0 else "hard"
                (tdir / "task.toml").write_text(
                    f'[metadata]\ncategory = "{cat}"\ndifficulty = "{diff}"\n'
                    f'[environment]\ndocker_image = "img:{idx}"\nmemory_mb = 2048\n',
                    encoding="utf-8",
                )

            adapter = TerminalBench2DockerAdapter(dataset_dir=root, auto_clone=False)
            split = adapter.get_split()
            self.assertEqual(len(split.held_in), 3)
            self.assertEqual(len(split.held_out), 3)
            self.assertEqual(
                {t.task_id for t in split.held_in} & {t.task_id for t in split.held_out},
                set(),
            )

    def test_checkpoint_append_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "checkpoint.jsonl"
            t1 = ExecutionTrace(task_id="task_a", harness_id="h1", model_name="arex-2", passed=True, total_tokens=123)
            t2 = ExecutionTrace(task_id="task_b", harness_id="h1", model_name="arex-2", passed=False, error_summary="boom")
            TerminalBench2DockerAdapter._append_checkpoint(ckpt, t1)
            TerminalBench2DockerAdapter._append_checkpoint(ckpt, t2)

            loaded = TerminalBench2DockerAdapter._load_checkpoint(ckpt)
            self.assertEqual(set(loaded.keys()), {"task_a", "task_b"})
            self.assertTrue(loaded["task_a"].passed)
            self.assertEqual(loaded["task_a"].total_tokens, 123)
            self.assertFalse(loaded["task_b"].passed)
            self.assertEqual(loaded["task_b"].error_summary, "boom")

    def test_isolated_container_models_json_has_no_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            profile = ModelProfile(
                provider="local",
                model_name="arex-2",
                context_window=262144,
                thinking_level="medium",
            )
            PiCodingAgent.write_isolated_container_models_json(
                target_dir=out,
                base_url="http://host.docker.internal:8080/v1",
                model_profile=profile,
                packages=["git:github.com/DietrichGebert/ponytail", "npm:pi-codegraph-extension"],
                compat={"thinkingFormat": "qwen-chat-template", "supportsStore": False},
            )
            models_data = json.loads((out / "models.json").read_text(encoding="utf-8"))
            self.assertEqual(list(models_data["providers"].keys()), ["local"])
            local_cfg = models_data["providers"]["local"]
            self.assertEqual(local_cfg["baseUrl"], "http://host.docker.internal:8080/v1")
            self.assertEqual(local_cfg["apiKey"], "local-unused")
            self.assertEqual(local_cfg["compat"]["thinkingFormat"], "qwen-chat-template")
            self.assertEqual(local_cfg["compat"]["supportsStore"], False)
            self.assertEqual([m["id"] for m in local_cfg["models"]], ["arex-2"])
            settings_data = json.loads((out / "settings.json").read_text(encoding="utf-8"))
            self.assertEqual(
                settings_data["packages"],
                ["git:github.com/DietrichGebert/ponytail", "npm:pi-codegraph-extension"],
            )

    def test_load_user_default_harness_manifest(self):
        manifest_path = (
            Path(__file__).resolve().parent.parent
            / "templates"
            / "pi_user_harness"
            / "configs"
            / "harness_arex2_user.manifest.json"
        )
        manifest = HarnessManifest.load(manifest_path)
        self.assertEqual(manifest.harness_id, "pi-arex2-user-default")
        self.assertEqual(manifest.model_profile.thinking_level, "high")
        self.assertTrue((manifest.base_dir / manifest.surfaces["system_prompt"]).exists())

    def test_engine_reuses_precomputed_baseline_and_fast_screens(self):
        class _CountingBenchmark(AbstractBenchmark):
            def __init__(self):
                self.calls = []
                self.split = BenchmarkSplit(
                    held_in=[
                        BenchmarkTask("in_pass", "tb2", "ok"),
                        BenchmarkTask("in_fail", "tb2", "fix dependency"),
                    ],
                    held_out=[
                        BenchmarkTask("out_pass", "tb2", "ok"),
                        BenchmarkTask("out_fail", "tb2", "fix dependency"),
                    ],
                )

            def get_split(self):
                return self.split

            def run_evaluation(self, agent, harness, tasks, work_base_dir, checkpoint_file=None):
                self.calls.append([t.task_id for t in tasks])
                has_env = "verify tool dependencies" in (harness.base_dir / "prompts/system.md").read_text().lower()
                traces = []
                for t in tasks:
                    passed = True if ("pass" in t.task_id or has_env) else False
                    traces.append(
                        ExecutionTrace(
                            task_id=t.task_id,
                            harness_id=harness.harness_id,
                            model_name="arex-2",
                            passed=passed,
                            error_summary=None if passed else "bash: x: command not found",
                        )
                    )
                return traces

        class _DummyAgent(AbstractCodingAgent):
            def run_task(self, task_id, instruction, workspace_dir, harness, timeout_sec=120):
                raise AssertionError("Should not be called directly")

        with tempfile.TemporaryDirectory() as h_tmp, tempfile.TemporaryDirectory() as o_tmp:
            h_dir = Path(h_tmp)
            (h_dir / "configs").mkdir()
            (h_dir / "prompts").mkdir()
            (h_dir / "configs" / "runtime.yaml").write_text("max_steps: 10\n", encoding="utf-8")
            (h_dir / "configs" / "tools.yaml").write_text("granularity: minimal\n", encoding="utf-8")
            (h_dir / "configs" / "verification.yaml").write_text("non_regression_margin: 0.0\n", encoding="utf-8")
            (h_dir / "prompts" / "system.md").write_text("Minimal prompt\n", encoding="utf-8")

            manifest = HarnessManifest(
                harness_id="tb2_test",
                version="1.0.0",
                model_profile=ModelProfile("local", "arex-2"),
                surfaces={
                    "system_prompt": "prompts/system.md",
                    "runtime_policy": "configs/runtime.yaml",
                    "tool_policy": "configs/tools.yaml",
                    "verification_policy": "configs/verification.yaml",
                },
                base_dir=h_dir,
            )

            precomputed = {
                "held_in": [
                    ExecutionTrace("in_pass", "tb2_test", "arex-2", True),
                    ExecutionTrace("in_fail", "tb2_test", "arex-2", False, error_summary="command not found"),
                ],
                "held_out": [
                    ExecutionTrace("out_pass", "tb2_test", "arex-2", True),
                    ExecutionTrace("out_fail", "tb2_test", "arex-2", False, error_summary="command not found"),
                ],
            }
            # Ensure failure miner picks up FC_MISSING_DEP
            from harness_tune.core.trace import StepTrace, ToolCallRecord
            precomputed["held_in"][1].steps = [
                StepTrace(1, tool_calls=[ToolCallRecord("bash", {}, "bash: foo: command not found", "error")])
            ]

            bench = _CountingBenchmark()
            proposer = DedicatedProposer(proposer_model_name="heuristic")
            engine = HarnessEvolutionEngine(_DummyAgent(), bench, proposer, o_tmp)
            res = engine.run_optimization_cycle(
                baseline_manifest=manifest,
                max_proposals=1,
                precomputed_baseline=precomputed,
                fast_screen_failed_only=True,
            )
            self.assertEqual(res["status"], "promoted")
            # First call is Stage A (only in_fail), then full held_in, then held_out (no baseline run!)
            self.assertEqual(bench.calls[0], ["in_fail"])
            self.assertEqual(bench.calls[1], ["in_pass", "in_fail"])
            self.assertEqual(bench.calls[2], ["out_pass", "out_fail"])


if __name__ == "__main__":
    unittest.main()
