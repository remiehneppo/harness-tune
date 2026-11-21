"""End-to-end unit test for HarnessEvolutionEngine with PiCodingAgent."""

import unittest
import tempfile
import json
import yaml
from pathlib import Path

from harness_tune.core.manifest import HarnessManifest, ModelProfile
from harness_tune.core.proposer import DedicatedProposer
from harness_tune.agents.pi_agent import PiCodingAgent
from harness_tune.benchmarks.terminal_bench import TerminalBenchAdapter
from harness_tune.benchmarks.base import BenchmarkTask
from harness_tune.engine import HarnessEvolutionEngine


class TestHarnessEvolutionEngine(unittest.TestCase):
    def test_end_to_end_optimization_cycle_perfect_baseline(self):
        with tempfile.TemporaryDirectory() as harness_dir, tempfile.TemporaryDirectory() as out_dir:
            h_dir = Path(harness_dir)
            (h_dir / "configs").mkdir()
            (h_dir / "prompts").mkdir()

            (h_dir / "configs" / "runtime.yaml").write_text("max_steps: 10\n", encoding="utf-8")
            (h_dir / "configs" / "tools.yaml").write_text("granularity: minimal\n", encoding="utf-8")
            (h_dir / "configs" / "verification.yaml").write_text("verifier_type: deterministic\n", encoding="utf-8")
            (h_dir / "prompts" / "system.md").write_text("Initial prompt\n", encoding="utf-8")

            manifest = HarnessManifest(
                harness_id="pi_agent_test",
                version="1.0.0",
                model_profile=ModelProfile("anthropic", "claude-3-5-sonnet"),
                surfaces={
                    "system_prompt": "prompts/system.md",
                    "runtime_policy": "configs/runtime.yaml",
                    "tool_policy": "configs/tools.yaml",
                    "verification_policy": "configs/verification.yaml",
                },
                immutable_boundaries=["secrets/**"],
                base_dir=h_dir,
            )
            manifest.save(h_dir / "configs" / "harness.manifest.json")

            tasks = [
                BenchmarkTask("task_1", "tb", "Solve task 1", verification_script="test -f solution.py"),
                BenchmarkTask("task_2", "tb", "Solve task 2", verification_script="test -f solution.py"),
                BenchmarkTask("task_3", "tb", "Solve task 3", verification_script="test -f solution.py"),
                BenchmarkTask("task_4", "tb", "Solve task 4", verification_script="test -f solution.py"),
            ]
            benchmark = TerminalBenchAdapter(custom_tasks=tasks)
            agent = PiCodingAgent(mode="simulated")
            proposer = DedicatedProposer(proposer_model_name="claude-3-7-sonnet")

            engine = HarnessEvolutionEngine(
                agent=agent,
                benchmark=benchmark,
                proposer=proposer,
                output_dir=out_dir,
            )

            result = engine.run_optimization_cycle(baseline_manifest=manifest, max_proposals=2)
            self.assertEqual(result["status"], "perfect_baseline")

    def test_end_to_end_optimization_cycle_with_failure_and_promotion(self):
        with tempfile.TemporaryDirectory() as harness_dir, tempfile.TemporaryDirectory() as out_dir:
            h_dir = Path(harness_dir)
            (h_dir / "configs").mkdir()
            (h_dir / "prompts").mkdir()

            (h_dir / "configs" / "runtime.yaml").write_text("max_steps: 10\nmax_consecutive_same_tool_errors: 5\n", encoding="utf-8")
            (h_dir / "configs" / "tools.yaml").write_text("granularity: minimal\n", encoding="utf-8")
            (h_dir / "configs" / "verification.yaml").write_text("verifier_type: deterministic\n", encoding="utf-8")
            # Baseline prompt lacks environment discipline -> causes task failure on dependency tasks
            (h_dir / "prompts" / "system.md").write_text("Minimal system prompt.\n", encoding="utf-8")

            manifest = HarnessManifest(
                harness_id="pi_agent_failing_baseline",
                version="1.0.0",
                model_profile=ModelProfile("anthropic", "claude-3-5-sonnet"),
                surfaces={
                    "system_prompt": "prompts/system.md",
                    "runtime_policy": "configs/runtime.yaml",
                    "tool_policy": "configs/tools.yaml",
                    "verification_policy": "configs/verification.yaml",
                },
                immutable_boundaries=["secrets/**"],
                base_dir=h_dir,
            )
            manifest.save(h_dir / "configs" / "harness.manifest.json")

            # Task requiring dependency checking: fails without environment discipline
            tasks = [
                BenchmarkTask("task_dep_1", "tb", "Solve task with dependency checks", verification_script="test -f solution.py"),
                BenchmarkTask("task_norm_2", "tb", "Solve normal task", verification_script="test -f solution.py"),
                BenchmarkTask("task_dep_3", "tb", "Solve task with dependency checks", verification_script="test -f solution.py"),
                BenchmarkTask("task_norm_4", "tb", "Solve normal task", verification_script="test -f solution.py"),
            ]
            benchmark = TerminalBenchAdapter(custom_tasks=tasks)
            agent = PiCodingAgent(mode="simulated")
            proposer = DedicatedProposer(proposer_model_name="claude-3-7-sonnet")

            engine = HarnessEvolutionEngine(
                agent=agent,
                benchmark=benchmark,
                proposer=proposer,
                output_dir=out_dir,
            )

            result = engine.run_optimization_cycle(baseline_manifest=manifest, max_proposals=2)
            self.assertEqual(result["status"], "promoted")
            self.assertTrue(Path(result["promoted_dir"]).exists())

            # Verify original baseline harness remained untouched
            baseline_prompt = (h_dir / "prompts" / "system.md").read_text()
            self.assertEqual(baseline_prompt, "Minimal system prompt.\n")

            # Verify promoted harness in separate directory received the new rules
            promoted_prompt = (Path(result["promoted_dir"]) / "prompts" / "system.md").read_text()
            self.assertIn("Environment Discipline", promoted_prompt)


if __name__ == "__main__":
    unittest.main()
