"""Unit tests for deterministic fault localization and surgical section patching."""

import tempfile
import unittest
from pathlib import Path

from harness_tune.core.failure_miner import FailureMiner
from harness_tune.core.fault_localizer import (
    compute_differential_attribution,
    extract_verifier_signal,
    localize_suspects,
    slice_failure_window,
)
from harness_tune.core.manifest import HarnessManifest, ModelProfile
from harness_tune.core.patcher import IsolatedPatcher
from harness_tune.core.proposer import DeclarativePatch, HarnessCandidate
from harness_tune.core.trace import ExecutionTrace, StepTrace, ToolCallRecord


class TestFaultLocalizer(unittest.TestCase):
    def test_extract_verifier_signal_strips_install_noise(self):
        raw_out = (
            "debconf: delaying package configuration\n"
            "Get:1 http://archive.ubuntu.com noble InRelease\n"
            "downloading uv 0.9.5 x86_64-unknown-linux-gnu\n"
            "Installed 6 packages in 18ms\n"
            "E       AssertionError: assert 0 == 2\n"
            "FAILED ../tests/test_outputs.py::test_tasks_cancel - AssertionError: assert 0 == 2\n"
        )
        sig = extract_verifier_signal(raw_out)
        self.assertNotIn("debconf", sig)
        self.assertNotIn("Installed 6 packages", sig)
        self.assertIn("FAILED ../tests/test_outputs.py::test_tasks_cancel", sig)
        self.assertIn("AssertionError: assert 0 == 2", sig)

    def test_slice_failure_window_captures_error_and_tail(self):
        steps = [
            StepTrace(
                step_number=i,
                tool_calls=[
                    ToolCallRecord(
                        "bash",
                        {"command": f"cmd_{i}"},
                        "command not found" if i == 12 else "ok",
                        "error" if i == 12 else "success",
                    )
                ],
                tokens_used=1000,
            )
            for i in range(1, 21)
        ]
        trace = ExecutionTrace("t_long", "h1", "arex-2", passed=False, steps=steps)
        sliced = slice_failure_window(trace)
        nums = [s.step_number for s in sliced]
        self.assertLessEqual(len(nums), 6)
        self.assertIn(12, nums)
        self.assertIn(20, nums)

    def test_localize_suspects_isolates_sections_and_omits_passed_raw_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            h_dir = Path(tmp)
            (h_dir / "prompts").mkdir()
            (h_dir / "configs").mkdir()
            (h_dir / "prompts" / "system.md").write_text(
                "## Open-Source First & Prior-Art Protocol\nSearch web first.\n\n"
                "## Task Tracking & Memory (Beads)\nUse bd CLI.\n\n"
                "## Codebase Navigation & Architecture Exploration Protocol\nUse codegraph_* tools.\n",
                encoding="utf-8",
            )
            (h_dir / "configs" / "runtime.yaml").write_text("thinking_level: high\nmax_steps: 0\n", encoding="utf-8")
            (h_dir / "configs" / "tools.yaml").write_text("use_default_tools: true\n", encoding="utf-8")
            (h_dir / "configs" / "verification.yaml").write_text("verifier_type: deterministic\n", encoding="utf-8")

            manifest = HarnessManifest(
                harness_id="pi-arex2-user-default",
                version="1.0.0",
                model_profile=ModelProfile("local", "arex-2", thinking_level="high"),
                surfaces={
                    "system_prompt": "prompts/system.md",
                    "runtime_policy": "configs/runtime.yaml",
                    "tool_policy": "configs/tools.yaml",
                    "verification_policy": "configs/verification.yaml",
                },
                base_dir=h_dir,
            )

            pass_trace = ExecutionTrace(
                task_id="pass_task_1",
                harness_id="pi-arex2-user-default",
                model_name="arex-2",
                passed=True,
                steps=[
                    StepTrace(1, tool_calls=[ToolCallRecord("read", {"path": "a.py"}, "ok", "success")], tokens_used=5000),
                    StepTrace(2, tool_calls=[ToolCallRecord("write", {"path": "a.py"}, "ok", "success")], tokens_used=5000),
                ],
                total_tokens=10000,
            )
            fail_trace = ExecutionTrace(
                task_id="fail_task_1",
                harness_id="pi-arex2-user-default",
                model_name="arex-2",
                passed=False,
                steps=[StepTrace(1, thought="Thinking deeply", tool_calls=[], tokens_used=23000)],
                total_tokens=23000,
                error_summary="Pi agent execution timed out after 1200s",
            )

            traces = [pass_trace, fail_trace]
            evidence = FailureMiner().analyze(traces)
            diff = compute_differential_attribution(traces)
            self.assertEqual(diff.zero_tool_call_failures, 1)
            self.assertEqual(diff.pass_avg_first_edit_step, 2.0)

            loc = localize_suspects(manifest, evidence, traces)
            self.assertIn("runtime_policy", loc.suspect_surfaces)
            self.assertIn("system_prompt", loc.suspect_surfaces)
            self.assertIn("Codebase Navigation & Architecture Exploration Protocol", loc.suspect_sections)
            self.assertIn("Task Tracking & Memory (Beads)", loc.suspect_sections)
            self.assertIn("fail_task_1", loc.compact_prompt_block)
            self.assertNotIn("write [success]", loc.compact_prompt_block)

    def test_patcher_replace_and_delete_markdown_section(self):
        with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as out_dir:
            src = Path(src_dir)
            (src / "prompts").mkdir()
            (src / "prompts" / "system.md").write_text(
                "## Keep Section\nKeep body.\n\n"
                "## Replace Section\nOld body.\n\n"
                "## Delete Section\nRemove me.\n",
                encoding="utf-8",
            )
            manifest = HarnessManifest(
                harness_id="test_sec",
                version="1.0.0",
                model_profile=ModelProfile("local", "arex-2"),
                surfaces={"system_prompt": "prompts/system.md"},
                base_dir=src,
            )
            cand = HarnessCandidate(
                candidate_id="cand_sec",
                target_model="arex-2",
                hypothesis="Surgical section update",
                evidence_cluster="FC_TIMEOUT",
                patches=[
                    DeclarativePatch("system_prompt", "prompts/system.md", "replace_section", "Replace Section", "New body."),
                    DeclarativePatch("system_prompt", "prompts/system.md", "delete_section", "Delete Section", ""),
                ],
            )
            updated = IsolatedPatcher().materialize(manifest, cand, out_dir)
            patched = (updated.base_dir / "prompts" / "system.md").read_text(encoding="utf-8")
            self.assertIn("## Keep Section\nKeep body.", patched)
            self.assertIn("## Replace Section\nNew body.", patched)
            self.assertNotIn("Old body.", patched)
            self.assertNotIn("Delete Section", patched)
            self.assertNotIn("Remove me.", patched)


if __name__ == "__main__":
    unittest.main()
