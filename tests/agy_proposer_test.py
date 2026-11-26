"""Unit tests for AgyCliClient and LLM-backed DedicatedProposer."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness_tune.core.agy_client import AgyCliClient, normalize_agy_model
from harness_tune.core.failure_miner import FailureMiner
from harness_tune.core.manifest import HarnessManifest, ModelProfile
from harness_tune.core.proposer import DedicatedProposer
from harness_tune.core.trace import ExecutionTrace, StepTrace, ToolCallRecord


class TestAgyClientAndProposer(unittest.TestCase):
    def test_normalize_agy_model(self):
        self.assertEqual(normalize_agy_model("gemini-3.8-flash"), "gemini-3.8-flash-high")
        self.assertEqual(normalize_agy_model("gemini-flash-3.8"), "gemini-3.8-flash-high")
        self.assertEqual(normalize_agy_model("gemini-3.8-flash-medium"), "gemini-3.8-flash-medium")
        self.assertEqual(normalize_agy_model("gemini-3.8-flash-low"), "gemini-3.8-flash-low")

    def test_agy_client_invocation_and_envelope_parse(self):
        sample_stdout = json.dumps(
            {
                "conversation_id": "conv-123",
                "status": "SUCCESS",
                "response": '{"diagnosis":"ok","candidates":[]}',
                "duration_seconds": 1.5,
                "usage": {"total_tokens": 120},
            }
        )
        fake_proc = subprocess.CompletedProcess(
            args=["agy"], returncode=0, stdout=sample_stdout, stderr=""
        )

        with patch("harness_tune.core.agy_client.shutil.which", return_value="/usr/local/bin/agy"), patch(
            "harness_tune.core.agy_client.subprocess.run", return_value=fake_proc
        ) as mock_run:
            client = AgyCliClient(default_model="gemini-3.8-flash")
            res = client.generate("Hello prompt")

            self.assertEqual(res.status, "SUCCESS")
            self.assertEqual(res.model, "gemini-3.8-flash-high")
            self.assertEqual(res.text, '{"diagnosis":"ok","candidates":[]}')

            called_cmd = mock_run.call_args[0][0]
            self.assertIn("--model", called_cmd)
            self.assertIn("gemini-3.8-flash-high", called_cmd)
            self.assertIn("-p=Hello prompt", called_cmd)
            self.assertNotIn("--json-schema", called_cmd)
            self.assertEqual(mock_run.call_args.kwargs["stdin"], subprocess.DEVNULL)

    def test_proposer_builds_held_in_prompt_only_and_parses_json_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            h_dir = Path(tmp)
            (h_dir / "prompts").mkdir()
            (h_dir / "configs").mkdir()
            (h_dir / "prompts" / "system.md").write_text("# Base Prompt\n", encoding="utf-8")
            (h_dir / "configs" / "runtime.yaml").write_text("max_steps: 25\n", encoding="utf-8")

            manifest = HarnessManifest(
                harness_id="pi-arex2-baseline",
                version="1.0.0",
                model_profile=ModelProfile("local", "arex-2", thinking_level="medium"),
                surfaces={
                    "system_prompt": "prompts/system.md",
                    "runtime_policy": "configs/runtime.yaml",
                },
                base_dir=h_dir,
            )

            held_in_trace = ExecutionTrace(
                task_id="tb_001_log_analyzer",
                harness_id="pi-arex2-baseline",
                model_name="arex-2",
                passed=False,
                steps=[
                    StepTrace(
                        step_number=1,
                        thought="Let me write script",
                        tool_calls=[
                            ToolCallRecord("bash", {"command": "cat access.log"}, "ok", "success")
                        ],
                        tokens_used=400,
                    )
                ],
                total_tokens=400,
                error_summary="Verifier failed: expected count=4, got count=5",
                verifier_output="AssertionError: expected count=4",
            )
            evidence = FailureMiner().analyze([held_in_trace])

            captured = {}

            def fake_llm(model_name: str, prompt: str) -> str:
                captured["model"] = model_name
                captured["prompt"] = prompt
                return (
                    "```json\n"
                    + json.dumps(
                        {
                            "diagnosis": "Agent matched URL path /500 as status 500.",
                            "candidates": [
                                {
                                    "candidate_id": "cand_arex2_1",
                                    "target_model": "arex-2",
                                    "hypothesis": "Verify log field boundaries against sample input",
                                    "evidence_cluster": "FC_GENERAL_LOGIC",
                                    "patches": [
                                        {
                                            "surface": "system_prompt",
                                            "target_file": "prompts/system.md",
                                            "operation": "append_rule",
                                            "path": "Output Verification",
                                            "value": "Test scripts on provided fixtures before finishing.",
                                        }
                                    ],
                                }
                            ],
                        }
                    )
                    + "\n```"
                )

            proposer = DedicatedProposer(
                proposer_model_name="gemini-3.8-flash-high",
                llm_caller=fake_llm,
                allow_heuristic_fallback=False,
            )
            cands = proposer.propose(
                manifest=manifest,
                evidence=evidence,
                max_proposals=2,
                traces=[held_in_trace],
            )

            self.assertEqual(captured["model"], "gemini-3.8-flash-high")
            self.assertIn("tb_001_log_analyzer", captured["prompt"])
            self.assertIn("# Base Prompt", captured["prompt"])
            self.assertNotIn("tb_004_sqlite_migration", captured["prompt"])
            self.assertEqual(proposer.last_diagnosis, "Agent matched URL path /500 as status 500.")
            self.assertEqual(len(cands), 1)
            self.assertEqual(cands[0].candidate_id, "cand_arex2_1")
            self.assertEqual(cands[0].patches[0].target_file, "prompts/system.md")


if __name__ == "__main__":
    unittest.main()
