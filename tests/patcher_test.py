"""Unit tests for IsolatedPatcher."""

import unittest
import tempfile
import yaml
from pathlib import Path
from harness_tune.core.manifest import HarnessManifest, ModelProfile
from harness_tune.core.proposer import HarnessCandidate, DeclarativePatch
from harness_tune.core.patcher import IsolatedPatcher


class TestIsolatedPatcher(unittest.TestCase):
    def test_materialize_in_separate_directory(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as out_dir:
            src = Path(source_dir)
            out = Path(out_dir)

            # Setup baseline harness
            (src / "configs").mkdir()
            (src / "configs" / "runtime.yaml").write_text("max_steps: 10\n", encoding="utf-8")
            (src / "prompts").mkdir()
            (src / "prompts" / "system.md").write_text("# Baseline System\n", encoding="utf-8")

            manifest = HarnessManifest(
                harness_id="test_harness",
                version="1.0.0",
                model_profile=ModelProfile("anthropic", "claude-3-5-sonnet"),
                surfaces={
                    "runtime_policy": "configs/runtime.yaml",
                    "system_prompt": "prompts/system.md",
                },
                immutable_boundaries=["secrets/**"],
                base_dir=src,
            )

            cand = HarnessCandidate(
                candidate_id="cand_1",
                target_model="claude-3-5-sonnet",
                hypothesis="Improve loop breaking",
                evidence_cluster="FC_TOOL_LOOP",
                patches=[
                    DeclarativePatch(
                        surface="runtime_policy",
                        target_file="configs/runtime.yaml",
                        operation="replace_key",
                        path="max_steps",
                        value=25,
                    ),
                    DeclarativePatch(
                        surface="system_prompt",
                        target_file="prompts/system.md",
                        operation="append_rule",
                        path="Rule 1",
                        value="Do not repeat errors.",
                    ),
                ],
            )

            patcher = IsolatedPatcher()
            new_manifest = patcher.materialize(manifest, cand, out)

            # Verify baseline was NOT modified
            original_runtime = yaml.safe_load((src / "configs" / "runtime.yaml").read_text())
            self.assertEqual(original_runtime["max_steps"], 10)

            # Verify candidate in separate dir has applied changes
            patched_runtime = yaml.safe_load((new_manifest.base_dir / "configs" / "runtime.yaml").read_text())
            self.assertEqual(patched_runtime["max_steps"], 25)

            patched_prompt = (new_manifest.base_dir / "prompts" / "system.md").read_text()
            self.assertIn("Do not repeat errors.", patched_prompt)

    def test_deny_mutation_on_protected_boundary(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as out_dir:
            src = Path(source_dir)
            out = Path(out_dir)
            (src / "secrets").mkdir()
            (src / "secrets" / "key.txt").write_text("supersecret", encoding="utf-8")

            manifest = HarnessManifest(
                harness_id="test_harness",
                version="1.0.0",
                model_profile=ModelProfile("anthropic", "claude-3-5-sonnet"),
                surfaces={"runtime_policy": "configs/runtime.yaml"},
                immutable_boundaries=["secrets/**"],
                base_dir=src,
            )

            cand = HarnessCandidate(
                candidate_id="cand_illegal",
                target_model="claude-3-5-sonnet",
                hypothesis="Try to modify secrets",
                evidence_cluster="FC_EVIL",
                patches=[
                    DeclarativePatch(
                        surface="secret",
                        target_file="secrets/key.txt",
                        operation="replace_content",
                        path="",
                        value="exposed",
                    )
                ],
            )

            patcher = IsolatedPatcher()
            with self.assertRaises(PermissionError):
                patcher.materialize(manifest, cand, out)


if __name__ == "__main__":
    unittest.main()
