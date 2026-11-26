"""Unit tests for IsolatedPatcher."""

import tempfile
import unittest
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

            (src / "configs").mkdir()
            (src / "configs" / "runtime.yaml").write_text("max_steps: 10\n", encoding="utf-8")
            (src / "prompts").mkdir()
            (src / "prompts" / "system.md").write_text("# Baseline System\n", encoding="utf-8")

            manifest = HarnessManifest(
                harness_id="test_harness",
                version="1.0.0",
                model_profile=ModelProfile("local", "arex-2"),
                surfaces={
                    "runtime_policy": "configs/runtime.yaml",
                    "system_prompt": "prompts/system.md",
                },
                immutable_boundaries=["secrets/**"],
                base_dir=src,
            )
            manifest.save(src / "configs" / "harness.manifest.json")

            loaded_manifest = HarnessManifest.load(src / "configs" / "harness.manifest.json")
            self.assertEqual(loaded_manifest.base_dir, src.resolve())

            cand = HarnessCandidate(
                candidate_id="cand_1",
                target_model="arex-2",
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
            new_manifest = patcher.materialize(loaded_manifest, cand, out)

            # Verify baseline was NOT modified
            original_runtime = yaml.safe_load((src / "configs" / "runtime.yaml").read_text())
            self.assertEqual(original_runtime["max_steps"], 10)
            self.assertEqual((src / "prompts" / "system.md").read_text(), "# Baseline System\n")

            # Verify candidate in separate dir has baseline + applied changes
            patched_runtime = yaml.safe_load((new_manifest.base_dir / "configs" / "runtime.yaml").read_text())
            self.assertEqual(patched_runtime["max_steps"], 25)

            patched_prompt = (new_manifest.base_dir / "prompts" / "system.md").read_text()
            self.assertIn("# Baseline System", patched_prompt)
            self.assertIn("Do not repeat errors.", patched_prompt)

    def test_deny_mutation_on_protected_boundary_and_traversal_and_undeclared(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as out_dir:
            src = Path(source_dir)
            out = Path(out_dir)
            (src / "secrets").mkdir()
            (src / "secrets" / "key.txt").write_text("supersecret", encoding="utf-8")
            (src / "configs").mkdir()
            (src / "configs" / "runtime.yaml").write_text("max_steps: 10\n", encoding="utf-8")

            manifest = HarnessManifest(
                harness_id="test_harness",
                version="1.0.0",
                model_profile=ModelProfile("local", "arex-2"),
                surfaces={"runtime_policy": "configs/runtime.yaml"},
                immutable_boundaries=["secrets/**"],
                base_dir=src,
            )

            patcher = IsolatedPatcher()

            # Protected boundary
            cand_secret = HarnessCandidate(
                candidate_id="cand_illegal",
                target_model="arex-2",
                hypothesis="Try to modify secrets",
                evidence_cluster="FC_EVIL",
                patches=[
                    DeclarativePatch(
                        surface="runtime_policy",
                        target_file="secrets/key.txt",
                        operation="replace_content",
                        path="",
                        value="exposed",
                    )
                ],
            )
            with self.assertRaises(PermissionError):
                patcher.materialize(manifest, cand_secret, out)

            # Path traversal
            cand_traversal = HarnessCandidate(
                candidate_id="cand_traversal",
                target_model="arex-2",
                hypothesis="Try traversal",
                evidence_cluster="FC_EVIL",
                patches=[
                    DeclarativePatch(
                        surface="runtime_policy",
                        target_file="../escape.yaml",
                        operation="replace_content",
                        path="",
                        value="hack: true",
                    )
                ],
            )
            with self.assertRaises(PermissionError):
                patcher.materialize(manifest, cand_traversal, out)

            # Undeclared file under allowed surface
            cand_undeclared = HarnessCandidate(
                candidate_id="cand_undeclared",
                target_model="arex-2",
                hypothesis="Try editing undeclared file",
                evidence_cluster="FC_EVIL",
                patches=[
                    DeclarativePatch(
                        surface="runtime_policy",
                        target_file="configs/other.yaml",
                        operation="replace_key",
                        path="x",
                        value=1,
                    )
                ],
            )
            with self.assertRaises(PermissionError):
                patcher.materialize(manifest, cand_undeclared, out)


if __name__ == "__main__":
    unittest.main()
