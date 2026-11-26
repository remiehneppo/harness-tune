"""Unit tests for HarnessManifest and security boundaries."""

import json
import tempfile
import unittest
from pathlib import Path
from harness_tune.core.manifest import HarnessManifest


class TestHarnessManifest(unittest.TestCase):
    def test_load_and_validate_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data = {
                "harness_id": "test-agent",
                "version": "1.0.0",
                "model_profile": {
                    "provider": "anthropic",
                    "model_name": "claude-3-5-sonnet",
                    "thinking_level": "medium",
                },
                "surfaces": {
                    "system_prompt": "prompts/system.md",
                    "runtime_policy": "configs/runtime.yaml",
                },
                "immutable_boundaries": ["secrets/**", "auth/**"],
            }
            manifest_file = tmp_path / "harness.manifest.json"
            manifest_file.write_text(json.dumps(data), encoding="utf-8")

            manifest = HarnessManifest.load(manifest_file)
            self.assertEqual(manifest.harness_id, "test-agent")
            self.assertEqual(manifest.model_profile.model_name, "claude-3-5-sonnet")
            self.assertEqual(manifest.model_profile.thinking_level, "medium")
            self.assertEqual(manifest.base_dir, tmp_path.resolve())
            self.assertTrue(manifest.is_protected("secrets/api_key.txt"))
            self.assertTrue(manifest.is_protected("auth/tokens.json"))
            self.assertTrue(manifest.is_protected("../secrets/key.txt"))
            self.assertTrue(manifest.is_protected("/etc/passwd"))
            self.assertFalse(manifest.is_protected("prompts/system.md"))

    def test_load_from_configs_subdir_resolves_harness_root(self):
        manifest_path = (
            Path(__file__).resolve().parent.parent
            / "templates"
            / "pi_agent_harness"
            / "configs"
            / "harness_arex2.manifest.json"
        )
        manifest = HarnessManifest.load(manifest_path)
        self.assertEqual(manifest.base_dir, manifest_path.parent.parent.resolve())
        self.assertEqual(manifest.manifest_rel_path, "configs/harness_arex2.manifest.json")
        self.assertTrue((manifest.base_dir / manifest.surfaces["system_prompt"]).exists())

    def test_reject_invalid_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data = {
                "harness_id": "test-agent",
                "version": "1.0.0",
                "surfaces": {
                    "arbitrary_code_injection": "hack.py",
                },
            }
            manifest_file = tmp_path / "harness.manifest.json"
            manifest_file.write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaises(ValueError):
                HarnessManifest.load(manifest_file)


if __name__ == "__main__":
    unittest.main()
