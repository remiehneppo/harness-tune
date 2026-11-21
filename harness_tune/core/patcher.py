"""Isolated Patcher: Materialize candidate harnesses into separate directories without touching baseline."""

from __future__ import annotations
import os
import shutil
import yaml
from pathlib import Path
from typing import Any, Dict
from harness_tune.core.manifest import HarnessManifest
from harness_tune.core.proposer import HarnessCandidate, DeclarativePatch


class IsolatedPatcher:
    """Clones baseline harness into a dedicated isolated directory and applies bounded patches."""

    def materialize(
        self,
        baseline_manifest: HarnessManifest,
        candidate: HarnessCandidate,
        output_base_dir: Path | str,
    ) -> HarnessManifest:
        output_base = Path(output_base_dir).resolve()
        candidate_dir = output_base / f"{baseline_manifest.harness_id}_{candidate.candidate_id}"
        
        # 1. Clean and duplicate baseline directory
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)
        
        source_dir = baseline_manifest.base_dir
        if not source_dir or not source_dir.exists():
            raise FileNotFoundError(f"Baseline harness directory not found: {source_dir}")

        shutil.copytree(source_dir, candidate_dir)

        # 2. Verify and apply each patch
        for patch in candidate.patches:
            self._apply_patch(candidate_dir, baseline_manifest, patch)

        # 3. Save new candidate manifest in the isolated directory
        candidate_manifest_path = candidate_dir / "configs" / "harness.manifest.json"
        if not candidate_manifest_path.parent.exists():
            candidate_manifest_path = candidate_dir / "harness.manifest.json"

        updated_manifest = HarnessManifest(
            harness_id=f"{baseline_manifest.harness_id}_{candidate.candidate_id}",
            version=f"{baseline_manifest.version}+{candidate.candidate_id}",
            model_profile=baseline_manifest.model_profile,
            surfaces=baseline_manifest.surfaces,
            immutable_boundaries=baseline_manifest.immutable_boundaries,
            base_dir=candidate_dir,
        )
        updated_manifest.save(candidate_manifest_path)
        return updated_manifest

    def _apply_patch(
        self,
        root_dir: Path,
        manifest: HarnessManifest,
        patch: DeclarativePatch,
    ) -> None:
        target_path = root_dir / patch.target_file
        rel_str = str(patch.target_file).replace("\\", "/")

        # Security check: Never edit immutable boundaries
        if manifest.is_protected(rel_str):
            raise PermissionError(
                f"Patch attempted to modify immutable boundary '{rel_str}'! Denied."
            )

        if not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.suffix in [".yaml", ".yml"]:
                target_path.write_text("{}\n", encoding="utf-8")
            else:
                target_path.write_text("", encoding="utf-8")

        # Apply based on file extension
        if target_path.suffix in [".yaml", ".yml"]:
            self._patch_yaml(target_path, patch)
        elif target_path.suffix == ".md":
            self._patch_markdown(target_path, patch)
        else:
            self._patch_text(target_path, patch)

    def _patch_yaml(self, path: Path, patch: DeclarativePatch) -> None:
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content) or {}

        keys = patch.path.strip("/").split("/")
        curr = data
        for k in keys[:-1]:
            if k not in curr or not isinstance(curr[k], dict):
                curr[k] = {}
            curr = curr[k]

        curr[keys[-1]] = patch.value

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)

    def _patch_markdown(self, path: Path, patch: DeclarativePatch) -> None:
        content = path.read_text(encoding="utf-8")
        if patch.operation == "append_rule":
            addition = f"\n\n### {patch.path}\n- {patch.value}\n"
            content += addition
        elif patch.operation == "replace_content":
            content = str(patch.value)
        path.write_text(content, encoding="utf-8")

    def _patch_text(self, path: Path, patch: DeclarativePatch) -> None:
        if patch.operation == "replace_content":
            path.write_text(str(patch.value), encoding="utf-8")
        else:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"\n{patch.value}\n")
