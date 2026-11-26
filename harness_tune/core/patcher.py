"""Isolated Patcher: Materialize candidate harnesses into separate directories without touching baseline."""

from __future__ import annotations
import fnmatch
import re
import shutil
import yaml
from pathlib import Path
from typing import List, Optional
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

        shutil.copytree(
            source_dir,
            candidate_dir,
            ignore=shutil.ignore_patterns("__pycache__", ".git", ".pi"),
        )

        # 2. Verify and apply each patch
        for patch in candidate.patches:
            self._apply_patch(candidate_dir, baseline_manifest, patch)

        # 3. Save new candidate manifest in the isolated directory
        if (candidate_dir / "configs").exists():
            candidate_manifest_path = candidate_dir / "configs" / "harness.manifest.json"
            manifest_rel_path = "configs/harness.manifest.json"
        else:
            candidate_manifest_path = candidate_dir / "harness.manifest.json"
            manifest_rel_path = "harness.manifest.json"

        updated_manifest = HarnessManifest(
            harness_id=f"{baseline_manifest.harness_id}_{candidate.candidate_id}",
            version=f"{baseline_manifest.version}+{candidate.candidate_id}",
            model_profile=baseline_manifest.model_profile,
            surfaces=baseline_manifest.surfaces,
            immutable_boundaries=baseline_manifest.immutable_boundaries,
            base_dir=candidate_dir,
            manifest_rel_path=manifest_rel_path,
        )
        updated_manifest.save(candidate_manifest_path)
        return updated_manifest

    @staticmethod
    def _normalize_rel(rel_path: str) -> str:
        parts = [p for p in rel_path.replace("\\", "/").split("/") if p and p != "."]
        return "/".join(parts)

    def _is_declared_surface_target(
        self, manifest: HarnessManifest, surface: str, rel_str: str
    ) -> bool:
        norm_target = self._normalize_rel(rel_str)
        surface_val = manifest.surfaces.get(surface)
        if surface_val is None:
            return False

        allowed_entries: List[str] = []
        if isinstance(surface_val, str):
            allowed_entries.append(surface_val)
        elif isinstance(surface_val, list):
            allowed_entries.extend(str(item) for item in surface_val)
        elif isinstance(surface_val, dict):
            if "path" in surface_val:
                allowed_entries.append(str(surface_val["path"]))
            if "repo_dir" in surface_val:
                allowed_entries.append(str(surface_val["repo_dir"]))

        for entry in allowed_entries:
            norm_entry = self._normalize_rel(entry)
            if norm_target == norm_entry:
                return True
            if surface == "skills":
                if norm_target.startswith(norm_entry.rstrip("/") + "/"):
                    return True
                if fnmatch.fnmatch(norm_target, norm_entry):
                    return True
        return False

    def _apply_patch(
        self,
        root_dir: Path,
        manifest: HarnessManifest,
        patch: DeclarativePatch,
    ) -> None:
        rel_str = str(patch.target_file).replace("\\", "/")

        # Security check: Never edit immutable boundaries or path traversal
        if manifest.is_protected(rel_str):
            raise PermissionError(
                f"Patch attempted to modify immutable boundary '{rel_str}'! Denied."
            )

        if patch.surface not in manifest.ALLOWED_SURFACES:
            raise ValueError(
                f"Invalid surface '{patch.surface}'. Allowed: {sorted(manifest.ALLOWED_SURFACES)}"
            )

        root_resolved = root_dir.resolve()
        target_path = (root_resolved / rel_str).resolve()
        try:
            target_path.relative_to(root_resolved)
        except ValueError as exc:
            raise PermissionError(
                f"Patch target '{rel_str}' escapes harness directory '{root_resolved}'! Denied."
            ) from exc

        if not self._is_declared_surface_target(manifest, patch.surface, rel_str):
            raise PermissionError(
                f"Patch target '{rel_str}' is not declared under surface '{patch.surface}'! Denied."
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
        if patch.operation == "replace_content":
            if isinstance(patch.value, (dict, list)):
                with open(path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(patch.value, f, default_flow_style=False, sort_keys=False)
            else:
                parsed = yaml.safe_load(str(patch.value))
                if not isinstance(parsed, dict):
                    raise ValueError("replace_content on YAML must produce a mapping")
                with open(path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(parsed, f, default_flow_style=False, sort_keys=False)
            return

        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content) or {}

        keys = [k for k in patch.path.strip("/").split("/") if k]
        if not keys:
            raise ValueError("YAML patch requires non-empty path for replace_key")
        curr = data
        for k in keys[:-1]:
            if k not in curr or not isinstance(curr[k], dict):
                curr[k] = {}
            curr = curr[k]

        curr[keys[-1]] = patch.value

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _find_markdown_section_span(lines: List[str], target_heading: str) -> tuple[int, int, int]:
        """Locate (start_idx, end_idx, heading_level) for a Markdown heading matching target_heading."""
        needle = re.sub(r"^#+\s*", "", target_heading).strip().lower()
        heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
        headings: List[tuple[int, int, str]] = []
        for idx, line in enumerate(lines):
            m = heading_re.match(line)
            if m:
                headings.append((idx, len(m.group(1)), m.group(2).strip()))

        match_pos: Optional[int] = None
        for pos, (_, _, title) in enumerate(headings):
            if title.lower() == needle:
                match_pos = pos
                break
        if match_pos is None and needle:
            for pos, (_, _, title) in enumerate(headings):
                if needle in title.lower() or title.lower() in needle:
                    match_pos = pos
                    break
        if match_pos is None:
            return -1, -1, 2

        start_idx, level, _ = headings[match_pos]
        end_idx = len(lines)
        for next_idx, next_level, _ in headings[match_pos + 1 :]:
            if next_level <= level:
                end_idx = next_idx
                break
        return start_idx, end_idx, level

    def _patch_markdown(self, path: Path, patch: DeclarativePatch) -> None:
        content = path.read_text(encoding="utf-8")
        if patch.operation == "append_rule":
            header = patch.path.strip() or "Rule"
            addition = f"\n\n### {header}\n- {patch.value}\n"
            content += addition
        elif patch.operation == "append_content":
            addition = str(patch.value)
            if content and not content.endswith("\n"):
                content += "\n"
            content += "\n" + addition.lstrip("\n") + "\n"
        elif patch.operation == "replace_content":
            content = str(patch.value)
        elif patch.operation == "replace_section":
            lines = content.splitlines()
            start_idx, end_idx, level = self._find_markdown_section_span(lines, patch.path)
            raw_val = str(patch.value).strip()
            clean_title = patch.path.lstrip("#").strip() or "Section"
            if raw_val.startswith("#"):
                new_block = raw_val.splitlines()
            else:
                new_block = [f"{'#' * level} {clean_title}", raw_val]
            if start_idx == -1:
                if content and not content.endswith("\n"):
                    content += "\n"
                content += "\n" + "\n".join(new_block) + "\n"
            else:
                updated = lines[:start_idx] + new_block + ([""] if end_idx < len(lines) else []) + lines[end_idx:]
                content = "\n".join(updated).rstrip() + "\n"
        elif patch.operation == "delete_section":
            lines = content.splitlines()
            start_idx, end_idx, _ = self._find_markdown_section_span(lines, patch.path)
            if start_idx != -1:
                updated = lines[:start_idx] + lines[end_idx:]
                content = "\n".join(updated).strip() + ("\n" if updated else "")
        else:
            raise ValueError(f"Unsupported markdown operation: {patch.operation}")
        path.write_text(content, encoding="utf-8")

    def _patch_text(self, path: Path, patch: DeclarativePatch) -> None:
        if patch.operation == "replace_content":
            path.write_text(str(patch.value), encoding="utf-8")
        else:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"\n{patch.value}\n")
