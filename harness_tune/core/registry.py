"""Harness Registry, Version Lineage, and Pareto Archive."""

from __future__ import annotations
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from harness_tune.core.manifest import HarnessManifest


@dataclass
class LineageNode:
    version: str
    harness_path: str
    parent_version: Optional[str]
    held_in_pass_rate: float
    held_out_pass_rate: float
    utility_score: float
    hypothesis: str
    promoted: bool


class HarnessRegistry:
    """Maintains optimized harness versions and Pareto specialists in isolated directories."""

    def __init__(self, registry_dir: Path | str):
        self.registry_dir = Path(registry_dir).resolve()
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.lineage_file = self.registry_dir / "lineage.json"
        self.history: List[LineageNode] = self._load_lineage()

    def _load_lineage(self) -> List[LineageNode]:
        if not self.lineage_file.exists():
            return []
        try:
            with open(self.lineage_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return [LineageNode(**item) for item in raw]
        except Exception:
            return []

    def _save_lineage(self) -> None:
        with open(self.lineage_file, "w", encoding="utf-8") as f:
            json.dump([node.__dict__ for node in self.history], f, indent=2)

    def record_run(
        self,
        manifest: HarnessManifest,
        parent_version: Optional[str],
        in_pass: float,
        ho_pass: float,
        utility: float,
        hypothesis: str,
        promoted: bool,
    ) -> LineageNode:
        node = LineageNode(
            version=manifest.version,
            harness_path=str(manifest.base_dir),
            parent_version=parent_version,
            held_in_pass_rate=in_pass,
            held_out_pass_rate=ho_pass,
            utility_score=utility,
            hypothesis=hypothesis,
            promoted=promoted,
        )
        self.history.append(node)
        self._save_lineage()
        return node

    def get_latest_promoted(self) -> Optional[LineageNode]:
        for node in reversed(self.history):
            if node.promoted:
                return node
        return None
