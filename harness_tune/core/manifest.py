"""Harness Manifest and typed editable surfaces."""

from __future__ import annotations
import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelProfile:
    provider: str
    model_name: str
    tier: str = "general"
    context_window: int = 128000
    temperature: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelProfile:
        return cls(
            provider=data.get("provider", "unknown"),
            model_name=data.get("model_name", "unknown"),
            tier=data.get("tier", "general"),
            context_window=data.get("context_window", 128000),
            temperature=data.get("temperature", 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "tier": self.tier,
            "context_window": self.context_window,
            "temperature": self.temperature,
        }


@dataclass
class HarnessManifest:
    harness_id: str
    version: str
    model_profile: ModelProfile
    surfaces: Dict[str, Any]
    immutable_boundaries: List[str] = field(
        default_factory=lambda: ["secrets/**", "auth/**", "billing/**", "sandbox/**"]
    )
    base_dir: Optional[Path] = None

    ALLOWED_SURFACES = {
        "system_prompt",
        "skills",
        "tool_policy",
        "runtime_policy",
        "memory_policy",
        "verification_policy",
    }

    @classmethod
    def load(cls, manifest_path: Path | str) -> HarnessManifest:
        path = Path(manifest_path).resolve()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        profile_data = data.get("model_profile", {})
        profile = ModelProfile.from_dict(profile_data)

        surfaces = data.get("surfaces", {})
        for surface_name in surfaces:
            if surface_name not in cls.ALLOWED_SURFACES:
                raise ValueError(
                    f"Invalid surface '{surface_name}'. Allowed: {sorted(cls.ALLOWED_SURFACES)}"
                )

        return cls(
            harness_id=data["harness_id"],
            version=data.get("version", "1.0.0"),
            model_profile=profile,
            surfaces=surfaces,
            immutable_boundaries=data.get(
                "immutable_boundaries",
                ["secrets/**", "auth/**", "billing/**", "sandbox/**"],
            ),
            base_dir=path.parent,
        )

    def is_protected(self, file_path: str) -> bool:
        normalized = file_path.replace("\\", "/").lstrip("./")
        for pattern in self.immutable_boundaries:
            if fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(normalized, f"*/{pattern}"):
                return True
        return False

    def save(self, target_path: Path | str) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "harness_id": self.harness_id,
            "version": self.version,
            "model_profile": self.model_profile.to_dict(),
            "surfaces": self.surfaces,
            "immutable_boundaries": self.immutable_boundaries,
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
