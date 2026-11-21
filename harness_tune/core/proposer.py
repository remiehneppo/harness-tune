"""Dedicated Proposer for generating bounded declarative harness patches."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from harness_tune.core.failure_miner import EvidenceBundle
from harness_tune.core.manifest import HarnessManifest


@dataclass
class DeclarativePatch:
    surface: str       # e.g. "runtime_policy", "tool_policy", "system_prompt"
    target_file: str   # Relative path in harness
    operation: str     # "replace_key", "append_rule", "replace_content"
    path: str          # JSON / YAML pointer or section header
    value: Any


@dataclass
class HarnessCandidate:
    candidate_id: str
    target_model: str
    hypothesis: str
    evidence_cluster: str
    patches: List[DeclarativePatch] = field(default_factory=list)
    expected_gain: str = "pass_rate"
    regression_risk: str = "none"


class DedicatedProposer:
    """Uses a dedicated, high-capacity proposer model to propose bounded harness edits."""

    def __init__(
        self,
        proposer_model_name: str = "claude-3-7-sonnet",
        llm_caller: Optional[Callable[[str, str], str]] = None,
    ):
        self.proposer_model_name = proposer_model_name
        self.llm_caller = llm_caller

    def propose(
        self,
        manifest: HarnessManifest,
        evidence: EvidenceBundle,
        max_proposals: int = 3,
    ) -> List[HarnessCandidate]:
        candidates: List[HarnessCandidate] = []

        # If custom LLM caller is provided, call external model with strict JSON prompt
        if self.llm_caller and evidence.clusters:
            prompt = self._build_prompt(manifest, evidence, max_proposals)
            try:
                response = self.llm_caller(self.proposer_model_name, prompt)
                parsed = self._parse_llm_proposals(response)
                if parsed:
                    return parsed
            except Exception:
                pass  # Fall back to heuristic rule-based proposal

        # Heuristic proposer grounded in paper findings:
        for idx, cluster in enumerate(evidence.clusters[:max_proposals]):
            cid = f"cand_{manifest.model_profile.model_name.replace('-', '_')}_{idx + 1}"

            if cluster.cluster_id == "FC_TOOL_LOOP":
                candidates.append(
                    HarnessCandidate(
                        candidate_id=cid,
                        target_model=manifest.model_profile.model_name,
                        hypothesis="Enforcing max consecutive tool errors stops infinite error loops",
                        evidence_cluster=cluster.cluster_id,
                        patches=[
                            DeclarativePatch(
                                surface="runtime_policy",
                                target_file=manifest.surfaces.get("runtime_policy", "configs/runtime.yaml"),
                                operation="replace_key",
                                path="max_consecutive_same_tool_errors",
                                value=2,
                            ),
                            DeclarativePatch(
                                surface="runtime_policy",
                                target_file=manifest.surfaces.get("runtime_policy", "configs/runtime.yaml"),
                                operation="replace_key",
                                path="early_stop_on_repeated_failure",
                                value=True,
                            ),
                        ],
                        expected_gain="eliminate_tool_loops",
                    )
                )

            elif cluster.cluster_id == "FC_MISSING_DEP":
                candidates.append(
                    HarnessCandidate(
                        candidate_id=cid,
                        target_model=manifest.model_profile.model_name,
                        hypothesis="Requiring pre-flight environment and dependency checks prevents path errors",
                        evidence_cluster=cluster.cluster_id,
                        patches=[
                            DeclarativePatch(
                                surface="tool_policy",
                                target_file=manifest.surfaces.get("tool_policy", "configs/tools.yaml"),
                                operation="replace_key",
                                path="error_handling/provide_safe_retry_hint",
                                value=True,
                            ),
                            DeclarativePatch(
                                surface="system_prompt",
                                target_file=manifest.surfaces.get("system_prompt", "prompts/system.md"),
                                operation="append_rule",
                                path="Environment Discipline",
                                value="Always verify tool dependencies and working directory before executing unfamiliar commands.",
                            ),
                        ],
                        expected_gain="reduce_command_failures",
                    )
                )

            elif cluster.cluster_id == "FC_SYNTAX_ERROR":
                candidates.append(
                    HarnessCandidate(
                        candidate_id=cid,
                        target_model=manifest.model_profile.model_name,
                        hypothesis="Mandatory deterministic verification before task completion catches syntax errors early",
                        evidence_cluster=cluster.cluster_id,
                        patches=[
                            DeclarativePatch(
                                surface="verification_policy",
                                target_file=manifest.surfaces.get("verification_policy", "configs/verification.yaml"),
                                operation="replace_key",
                                path="non_regression_margin",
                                value=0.0,
                            ),
                            DeclarativePatch(
                                surface="system_prompt",
                                target_file=manifest.surfaces.get("system_prompt", "prompts/system.md"),
                                operation="append_rule",
                                path="Self-Verification Rule",
                                value="Run compile or test checks on every edited file before claiming completion.",
                            ),
                        ],
                        expected_gain="eliminate_syntax_regressions",
                    )
                )

            elif cluster.cluster_id == "FC_STEP_EXHAUSTION":
                candidates.append(
                    HarnessCandidate(
                        candidate_id=cid,
                        target_model=manifest.model_profile.model_name,
                        hypothesis="Bounding exploration steps accelerates transition to concrete implementation",
                        evidence_cluster=cluster.cluster_id,
                        patches=[
                            DeclarativePatch(
                                surface="runtime_policy",
                                target_file=manifest.surfaces.get("runtime_policy", "configs/runtime.yaml"),
                                operation="replace_key",
                                path="max_steps",
                                value=25,
                            ),
                            DeclarativePatch(
                                surface="system_prompt",
                                target_file=manifest.surfaces.get("system_prompt", "prompts/system.md"),
                                operation="append_rule",
                                path="Exploration Boundary",
                                value="Limit exploratory reads to at most 4 steps before beginning concrete edits.",
                            ),
                        ],
                        expected_gain="faster_convergence",
                    )
                )

            elif cluster.cluster_id == "FC_GENERAL_LOGIC":
                candidates.append(
                    HarnessCandidate(
                        candidate_id=cid,
                        target_model=manifest.model_profile.model_name,
                        hypothesis="Adding explicit acceptance verification and test-driven checkpoints resolves assertion failures",
                        evidence_cluster=cluster.cluster_id,
                        patches=[
                            DeclarativePatch(
                                surface="system_prompt",
                                target_file=manifest.surfaces.get("system_prompt", "prompts/system.md"),
                                operation="append_rule",
                                path="Acceptance Verification",
                                value="Review all task success criteria, write tests to assert behavior, and verify script execution before completing.",
                            ),
                            DeclarativePatch(
                                surface="verification_policy",
                                target_file=manifest.surfaces.get("verification_policy", "configs/verification.yaml"),
                                operation="replace_key",
                                path="verifier_type",
                                value="deterministic",
                            ),
                        ],
                        expected_gain="improve_task_completion",
                    )
                )

        return candidates

    def _build_prompt(self, manifest: HarnessManifest, evidence: EvidenceBundle, max_proposals: int) -> str:
        return (
            f"You are a dedicated Harness Proposer optimizing a coding agent running model '{manifest.model_profile.model_name}'.\n"
            f"Observed Evidence:\n{evidence.summary()}\n\n"
            f"Generate up to {max_proposals} bounded declarative patches strictly within allowed surfaces:\n"
            f"{json.dumps(manifest.surfaces, indent=2)}\n"
            f"Never edit immutable boundaries: {manifest.immutable_boundaries}.\n"
            "Return JSON matching list of HarnessCandidate objects."
        )

    def _parse_llm_proposals(self, raw_text: str) -> List[HarnessCandidate]:
        try:
            start = raw_text.find("[")
            end = raw_text.rfind("]")
            if start != -1 and end != -1:
                items = json.loads(raw_text[start : end + 1])
                candidates = []
                for item in items:
                    patches = [
                        DeclarativePatch(
                            surface=p["surface"],
                            target_file=p["target_file"],
                            operation=p["operation"],
                            path=p["path"],
                            value=p["value"],
                        )
                        for p in item.get("patches", [])
                    ]
                    candidates.append(
                        HarnessCandidate(
                            candidate_id=item["candidate_id"],
                            target_model=item.get("target_model", "target"),
                            hypothesis=item.get("hypothesis", ""),
                            evidence_cluster=item.get("evidence_cluster", "FC_GEN"),
                            patches=patches,
                            expected_gain=item.get("expected_gain", "pass_rate"),
                            regression_risk=item.get("regression_risk", "none"),
                        )
                    )
                return candidates
        except Exception:
            pass
        return []
