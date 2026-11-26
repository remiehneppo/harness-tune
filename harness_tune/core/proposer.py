"""Dedicated Proposer for generating bounded declarative harness patches."""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from harness_tune.core.agy_client import DEFAULT_AGY_MODEL
from harness_tune.core.failure_miner import EvidenceBundle
from harness_tune.core.fault_localizer import localize_suspects
from harness_tune.core.manifest import HarnessManifest
from harness_tune.core.trace import ExecutionTrace


@dataclass
class DeclarativePatch:
    surface: str       # e.g. "runtime_policy", "tool_policy", "system_prompt"
    target_file: str   # Relative path in harness
    operation: str     # "replace_key", "append_rule", "replace_content", "append_content"
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
        proposer_model_name: str = DEFAULT_AGY_MODEL,
        llm_caller: Optional[Callable[[str, str], str]] = None,
        allow_heuristic_fallback: bool = True,
        use_fault_localization: bool = True,
    ):
        self.proposer_model_name = proposer_model_name
        self.llm_caller = llm_caller
        self.allow_heuristic_fallback = allow_heuristic_fallback
        self.use_fault_localization = use_fault_localization
        self.last_prompt: Optional[str] = None
        self.last_raw_response: Optional[str] = None
        self.last_diagnosis: Optional[str] = None
        self.last_localization: Optional[Dict[str, Any]] = None

    def propose(
        self,
        manifest: HarnessManifest,
        evidence: EvidenceBundle,
        max_proposals: int = 3,
        traces: Optional[List[ExecutionTrace]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[HarnessCandidate]:
        # If custom LLM caller is provided, call external orchestrator model with strict JSON prompt
        if self.llm_caller and evidence.clusters:
            prompt = self._build_prompt(
                manifest=manifest,
                evidence=evidence,
                max_proposals=max_proposals,
                traces=traces,
                history=history,
            )
            self.last_prompt = prompt
            try:
                response = self.llm_caller(self.proposer_model_name, prompt)
                self.last_raw_response = response
                parsed = self._parse_llm_proposals(
                    response,
                    default_model=manifest.model_profile.model_name,
                    max_proposals=max_proposals,
                )
                if parsed:
                    return parsed
                raise ValueError("LLM orchestrator returned no valid candidate patches")
            except Exception as ex:
                if not self.allow_heuristic_fallback:
                    raise RuntimeError(
                        f"Orchestrator proposer '{self.proposer_model_name}' failed: {ex}"
                    ) from ex
                print(
                    f"   ⚠️ Orchestrator proposer failed ({ex}); falling back to heuristic proposer."
                )

        return self._propose_heuristic(manifest, evidence, max_proposals)

    def _propose_heuristic(
        self,
        manifest: HarnessManifest,
        evidence: EvidenceBundle,
        max_proposals: int,
    ) -> List[HarnessCandidate]:
        candidates: List[HarnessCandidate] = []

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

            elif cluster.cluster_id in ("FC_STEP_EXHAUSTION", "FC_TIMEOUT"):
                candidates.append(
                    HarnessCandidate(
                        candidate_id=cid,
                        target_model=manifest.model_profile.model_name,
                        hypothesis="Bounding exploration steps and using lower thinking overhead prevents step/time exhaustion",
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

    def _read_surface_contents(self, manifest: HarnessManifest) -> str:
        if not manifest.base_dir:
            return "(no base_dir available)"

        seen_paths: List[str] = []
        for surface_name, val in manifest.surfaces.items():
            if isinstance(val, str):
                seen_paths.append(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        seen_paths.append(item)

        blocks: List[str] = []
        for rel in seen_paths:
            p = (manifest.base_dir / rel).resolve()
            if p.exists() and p.is_file():
                content = p.read_text(encoding="utf-8", errors="replace")
                if len(content) > 2500:
                    content = content[:2500] + "\n...[truncated]..."
                blocks.append(f"--- FILE: {rel} ---\n{content}")
            else:
                blocks.append(f"--- FILE: {rel} (missing) ---")
        return "\n\n".join(blocks)

    @staticmethod
    def _format_trace_digest(traces: Optional[List[ExecutionTrace]]) -> str:
        if not traces:
            return "(no raw execution traces attached)"

        lines: List[str] = []
        for t in traces:
            status = "PASS" if t.passed else "FAIL"
            lines.append(
                f"Task {t.task_id}: {status} | steps={len(t.steps)} | tokens={t.total_tokens} | latency_ms={t.total_latency_ms:.0f}"
            )
            if t.error_summary:
                lines.append(f"  error_summary: {t.error_summary[:300]}")
            if t.verifier_output:
                lines.append(f"  verifier_output: {t.verifier_output.strip()[:400]}")
            for step in t.steps[:8]:
                thought_snip = (step.thought or "").replace("\n", " ")[:180]
                if thought_snip:
                    lines.append(f"  Step {step.step_number} thought: {thought_snip}")
                for tc in step.tool_calls[:4]:
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)[:180]
                    out_str = (tc.output or "").replace("\n", " ")[:220]
                    lines.append(
                        f"    tool={tc.tool_name} status={tc.status} args={args_str} -> out={out_str}"
                    )
        return "\n".join(lines)

    def _build_prompt(
        self,
        manifest: HarnessManifest,
        evidence: EvidenceBundle,
        max_proposals: int,
        traces: Optional[List[ExecutionTrace]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        history_block = (
            json.dumps(history[-6:], indent=2, ensure_ascii=False)
            if history
            else "None (first optimization round)"
        )

        if self.use_fault_localization and traces:
            localized = localize_suspects(manifest=manifest, evidence=evidence, traces=traces)
            self.last_localization = localized.metadata_dict()
            context_section = localized.compact_prompt_block
        else:
            self.last_localization = None
            surface_files = self._read_surface_contents(manifest)
            trace_digest = self._format_trace_digest(traces)
            context_section = (
                "CURRENT HARNESS SURFACE FILES:\n"
                f"{surface_files}\n\n"
                "HELD-IN EXECUTION TRACE DIGEST:\n"
                f"{trace_digest}"
            )

        return (
            "You are the Dedicated Harness Orchestrator optimizing a `pi` coding agent harness.\n"
            "CRITICAL EXECUTION RULES:\n"
            "- Do NOT call any tools. Do NOT inspect or modify workspace files.\n"
            "- Respond ONLY with a single valid JSON object (no markdown commentary outside JSON).\n\n"
            "TARGET WORKER AGENT & MODEL:\n"
            f"- Agent: Pi Coding Agent (`pi` CLI)\n"
            f"- Model: {manifest.model_profile.model_name} (provider: {manifest.model_profile.provider}, "
            f"thinking_level: {manifest.model_profile.thinking_level or 'default'})\n\n"
            "ALLOWED EDITABLE SURFACES (relative target_file paths MUST match these exact paths):\n"
            f"{json.dumps(manifest.surfaces, indent=2)}\n\n"
            f"IMMUTABLE BOUNDARIES (NEVER touch): {manifest.immutable_boundaries}\n\n"
            "HELD-IN FAILURE EVIDENCE:\n"
            f"{evidence.summary()}\n\n"
            f"{context_section}\n\n"
            "PRIOR ROUND HISTORY (for avoiding repeated failed patches):\n"
            f"{history_block}\n\n"
            "PATCH SCHEMA & DESIGN GUIDELINES:\n"
            f"- Generate between 1 and {max_proposals} distinct candidate proposals.\n"
            "- Keep `prompts/system.md` concise (< 1000 tokens total) and actionable for a 27B local coding model (arex-2).\n"
            "- Prefer surgical patches on the isolated suspect surfaces/sections (`delete_section`, `replace_section`, `append_rule`, `replace_key`) instead of rewriting unrelated non-suspect sections.\n"
            "- Do NOT hardcode task-specific answers or task filenames into the harness.\n"
            "- Allowed patch operations:\n"
            "  * For `.yaml` files (`runtime_policy`, `tool_policy`, `verification_policy`): `replace_key` (path is YAML key or `parent/child`) or `replace_content`.\n"
            "  * For `.md` files (`system_prompt`, `skills`): `replace_section` (path is heading title), `delete_section` (path is heading title), `append_rule` (path is section heading, value is bullet rule text), `append_content`, or `replace_content`.\n\n"
            "REQUIRED JSON OUTPUT FORMAT:\n"
            "{\n"
            '  "diagnosis": "Concise root-cause analysis of why arex-2 failed on held-in tasks",\n'
            '  "candidates": [\n'
            "    {\n"
            '      "candidate_id": "cand_arex2_1",\n'
            f'      "target_model": "{manifest.model_profile.model_name}",\n'
            '      "hypothesis": "Why these surgical edits fix the failures without overfitting",\n'
            '      "evidence_cluster": "FC_TIMEOUT",\n'
            '      "expected_gain": "pass_rate",\n'
            '      "regression_risk": "low",\n'
            '      "patches": [\n'
            "        {\n"
            '          "surface": "system_prompt",\n'
            '          "target_file": "prompts/system.md",\n'
            '          "operation": "delete_section",\n'
            '          "path": "Codebase Navigation & Architecture Exploration Protocol",\n'
            '          "value": ""\n'
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}"
        )

    @staticmethod
    def _strip_code_fences(raw_text: str) -> str:
        stripped = raw_text.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, re.IGNORECASE)
        if fence_match:
            return fence_match.group(1).strip()
        return stripped

    def _parse_llm_proposals(
        self,
        raw_text: str,
        default_model: str = "target",
        max_proposals: int = 3,
    ) -> List[HarnessCandidate]:
        cleaned = self._strip_code_fences(raw_text)
        payload: Any = None

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            obj_start, obj_end = cleaned.find("{"), cleaned.rfind("}")
            arr_start, arr_end = cleaned.find("["), cleaned.rfind("]")
            if obj_start != -1 and obj_end > obj_start:
                try:
                    payload = json.loads(cleaned[obj_start : obj_end + 1])
                except json.JSONDecodeError:
                    payload = None
            if payload is None and arr_start != -1 and arr_end > arr_start:
                try:
                    payload = json.loads(cleaned[arr_start : arr_end + 1])
                except json.JSONDecodeError:
                    payload = None

        if isinstance(payload, dict):
            diagnosis = payload.get("diagnosis")
            if isinstance(diagnosis, str) and diagnosis.strip():
                self.last_diagnosis = diagnosis.strip()
            items = payload.get("candidates", [])
        elif isinstance(payload, list):
            items = payload
        else:
            return []

        if not isinstance(items, list):
            return []

        candidates: List[HarnessCandidate] = []
        for idx, item in enumerate(items[:max_proposals]):
            if not isinstance(item, dict):
                continue
            raw_cid = str(item.get("candidate_id") or f"cand_{idx + 1}")
            safe_cid = re.sub(r"[^A-Za-z0-9_-]", "_", raw_cid).strip("_") or f"cand_{idx + 1}"

            patches: List[DeclarativePatch] = []
            for p in item.get("patches", []):
                if not isinstance(p, dict):
                    continue
                if "surface" not in p or "target_file" not in p or "operation" not in p:
                    continue
                patches.append(
                    DeclarativePatch(
                        surface=str(p["surface"]),
                        target_file=str(p["target_file"]),
                        operation=str(p["operation"]),
                        path=str(p.get("path", "")),
                        value=p.get("value"),
                    )
                )

            if not patches:
                continue

            candidates.append(
                HarnessCandidate(
                    candidate_id=safe_cid,
                    target_model=str(item.get("target_model") or default_model),
                    hypothesis=str(item.get("hypothesis") or "Orchestrator proposed patch"),
                    evidence_cluster=str(item.get("evidence_cluster") or "FC_GENERAL_LOGIC"),
                    patches=patches,
                    expected_gain=str(item.get("expected_gain") or "pass_rate"),
                    regression_risk=str(item.get("regression_risk") or "low"),
                )
            )
        return candidates
