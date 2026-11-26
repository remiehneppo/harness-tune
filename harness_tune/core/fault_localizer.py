"""Deterministic Fault Localization for isolating suspect harness surfaces, prompt sections, and failure windows."""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from harness_tune.core.manifest import HarnessManifest
from harness_tune.core.trace import ExecutionTrace, StepTrace

if TYPE_CHECKING:
    from harness_tune.core.failure_miner import EvidenceBundle

NOISE_LINE_PREFIXES = (
    "debconf:",
    "get:",
    "hit:",
    "ign:",
    "fetched ",
    "reading package lists",
    "building dependency tree",
    "reading state information",
    "the following ",
    "need to get ",
    "after this operation",
    "selecting previously unselected",
    "preparing to unpack",
    "unpacking ",
    "setting up ",
    "processing triggers",
    "downloading ",
    "installed ",
    "audited ",
)

HIGH_SIGNAL_MARKERS = (
    "FAILED ",
    "ERROR ",
    "E       ",
    "AssertionError",
    "FileNotFoundError",
    "ModuleNotFoundError",
    "SyntaxError",
    "TypeError",
    "ValueError",
    "Traceback",
    "failed in ",
    "command not found",
    "Permission denied",
    "Segmentation fault",
)


@dataclass
class MarkdownSection:
    title: str
    level: int
    body: str
    full_text: str


@dataclass
class DifferentialReport:
    total_runs: int
    passed_runs: int
    failed_runs: int
    pass_avg_steps: float
    fail_avg_steps: float
    pass_avg_tokens: float
    fail_avg_tokens: float
    pass_tokens_per_step: float
    fail_tokens_per_step: float
    pass_avg_first_edit_step: Optional[float]
    fail_avg_first_edit_step: Optional[float]
    zero_tool_call_failures: int
    top_failing_tools: List[Tuple[str, int, str]] = field(default_factory=list)
    skills_read_in_pass: List[str] = field(default_factory=list)
    skills_read_in_fail: List[str] = field(default_factory=list)
    missing_or_unused_protocol_tools: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_runs": self.total_runs,
            "passed_runs": self.passed_runs,
            "failed_runs": self.failed_runs,
            "pass_avg_steps": round(self.pass_avg_steps, 2),
            "fail_avg_steps": round(self.fail_avg_steps, 2),
            "pass_avg_tokens": round(self.pass_avg_tokens, 1),
            "fail_avg_tokens": round(self.fail_avg_tokens, 1),
            "pass_tokens_per_step": round(self.pass_tokens_per_step, 1),
            "fail_tokens_per_step": round(self.fail_tokens_per_step, 1),
            "pass_avg_first_edit_step": (
                round(self.pass_avg_first_edit_step, 2)
                if self.pass_avg_first_edit_step is not None
                else None
            ),
            "fail_avg_first_edit_step": (
                round(self.fail_avg_first_edit_step, 2)
                if self.fail_avg_first_edit_step is not None
                else None
            ),
            "zero_tool_call_failures": self.zero_tool_call_failures,
            "top_failing_tools": [
                {"tool": t, "count": c, "sample_error": s}
                for t, c, s in self.top_failing_tools
            ],
            "skills_read_in_pass": self.skills_read_in_pass,
            "skills_read_in_fail": self.skills_read_in_fail,
            "missing_or_unused_protocol_tools": self.missing_or_unused_protocol_tools,
        }


@dataclass
class LocalizedContext:
    suspect_surfaces: List[str]
    suspect_sections: List[str]
    active_skills_in_failures: List[str]
    differential: DifferentialReport
    compact_prompt_block: str
    full_context_chars: int
    localized_context_chars: int

    @property
    def reduction_pct(self) -> float:
        if self.full_context_chars <= 0:
            return 0.0
        return max(
            0.0,
            (1.0 - (self.localized_context_chars / self.full_context_chars)) * 100.0,
        )

    def metadata_dict(self) -> Dict[str, Any]:
        return {
            "suspect_surfaces": self.suspect_surfaces,
            "suspect_sections": self.suspect_sections,
            "active_skills_in_failures": self.active_skills_in_failures,
            "full_context_chars": self.full_context_chars,
            "localized_context_chars": self.localized_context_chars,
            "reduction_pct": round(self.reduction_pct, 1),
            "differential": self.differential.to_dict(),
        }


def extract_verifier_signal(verifier_output: Optional[str], max_chars: int = 500) -> str:
    """Strip apt/uvx/download install noise and retain failing test assertions."""
    if not verifier_output:
        return ""

    cleaned_lines: List[str] = []
    for raw in verifier_output.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if any(low.startswith(prefix) for prefix in NOISE_LINE_PREFIXES):
            continue
        if low.startswith("downloading ") or low.startswith("download "):
            continue
        if re.match(r"^[=\-]{10,}$", line):
            continue
        cleaned_lines.append(line)

    if not cleaned_lines:
        return verifier_output.strip()[-max_chars:]

    signal_lines: List[str] = []
    for idx, line in enumerate(cleaned_lines):
        if any(marker in line for marker in HIGH_SIGNAL_MARKERS):
            if line not in signal_lines:
                signal_lines.append(line)
        elif line.startswith("E   ") or line.startswith("> "):
            if line not in signal_lines:
                signal_lines.append(line)

    chosen = signal_lines if signal_lines else cleaned_lines[-6:]
    joined = "\n".join(chosen)
    if len(joined) > max_chars:
        return joined[: max_chars - 15] + "\n...[truncated]"
    return joined


def parse_markdown_sections(md_text: str) -> List[MarkdownSection]:
    """Parse Markdown into top-level / sub-level heading sections."""
    lines = md_text.splitlines()
    sections: List[MarkdownSection] = []
    current_title = "(intro)"
    current_level = 0
    current_lines: List[str] = []

    heading_re = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
    for line in lines:
        m = heading_re.match(line)
        if m and (len(m.group(1)) >= 2 or current_lines or sections):
            if current_lines or current_title != "(intro)":
                full = "\n".join(current_lines).strip()
                body = "\n".join(current_lines[1:] if current_level > 0 else current_lines).strip()
                sections.append(
                    MarkdownSection(
                        title=current_title,
                        level=current_level,
                        body=body,
                        full_text=full,
                    )
                )
            current_level = len(m.group(1))
            current_title = m.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines or current_title != "(intro)":
        full = "\n".join(current_lines).strip()
        body = "\n".join(current_lines[1:] if current_level > 0 else current_lines).strip()
        sections.append(
            MarkdownSection(
                title=current_title,
                level=current_level,
                body=body,
                full_text=full,
            )
        )
    return sections


def slice_failure_window(
    trace: ExecutionTrace,
    window_before: int = 1,
    window_after: int = 1,
    tail_steps: int = 3,
    max_selected_steps: int = 6,
) -> List[StepTrace]:
    """Select only the failure-relevant step windows instead of steps[:N]."""
    steps = trace.steps
    if not steps:
        return []
    if len(steps) <= max_selected_steps:
        return list(steps)

    selected_indices = set()
    # Always include step 1 to show initial orientation / failure
    selected_indices.add(0)

    first_error_idx: Optional[int] = None
    for idx, step in enumerate(steps):
        if any(tc.status == "error" for tc in step.tool_calls):
            first_error_idx = idx
            break

    if first_error_idx is not None:
        start = max(0, first_error_idx - window_before)
        end = min(len(steps), first_error_idx + window_after + 1)
        for idx in range(start, end):
            selected_indices.add(idx)

    for idx in range(max(0, len(steps) - tail_steps), len(steps)):
        selected_indices.add(idx)

    if len(selected_indices) > max_selected_steps:
        last_idx = len(steps) - 1
        must_keep = {last_idx}
        if first_error_idx is not None:
            must_keep.add(first_error_idx)
        extras = sorted(i for i in selected_indices if i not in must_keep)
        keep_budget = max(0, max_selected_steps - len(must_keep))
        # Prefer recent tail & error-adjacent steps over step 0 if over budget
        selected_indices = must_keep | set(extras[-keep_budget:]) if keep_budget else must_keep

    ordered = sorted(selected_indices)
    return [steps[i] for i in ordered]


def _extract_skill_name_from_path(path_str: str) -> Optional[str]:
    norm = path_str.replace("\\", "/")
    if "skill" not in norm.lower():
        return None
    parts = [p for p in norm.split("/") if p]
    if parts and parts[-1].lower() == "skill.md" and len(parts) >= 2:
        return parts[-2]
    return parts[-1] if parts else None


def compute_differential_attribution(traces: Sequence[ExecutionTrace]) -> DifferentialReport:
    """Compute Pass-vs-Fail behavioral contrasts across held-in traces."""
    passed = [t for t in traces if t.passed]
    failed = [t for t in traces if not t.passed]

    def _avg_steps(items: Sequence[ExecutionTrace]) -> float:
        return (sum(len(t.steps) for t in items) / len(items)) if items else 0.0

    def _avg_tokens(items: Sequence[ExecutionTrace]) -> float:
        return (sum(t.total_tokens for t in items) / len(items)) if items else 0.0

    def _tps(items: Sequence[ExecutionTrace]) -> float:
        tot_steps = sum(len(t.steps) for t in items)
        tot_toks = sum(t.total_tokens for t in items)
        return (tot_toks / tot_steps) if tot_steps > 0 else 0.0

    def _avg_first_edit(items: Sequence[ExecutionTrace]) -> Optional[float]:
        vals: List[int] = []
        for t in items:
            for s in t.steps:
                if any(tc.tool_name in ("write", "edit") for tc in s.tool_calls):
                    vals.append(s.step_number)
                    break
        return (sum(vals) / len(vals)) if vals else None

    tool_fail_counts: Dict[str, int] = {}
    tool_fail_samples: Dict[str, str] = {}
    tool_usage_counts: Dict[str, int] = {}

    for t in failed:
        for s in t.steps:
            for tc in s.tool_calls:
                tool_usage_counts[tc.tool_name] = tool_usage_counts.get(tc.tool_name, 0) + 1
                if tc.status == "error":
                    tool_fail_counts[tc.tool_name] = tool_fail_counts.get(tc.tool_name, 0) + 1
                    if tc.tool_name not in tool_fail_samples:
                        tool_fail_samples[tc.tool_name] = (tc.output or "").replace("\n", " ")[:120]

    top_failing_tools = [
        (name, cnt, tool_fail_samples.get(name, ""))
        for name, cnt in sorted(tool_fail_counts.items(), key=lambda x: -x[1])[:5]
    ]

    def _skills_read(items: Sequence[ExecutionTrace]) -> List[str]:
        found: List[str] = []
        for t in items:
            for s in t.steps:
                for tc in s.tool_calls:
                    if tc.tool_name == "read":
                        p = str(tc.arguments.get("path", ""))
                        sk = _extract_skill_name_from_path(p)
                        if sk and sk not in found:
                            found.append(sk)
        return found

    zero_tool_failures = sum(1 for t in failed if t.total_tool_calls == 0)

    unused_protocol_tools: List[str] = []
    if failed:
        if not any(k.startswith("codegraph_") for k in tool_usage_counts):
            unused_protocol_tools.append("codegraph_*")
        if not any("bd " in json.dumps(tc.arguments) for t in failed for s in t.steps for tc in s.tool_calls):
            unused_protocol_tools.append("bd (beads)")

    return DifferentialReport(
        total_runs=len(traces),
        passed_runs=len(passed),
        failed_runs=len(failed),
        pass_avg_steps=_avg_steps(passed),
        fail_avg_steps=_avg_steps(failed),
        pass_avg_tokens=_avg_tokens(passed),
        fail_avg_tokens=_avg_tokens(failed),
        pass_tokens_per_step=_tps(passed),
        fail_tokens_per_step=_tps(failed),
        pass_avg_first_edit_step=_avg_first_edit(passed),
        fail_avg_first_edit_step=_avg_first_edit(failed),
        zero_tool_call_failures=zero_tool_failures,
        top_failing_tools=top_failing_tools,
        skills_read_in_pass=_skills_read(passed),
        skills_read_in_fail=_skills_read(failed),
        missing_or_unused_protocol_tools=unused_protocol_tools,
    )


def _format_trace_slice(trace: ExecutionTrace) -> str:
    lines: List[str] = [
        f"Task `{trace.task_id}`: FAIL | steps={len(trace.steps)} | tokens={trace.total_tokens} | latency={trace.total_latency_ms / 1000.0:.1f}s"
    ]
    if trace.error_summary:
        lines.append(f"  error_summary: {trace.error_summary[:220]}")
    ver_sig = extract_verifier_signal(trace.verifier_output, max_chars=420)
    if ver_sig:
        lines.append(f"  verifier_signal:\n    " + ver_sig.replace("\n", "\n    "))

    sliced = slice_failure_window(trace)
    if sliced:
        shown_nums = [s.step_number for s in sliced]
        lines.append(f"  failure_window_steps ({shown_nums} of {len(trace.steps)}):")
        for step in sliced:
            thought_snip = (step.thought or "").replace("\n", " ").strip()[:140]
            lines.append(
                f"    Step {step.step_number} (tokens={step.tokens_used})"
                + (f" thought={thought_snip}" if thought_snip else "")
            )
            for tc in step.tool_calls[:3]:
                args_str = json.dumps(tc.arguments, ensure_ascii=False)[:140]
                out_str = (tc.output or "").replace("\n", " ").strip()[:160]
                lines.append(
                    f"      - {tc.tool_name} [{tc.status}] args={args_str} -> {out_str}"
                )
    return "\n".join(lines)


def _estimate_full_dump_chars(manifest: HarnessManifest, evidence: EvidenceBundle, traces: Sequence[ExecutionTrace]) -> int:
    total = len(evidence.summary())
    if manifest.base_dir:
        for val in manifest.surfaces.values():
            paths = [val] if isinstance(val, str) else (val if isinstance(val, list) else [])
            for rel in paths:
                if isinstance(rel, str):
                    p = (manifest.base_dir / rel).resolve()
                    if p.exists() and p.is_file():
                        total += min(2500, len(p.read_text(encoding="utf-8", errors="replace")))
    for t in traces:
        total += 220 + min(400, len(t.verifier_output or ""))
        for s in t.steps[:8]:
            total += min(180, len(s.thought or "")) + 200 * len(s.tool_calls[:4])
    return total


def localize_suspects(
    manifest: HarnessManifest,
    evidence: EvidenceBundle,
    traces: Sequence[ExecutionTrace],
) -> LocalizedContext:
    """Isolate suspect surfaces, prompt sections, and failure windows for the proposer."""
    diff = compute_differential_attribution(traces)

    suspect_surfaces: List[str] = []
    for c in evidence.clusters:
        if c.suggested_surface in manifest.surfaces and c.suggested_surface not in suspect_surfaces:
            suspect_surfaces.append(c.suggested_surface)

    cluster_ids = {c.cluster_id for c in evidence.clusters}
    if (
        diff.zero_tool_call_failures > 0
        or (diff.pass_tokens_per_step > 0 and diff.fail_tokens_per_step > 1.4 * diff.pass_tokens_per_step)
        or "FC_TIMEOUT" in cluster_ids
        or "FC_STEP_EXHAUSTION" in cluster_ids
    ):
        if "runtime_policy" in manifest.surfaces and "runtime_policy" not in suspect_surfaces:
            suspect_surfaces.append("runtime_policy")

    if (
        "FC_GENERAL_LOGIC" in cluster_ids
        or "FC_TIMEOUT" in cluster_ids
        or diff.zero_tool_call_failures > 0
        or (
            diff.pass_avg_first_edit_step is not None
            and diff.fail_avg_first_edit_step is not None
            and diff.fail_avg_first_edit_step > diff.pass_avg_first_edit_step + 2.0
        )
    ):
        if "system_prompt" in manifest.surfaces and "system_prompt" not in suspect_surfaces:
            suspect_surfaces.append("system_prompt")

    if not suspect_surfaces:
        for s_name in ("system_prompt", "runtime_policy", "tool_policy"):
            if s_name in manifest.surfaces:
                suspect_surfaces.append(s_name)

    # Section-level suspect detection on system_prompt
    suspect_sections: List[str] = []
    sys_rel = manifest.surfaces.get("system_prompt")
    sys_text = ""
    if isinstance(sys_rel, str) and manifest.base_dir:
        sys_path = (manifest.base_dir / sys_rel).resolve()
        if sys_path.exists():
            sys_text = sys_path.read_text(encoding="utf-8", errors="replace")

    sections = parse_markdown_sections(sys_text) if sys_text else []
    for sec in sections:
        sec_low = (sec.title + "\n" + sec.body).lower()
        is_suspect = False
        if "codegraph" in sec_low and "codegraph_*" in diff.missing_or_unused_protocol_tools:
            is_suspect = True
        if ("beads" in sec_low or "`bd " in sec_low) and "bd (beads)" in diff.missing_or_unused_protocol_tools:
            is_suspect = True
        if ("web-search" in sec_low or "mandatory research" in sec_low) and (
            "FC_TIMEOUT" in cluster_ids or "FC_STEP_EXHAUSTION" in cluster_ids or diff.zero_tool_call_failures > 0
        ):
            is_suspect = True
        if "verification" in sec_low and ("FC_GENERAL_LOGIC" in cluster_ids or "FC_SYNTAX_ERROR" in cluster_ids):
            is_suspect = True
        if is_suspect and sec.title not in suspect_sections:
            suspect_sections.append(sec.title)

    # Render prompt block
    parts: List[str] = []
    parts.append(
        "PASS-VS-FAIL DIFFERENTIAL ATTRIBUTION (Held-in):\n"
        f"- Pass: {diff.passed_runs}/{diff.total_runs} | avg_steps={diff.pass_avg_steps:.1f} | avg_tokens={diff.pass_avg_tokens:.0f} | tokens/step={diff.pass_tokens_per_step:.0f} | avg_first_edit_step={diff.pass_avg_first_edit_step}\n"
        f"- Fail: {diff.failed_runs}/{diff.total_runs} | avg_steps={diff.fail_avg_steps:.1f} | avg_tokens={diff.fail_avg_tokens:.0f} | tokens/step={diff.fail_tokens_per_step:.0f} | avg_first_edit_step={diff.fail_avg_first_edit_step}\n"
        f"- Zero-tool-call failures: {diff.zero_tool_call_failures}\n"
        f"- Unused protocol tools in failed traces: {diff.missing_or_unused_protocol_tools or 'none'}\n"
        f"- Top failing tools: {diff.top_failing_tools or 'none'}\n"
        f"- Skills read in failed runs: {diff.skills_read_in_fail or 'none'}"
    )

    parts.append(
        "SUSPECT SURFACES ROUTED BY LOCALIZER:\n"
        f"- Suspect surfaces (expanded below): {suspect_surfaces}\n"
        f"- Suspect prompt sections flagged: {suspect_sections or ['(all sections shown — no narrow mismatch)']}"
    )

    surface_blocks: List[str] = []
    for s_name, val in manifest.surfaces.items():
        if isinstance(val, list):
            surface_blocks.append(
                f"--- SURFACE `{s_name}` (LIST OUTLINE: {val}; skills read in failed runs: {diff.skills_read_in_fail or 'none'}) ---"
            )
            continue
        if not isinstance(val, str) or not manifest.base_dir:
            continue
        fpath = (manifest.base_dir / val).resolve()
        content = fpath.read_text(encoding="utf-8", errors="replace") if fpath.exists() else "(missing)"

        if s_name not in suspect_surfaces:
            keys_preview = [ln.strip() for ln in content.splitlines() if ln.strip() and not ln.strip().startswith("#")][:8]
            surface_blocks.append(
                f"--- NON-SUSPECT SURFACE `{s_name}` ({val}) [OUTLINE ONLY] ---\n"
                + "\n".join(keys_preview)
            )
            continue

        if s_name == "system_prompt" and sections and suspect_sections:
            sec_out: List[str] = []
            for sec in sections:
                if sec.title in suspect_sections:
                    sec_out.append(f"[SUSPECT SECTION]\n{sec.full_text}")
                else:
                    sec_out.append(f"[NON-SUSPECT SECTION OUTLINE]: {'#' * max(1, sec.level)} {sec.title}")
            surface_blocks.append(
                f"--- SUSPECT SURFACE `system_prompt` ({val}) ---\n" + "\n\n".join(sec_out)
            )
        else:
            if len(content) > 2200:
                content = content[:2200] + "\n...[truncated]..."
            surface_blocks.append(f"--- SUSPECT SURFACE `{s_name}` ({val}) ---\n{content}")

    parts.append("LOCALIZED HARNESS SURFACES:\n" + "\n\n".join(surface_blocks))

    failed_traces = [t for t in traces if not t.passed]
    passed_ids = [t.task_id for t in traces if t.passed]
    slice_blocks = [_format_trace_slice(t) for t in failed_traces]
    parts.append(
        f"HELD-IN FAILURE WINDOW SLICES ({len(failed_traces)} failed tasks; {len(passed_ids)} passed tasks omitted from raw trace dump: {passed_ids}):\n"
        + "\n\n".join(slice_blocks)
    )

    compact_block = "\n\n".join(parts)
    full_chars = _estimate_full_dump_chars(manifest, evidence, traces)
    return LocalizedContext(
        suspect_surfaces=suspect_surfaces,
        suspect_sections=suspect_sections,
        active_skills_in_failures=diff.skills_read_in_fail,
        differential=diff,
        compact_prompt_block=compact_block,
        full_context_chars=full_chars,
        localized_context_chars=len(compact_block),
    )
