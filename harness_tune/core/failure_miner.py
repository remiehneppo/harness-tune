"""Failure Miner: Cluster failure traces into structured symptom evidence."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict
from harness_tune.core.trace import ExecutionTrace
from harness_tune.core.fault_localizer import extract_verifier_signal


@dataclass
class FailureCluster:
    cluster_id: str
    pattern_name: str
    frequency: int
    affected_tasks: List[str]
    symptoms: List[str]
    sample_errors: List[str]
    inferred_cause: str
    suggested_surface: str


@dataclass
class EvidenceBundle:
    total_runs: int
    failed_runs: int
    failure_rate: float
    clusters: List[FailureCluster] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Evaluated Runs: {self.total_runs}, Failed: {self.failed_runs} ({self.failure_rate:.1%})",
            f"Identified {len(self.clusters)} Failure Clusters:",
        ]
        for c in self.clusters:
            lines.append(
                f"- [{c.cluster_id}] {c.pattern_name} (Count: {c.frequency}) -> Surface: {c.suggested_surface}"
            )
            lines.append(f"  Cause: {c.inferred_cause}")
            if c.affected_tasks:
                lines.append(f"  Affected Tasks: {', '.join(c.affected_tasks)}")
            if c.sample_errors:
                samples = " | ".join(err.replace("\n", " ")[:180] for err in c.sample_errors[:3])
                lines.append(f"  Sample Errors: {samples}")
        return "\n".join(lines)


class FailureMiner:
    """Mines execution traces to discover recurring failure patterns."""

    def analyze(self, traces: List[ExecutionTrace]) -> EvidenceBundle:
        failed_traces = [t for t in traces if not t.passed]
        total = len(traces)
        failed_count = len(failed_traces)
        rate = (failed_count / total) if total > 0 else 0.0

        clusters: Dict[str, FailureCluster] = {}

        for t in failed_traces:
            err_lower = (t.error_summary or "").lower()

            # 1. Check timeout failures
            if "timed out" in err_lower or "timeout" in err_lower:
                cid = "FC_TIMEOUT"
                if cid not in clusters:
                    clusters[cid] = FailureCluster(
                        cluster_id=cid,
                        pattern_name="Execution Timeout",
                        frequency=0,
                        affected_tasks=[],
                        symptoms=["Agent exceeded per-task execution timeout before finishing"],
                        sample_errors=[],
                        inferred_cause="Excessive reasoning/exploration steps or hanging command",
                        suggested_surface="runtime_policy",
                    )
                clusters[cid].frequency += 1
                if t.task_id not in clusters[cid].affected_tasks:
                    clusters[cid].affected_tasks.append(t.task_id)
                if t.error_summary and len(clusters[cid].sample_errors) < 3:
                    clusters[cid].sample_errors.append(t.error_summary[:180])

            # 2. Check repeated tool error streak
            streak = t.detect_consecutive_same_errors()
            if streak >= 2:
                cid = "FC_TOOL_LOOP"
                if cid not in clusters:
                    clusters[cid] = FailureCluster(
                        cluster_id=cid,
                        pattern_name="Repeated Tool Failure Loop",
                        frequency=0,
                        affected_tasks=[],
                        symptoms=["Agent repeatedly calls failing tool with identical arguments"],
                        sample_errors=[],
                        inferred_cause="Lack of loop breaker or missing error recovery instructions",
                        suggested_surface="runtime_policy",
                    )
                clusters[cid].frequency += 1
                if t.task_id not in clusters[cid].affected_tasks:
                    clusters[cid].affected_tasks.append(t.task_id)
                if t.failed_tool_calls and len(clusters[cid].sample_errors) < 3:
                    clusters[cid].sample_errors.append(t.failed_tool_calls[-1].output[:180])

            # 3. Check bash syntax / missing dependency
            for tc in t.failed_tool_calls:
                err = tc.output.lower()
                if "command not found" in err or "no such file or directory" in err:
                    cid = "FC_MISSING_DEP"
                    if cid not in clusters:
                        clusters[cid] = FailureCluster(
                            cluster_id=cid,
                            pattern_name="Missing Dependency or Path Error",
                            frequency=0,
                            affected_tasks=[],
                            symptoms=["Command execution failed due to missing binary or wrong working directory"],
                            sample_errors=[],
                            inferred_cause="Agent does not verify dependencies or path before running commands",
                            suggested_surface="tool_policy",
                        )
                    clusters[cid].frequency += 1
                    if t.task_id not in clusters[cid].affected_tasks:
                        clusters[cid].affected_tasks.append(t.task_id)
                    if len(clusters[cid].sample_errors) < 3:
                        clusters[cid].sample_errors.append(tc.output[:180])

                elif "syntaxerror" in err or "invalid syntax" in err:
                    cid = "FC_SYNTAX_ERROR"
                    if cid not in clusters:
                        clusters[cid] = FailureCluster(
                            cluster_id=cid,
                            pattern_name="Code Syntax Generation Error",
                            frequency=0,
                            affected_tasks=[],
                            symptoms=["Generated code fails basic compilation or parsing"],
                            sample_errors=[],
                            inferred_cause="Agent lacks verification phase before completing task",
                            suggested_surface="verification_policy",
                        )
                    clusters[cid].frequency += 1
                    if t.task_id not in clusters[cid].affected_tasks:
                        clusters[cid].affected_tasks.append(t.task_id)
                    if len(clusters[cid].sample_errors) < 3:
                        clusters[cid].sample_errors.append(tc.output[:180])

            # 4. Check step exhaustion
            if len(t.steps) >= 25 or "max_steps" in err_lower:
                cid = "FC_STEP_EXHAUSTION"
                if cid not in clusters:
                    clusters[cid] = FailureCluster(
                        cluster_id=cid,
                        pattern_name="Max Step Exhaustion",
                        frequency=0,
                        affected_tasks=[],
                        symptoms=["Agent took maximum allowed steps without reaching solution"],
                        sample_errors=[],
                        inferred_cause="Agent spent too many steps exploring without transitioning to implementation",
                        suggested_surface="runtime_policy",
                    )
                clusters[cid].frequency += 1
                if t.task_id not in clusters[cid].affected_tasks:
                    clusters[cid].affected_tasks.append(t.task_id)

            # 5. Default unclassified cluster if none matched
            if not any(t.task_id in c.affected_tasks for c in clusters.values()):
                cid = "FC_GENERAL_LOGIC"
                if cid not in clusters:
                    clusters[cid] = FailureCluster(
                        cluster_id=cid,
                        pattern_name="General Logic or Assertion Failure",
                        frequency=0,
                        affected_tasks=[],
                        symptoms=[t.error_summary or "Verifier test assertion failed"],
                        sample_errors=[],
                        inferred_cause="Agent output failed grader criteria",
                        suggested_surface="system_prompt",
                    )
                clusters[cid].frequency += 1
                clusters[cid].affected_tasks.append(t.task_id)
                clean_ver = extract_verifier_signal(t.verifier_output, max_chars=180)
                sample = (clean_ver or t.error_summary or "").strip()
                if sample and len(clusters[cid].sample_errors) < 3:
                    clusters[cid].sample_errors.append(sample[:180])

        return EvidenceBundle(
            total_runs=total,
            failed_runs=failed_count,
            failure_rate=rate,
            clusters=list(clusters.values()),
        )
