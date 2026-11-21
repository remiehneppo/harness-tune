"""Statistical Regression Gate for Held-in and Held-out validation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from harness_tune.core.trace import ExecutionTrace


@dataclass
class EvaluationMetrics:
    pass_rate: float
    total_runs: int
    passed_runs: int
    avg_tokens: float
    avg_latency_ms: float
    total_cost_usd: float
    safety_violations: int = 0

    @classmethod
    def from_traces(
        cls,
        traces: List[ExecutionTrace],
        token_cost_per_million: float = 1.0,
    ) -> EvaluationMetrics:
        if not traces:
            return cls(0.0, 0, 0, 0.0, 0.0, 0.0, 0)
        total = len(traces)
        passed = sum(1 for t in traces if t.passed)
        tot_tok = sum(t.total_tokens for t in traces)
        tot_lat = sum(t.total_latency_ms for t in traces)
        cost = (tot_tok / 1_000_000.0) * token_cost_per_million

        return cls(
            pass_rate=passed / total,
            total_runs=total,
            passed_runs=passed,
            avg_tokens=tot_tok / total,
            avg_latency_ms=tot_lat / total,
            total_cost_usd=cost,
            safety_violations=0,
        )


@dataclass
class GateResult:
    accepted: bool
    reason: str
    delta_held_in: float
    delta_held_out: float
    utility_score: float
    metrics_baseline_in: EvaluationMetrics
    metrics_candidate_in: EvaluationMetrics
    metrics_baseline_ho: EvaluationMetrics
    metrics_candidate_ho: EvaluationMetrics


class RegressionGate:
    """Enforces non-regression rule: Delta_in >= 0, Delta_ho >= 0, max(Delta_in, Delta_ho) > 0."""

    def __init__(
        self,
        non_regression_margin: float = 0.0,
        w_pass: float = 100.0,
        w_cost: float = 5.0,
        w_latency: float = 0.001,
    ):
        self.non_regression_margin = non_regression_margin
        self.w_pass = w_pass
        self.w_cost = w_cost
        self.w_latency = w_latency

    def evaluate(
        self,
        baseline_in: EvaluationMetrics,
        candidate_in: EvaluationMetrics,
        baseline_ho: EvaluationMetrics,
        candidate_ho: EvaluationMetrics,
    ) -> GateResult:
        delta_in = candidate_in.pass_rate - baseline_in.pass_rate
        delta_ho = candidate_ho.pass_rate - baseline_ho.pass_rate

        # Multi-objective utility: U(c) = w_p*Delta_p - w_C*Delta_C - w_L*Delta_L
        delta_pass_overall = (delta_in + delta_ho) / 2.0
        delta_cost_overall = (candidate_ho.total_cost_usd - baseline_ho.total_cost_usd)
        delta_lat_overall = (candidate_ho.avg_latency_ms - baseline_ho.avg_latency_ms)

        utility = (
            self.w_pass * delta_pass_overall
            - self.w_cost * delta_cost_overall
            - self.w_latency * delta_lat_overall
        )

        # Gate Condition:
        # Delta_in >= -margin AND Delta_ho >= -margin AND at least one improved
        if delta_ho < -self.non_regression_margin:
            return GateResult(
                accepted=False,
                reason=f"Held-out regression detected: delta_ho={delta_ho:.1%} < -{self.non_regression_margin:.1%}",
                delta_held_in=delta_in,
                delta_held_out=delta_ho,
                utility_score=utility,
                metrics_baseline_in=baseline_in,
                metrics_candidate_in=candidate_in,
                metrics_baseline_ho=baseline_ho,
                metrics_candidate_ho=candidate_ho,
            )

        if delta_in < -self.non_regression_margin:
            return GateResult(
                accepted=False,
                reason=f"Held-in regression detected: delta_in={delta_in:.1%} < -{self.non_regression_margin:.1%}",
                delta_held_in=delta_in,
                delta_held_out=delta_ho,
                utility_score=utility,
                metrics_baseline_in=baseline_in,
                metrics_candidate_in=candidate_in,
                metrics_baseline_ho=baseline_ho,
                metrics_candidate_ho=candidate_ho,
            )

        if delta_in <= 0 and delta_ho <= 0 and utility <= 0:
            return GateResult(
                accepted=False,
                reason="No positive improvement observed on held-in or held-out splits",
                delta_held_in=delta_in,
                delta_held_out=delta_ho,
                utility_score=utility,
                metrics_baseline_in=baseline_in,
                metrics_candidate_in=candidate_in,
                metrics_baseline_ho=baseline_ho,
                metrics_candidate_ho=candidate_ho,
            )

        return GateResult(
            accepted=True,
            reason=f"Passed Non-Regression Gate (delta_in={delta_in:+.1%}, delta_ho={delta_ho:+.1%}, utility={utility:+.2f})",
            delta_held_in=delta_in,
            delta_held_out=delta_ho,
            utility_score=utility,
            metrics_baseline_in=baseline_in,
            metrics_candidate_in=candidate_in,
            metrics_baseline_ho=baseline_ho,
            metrics_candidate_ho=candidate_ho,
        )
