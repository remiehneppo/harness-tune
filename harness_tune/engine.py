"""Harness Evolution Engine coordinating the end-to-end optimization loop."""

from __future__ import annotations
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from harness_tune.core.manifest import HarnessManifest
from harness_tune.core.trace import ExecutionTrace
from harness_tune.core.failure_miner import FailureMiner, EvidenceBundle
from harness_tune.core.proposer import DedicatedProposer, HarnessCandidate
from harness_tune.core.patcher import IsolatedPatcher
from harness_tune.core.regression_gate import RegressionGate, EvaluationMetrics, GateResult
from harness_tune.core.registry import HarnessRegistry
from harness_tune.agents.base import AbstractCodingAgent
from harness_tune.benchmarks.base import AbstractBenchmark, BenchmarkSplit


class HarnessEvolutionEngine:
    """End-to-end engine optimizing a coding agent's harness for a specific model."""

    def __init__(
        self,
        agent: AbstractCodingAgent,
        benchmark: AbstractBenchmark,
        proposer: DedicatedProposer,
        output_dir: Path | str,
        regression_gate: Optional[RegressionGate] = None,
        patcher: Optional[IsolatedPatcher] = None,
    ):
        self.agent = agent
        self.benchmark = benchmark
        self.proposer = proposer
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gate = regression_gate or RegressionGate()
        self.patcher = patcher or IsolatedPatcher()
        self.miner = FailureMiner()
        self.registry = HarnessRegistry(self.output_dir / ".registry")

    def run_optimization_cycle(
        self,
        baseline_manifest: HarnessManifest,
        max_proposals: int = 3,
    ) -> Dict[str, Any]:
        cycle_start = time.time()
        print(f"\n==================================================")
        print(f"🚀 Starting Harness Optimization Cycle")
        print(f"Target Agent: {self.agent.__class__.__name__}")
        print(f"Target Model: {baseline_manifest.model_profile.model_name}")
        print(f"Baseline Harness: {baseline_manifest.harness_id} ({baseline_manifest.version})")
        print(f"Proposer Model: {self.proposer.proposer_model_name}")
        print(f"Output Directory (Isolated): {self.output_dir}")
        print(f"==================================================\n")

        # 1. Obtain benchmark split (Held-in / Held-out)
        split: BenchmarkSplit = self.benchmark.get_split()
        scratch_dir = self.output_dir / "workspace_runs"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        # 2. Evaluate Baseline on Held-in & Held-out
        print("📊 [Phase 1/5] Evaluating Baseline Harness...")
        traces_baseline_in = self.benchmark.run_evaluation(
            agent=self.agent,
            harness=baseline_manifest,
            tasks=split.held_in,
            work_base_dir=str(scratch_dir / "baseline_held_in"),
        )
        traces_baseline_ho = self.benchmark.run_evaluation(
            agent=self.agent,
            harness=baseline_manifest,
            tasks=split.held_out,
            work_base_dir=str(scratch_dir / "baseline_held_out"),
        )

        metrics_baseline_in = EvaluationMetrics.from_traces(traces_baseline_in)
        metrics_baseline_ho = EvaluationMetrics.from_traces(traces_baseline_ho)
        print(f"   Baseline Held-in Pass:  {metrics_baseline_in.pass_rate:.1%}")
        print(f"   Baseline Held-out Pass: {metrics_baseline_ho.pass_rate:.1%}")

        # 3. Mine failure traces from Held-in split
        print("\n🔍 [Phase 2/5] Mining Failure Patterns from Traces...")
        evidence: EvidenceBundle = self.miner.analyze(traces_baseline_in)
        print(evidence.summary())

        if not evidence.clusters:
            print("✨ Baseline already achieved 100% on held-in split! No failures to optimize.")
            return {
                "status": "perfect_baseline",
                "promoted_manifest": baseline_manifest,
                "metrics_in": metrics_baseline_in,
                "metrics_ho": metrics_baseline_ho,
            }

        # 4. Propose bounded patches using Dedicated Proposer
        print("\n💡 [Phase 3/5] Dedicated Proposer Generating Candidates...")
        candidates: List[HarnessCandidate] = self.proposer.propose(
            manifest=baseline_manifest,
            evidence=evidence,
            max_proposals=max_proposals,
        )
        print(f"   Generated {len(candidates)} candidate patches.")

        # 5. Materialize and Evaluate Candidates in separate directories
        print("\n🧪 [Phase 4/5] Materializing and Screening Candidates...")
        accepted_candidates: List[tuple[HarnessCandidate, HarnessManifest, GateResult]] = []

        candidates_dir = self.output_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)

        for cand in candidates:
            print(f"\n   -> Candidate [{cand.candidate_id}]: {cand.hypothesis}")
            try:
                cand_manifest = self.patcher.materialize(
                    baseline_manifest=baseline_manifest,
                    candidate=cand,
                    output_base_dir=candidates_dir,
                )
            except Exception as ex:
                print(f"      ❌ Failed to materialize patch: {ex}")
                continue

            # Run candidate on held-in & held-out
            cand_runs_dir = scratch_dir / f"cand_{cand.candidate_id}"
            cand_traces_in = self.benchmark.run_evaluation(
                agent=self.agent,
                harness=cand_manifest,
                tasks=split.held_in,
                work_base_dir=str(cand_runs_dir / "held_in"),
            )
            cand_traces_ho = self.benchmark.run_evaluation(
                agent=self.agent,
                harness=cand_manifest,
                tasks=split.held_out,
                work_base_dir=str(cand_runs_dir / "held_out"),
            )

            metrics_cand_in = EvaluationMetrics.from_traces(cand_traces_in)
            metrics_cand_ho = EvaluationMetrics.from_traces(cand_traces_ho)

            # Check Non-Regression Gate
            gate_res: GateResult = self.gate.evaluate(
                baseline_in=metrics_baseline_in,
                candidate_in=metrics_cand_in,
                baseline_ho=metrics_baseline_ho,
                candidate_ho=metrics_cand_ho,
            )

            print(f"      Held-in:  {metrics_cand_in.pass_rate:.1%} (Δ {gate_res.delta_held_in:+.1%})")
            print(f"      Held-out: {metrics_cand_ho.pass_rate:.1%} (Δ {gate_res.delta_held_out:+.1%})")
            print(f"      Gate Decision: {'ACCEPTED' if gate_res.accepted else 'REJECTED'} ({gate_res.reason})")

            if gate_res.accepted:
                accepted_candidates.append((cand, cand_manifest, gate_res))

        # 6. Promotion Phase
        print("\n🏆 [Phase 5/5] Promotion & Lineage Update...")
        if not accepted_candidates:
            print("   ⚠️ No candidate passed the non-regression gate. Retaining baseline.")
            return {
                "status": "no_promotion",
                "promoted_manifest": baseline_manifest,
                "metrics_in": metrics_baseline_in,
                "metrics_ho": metrics_baseline_ho,
            }

        # Select candidate with highest utility score
        accepted_candidates.sort(key=lambda x: x[2].utility_score, reverse=True)
        best_cand, best_manifest, best_gate = accepted_candidates[0]

        # Promote to dedicated output directory
        promoted_dir = self.output_dir / "promoted_harness"
        if promoted_dir.exists():
            shutil.rmtree(promoted_dir)
        shutil.copytree(best_manifest.base_dir, promoted_dir)

        promoted_manifest = HarnessManifest.load(promoted_dir / "configs" / "harness.manifest.json")
        promoted_manifest.base_dir = promoted_dir

        self.registry.record_run(
            manifest=promoted_manifest,
            parent_version=baseline_manifest.version,
            in_pass=best_gate.metrics_candidate_in.pass_rate,
            ho_pass=best_gate.metrics_candidate_ho.pass_rate,
            utility=best_gate.utility_score,
            hypothesis=best_cand.hypothesis,
            promoted=True,
        )

        duration = time.time() - cycle_start
        print(f"\n🎉 SUCCESS! Promoted new harness to separate directory:")
        print(f"   Location: {promoted_dir}")
        print(f"   Held-in:  {metrics_baseline_in.pass_rate:.1%} -> {best_gate.metrics_candidate_in.pass_rate:.1%}")
        print(f"   Held-out: {metrics_baseline_ho.pass_rate:.1%} -> {best_gate.metrics_candidate_ho.pass_rate:.1%}")
        print(f"   Cycle duration: {duration:.1f}s")

        return {
            "status": "promoted",
            "candidate_id": best_cand.candidate_id,
            "promoted_manifest": promoted_manifest,
            "promoted_dir": str(promoted_dir),
            "delta_held_in": best_gate.delta_held_in,
            "delta_held_out": best_gate.delta_held_out,
            "utility_score": best_gate.utility_score,
        }
