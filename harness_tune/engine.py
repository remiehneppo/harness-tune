"""Harness Evolution Engine coordinating the end-to-end optimization loop."""

from __future__ import annotations
import inspect
import json
import shutil
import time
import yaml
from dataclasses import asdict
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
from harness_tune.benchmarks.base import AbstractBenchmark, BenchmarkSplit, BenchmarkTask


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
        self._explicit_gate = regression_gate
        self.gate = regression_gate or RegressionGate()
        self.patcher = patcher or IsolatedPatcher()
        self.miner = FailureMiner()
        self.registry = HarnessRegistry(self.output_dir / ".registry")

    @staticmethod
    def _load_gate_from_manifest(manifest: HarnessManifest) -> Optional[RegressionGate]:
        rel = manifest.surfaces.get("verification_policy")
        if not rel or not manifest.base_dir:
            return None
        rel_path = rel.get("path") if isinstance(rel, dict) else str(rel)
        vp_path = manifest.base_dir / rel_path
        if not vp_path.exists():
            return None
        try:
            cfg = yaml.safe_load(vp_path.read_text(encoding="utf-8")) or {}
            margin = float(cfg.get("non_regression_margin", 0.0))
            return RegressionGate(non_regression_margin=margin)
        except Exception:
            return None

    @staticmethod
    def _serialize_traces(traces: List[ExecutionTrace]) -> List[Dict[str, Any]]:
        return [asdict(t) for t in traces]

    @staticmethod
    def _serialize_metrics(m: EvaluationMetrics) -> Dict[str, Any]:
        return asdict(m)

    @staticmethod
    def _reload_manifest(manifest_dir: Path) -> HarnessManifest:
        configs_manifest = manifest_dir / "configs" / "harness.manifest.json"
        if configs_manifest.exists():
            loaded = HarnessManifest.load(configs_manifest)
        else:
            loaded = HarnessManifest.load(manifest_dir / "harness.manifest.json")
        loaded.base_dir = manifest_dir
        return loaded

    def _run_benchmark_eval(
        self,
        harness: HarnessManifest,
        tasks: List[BenchmarkTask],
        work_base_dir: Path,
        checkpoint_file: Optional[Path] = None,
    ) -> List[ExecutionTrace]:
        if not tasks:
            return []
        sig = inspect.signature(self.benchmark.run_evaluation)
        if "checkpoint_file" in sig.parameters:
            return self.benchmark.run_evaluation(
                agent=self.agent,
                harness=harness,
                tasks=tasks,
                work_base_dir=str(work_base_dir),
                checkpoint_file=checkpoint_file,
            )
        return self.benchmark.run_evaluation(
            agent=self.agent,
            harness=harness,
            tasks=tasks,
            work_base_dir=str(work_base_dir),
        )

    def run_optimization_cycle(
        self,
        baseline_manifest: HarnessManifest,
        max_proposals: int = 3,
        max_rounds: int = 1,
        precomputed_baseline: Optional[Dict[str, List[ExecutionTrace]]] = None,
        fast_screen_failed_only: bool = False,
    ) -> Dict[str, Any]:
        cycle_start = time.time()
        if self._explicit_gate is None:
            manifest_gate = self._load_gate_from_manifest(baseline_manifest)
            if manifest_gate is not None:
                self.gate = manifest_gate

        print("\n==================================================")
        print("🚀 Starting Harness Optimization Cycle")
        print(f"Target Agent: {self.agent.__class__.__name__}")
        print(f"Target Model: {baseline_manifest.model_profile.model_name}")
        print(f"Baseline Harness: {baseline_manifest.harness_id} ({baseline_manifest.version})")
        print(f"Proposer Model: {self.proposer.proposer_model_name}")
        print(f"Rounds: {max_rounds} | Max Proposals/Round: {max_proposals}")
        print(f"Output Directory (Isolated): {self.output_dir}")
        print("==================================================\n")

        split: BenchmarkSplit = self.benchmark.get_split()
        scratch_dir = self.output_dir / "workspace_runs"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # 1. Obtain or evaluate Baseline on Held-in & Held-out
        print("📊 [Phase 1/5] Evaluating Baseline Harness...")
        if precomputed_baseline is not None:
            print("   ♻️  Reusing precomputed baseline traces.")
            traces_baseline_in = list(precomputed_baseline.get("held_in", []))
            traces_baseline_ho = list(precomputed_baseline.get("held_out", []))
        else:
            traces_baseline_in = self._run_benchmark_eval(
                harness=baseline_manifest,
                tasks=split.held_in,
                work_base_dir=scratch_dir / "baseline_held_in",
                checkpoint_file=scratch_dir / "baseline_held_in" / "checkpoint.jsonl",
            )
            traces_baseline_ho = self._run_benchmark_eval(
                harness=baseline_manifest,
                tasks=split.held_out,
                work_base_dir=scratch_dir / "baseline_held_out",
                checkpoint_file=scratch_dir / "baseline_held_out" / "checkpoint.jsonl",
            )

        metrics_baseline_in = EvaluationMetrics.from_traces(traces_baseline_in)
        metrics_baseline_ho = EvaluationMetrics.from_traces(traces_baseline_ho)
        print(f"   Baseline Held-in Pass:  {metrics_baseline_in.pass_rate:.1%} ({metrics_baseline_in.passed_runs}/{metrics_baseline_in.total_runs})")
        print(f"   Baseline Held-out Pass: {metrics_baseline_ho.pass_rate:.1%} ({metrics_baseline_ho.passed_runs}/{metrics_baseline_ho.total_runs})")

        (reports_dir / "baseline_traces.json").write_text(
            json.dumps(
                {
                    "held_in": self._serialize_traces(traces_baseline_in),
                    "held_out": self._serialize_traces(traces_baseline_ho),
                    "metrics_in": self._serialize_metrics(metrics_baseline_in),
                    "metrics_ho": self._serialize_metrics(metrics_baseline_ho),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        current_manifest = baseline_manifest
        current_traces_in = traces_baseline_in
        current_metrics_in = metrics_baseline_in
        current_metrics_ho = metrics_baseline_ho

        round_history: List[Dict[str, Any]] = []
        any_promoted = False
        best_overall_cand: Optional[HarnessCandidate] = None
        best_overall_gate: Optional[GateResult] = None

        candidates_dir = self.output_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        promoted_dir = self.output_dir / "promoted_harness"

        held_in_by_id = {t.task_id: t for t in split.held_in}

        for round_idx in range(1, max(1, max_rounds) + 1):
            round_label = f"[Round {round_idx}/{max_rounds}] " if max_rounds > 1 else ""

            print(f"\n🔍 [Phase 2/5] {round_label}Mining Failure Patterns from Traces...")
            evidence: EvidenceBundle = self.miner.analyze(current_traces_in)
            print(evidence.summary())

            if not evidence.clusters:
                if round_idx == 1:
                    print("✨ Baseline already achieved 100% on held-in split! No failures to optimize.")
                    summary_payload = {
                        "status": "perfect_baseline",
                        "baseline_metrics_in": self._serialize_metrics(metrics_baseline_in),
                        "baseline_metrics_ho": self._serialize_metrics(metrics_baseline_ho),
                        "rounds": round_history,
                        "duration_seconds": round(time.time() - cycle_start, 2),
                    }
                    (reports_dir / "summary.json").write_text(
                        json.dumps(summary_payload, indent=2), encoding="utf-8"
                    )
                    return {
                        "status": "perfect_baseline",
                        "promoted_manifest": baseline_manifest,
                        "metrics_in": metrics_baseline_in,
                        "metrics_ho": metrics_baseline_ho,
                    }
                print("✨ Current promoted harness achieved 100% on held-in split! Stopping early.")
                break

            print(f"\n💡 [Phase 3/5] {round_label}Dedicated Proposer Generating Candidates...")
            candidates: List[HarnessCandidate] = self.proposer.propose(
                manifest=current_manifest,
                evidence=evidence,
                max_proposals=max_proposals,
                traces=current_traces_in,
                history=round_history,
            )
            print(f"   Generated {len(candidates)} candidate patches.")
            loc_meta = getattr(self.proposer, "last_localization", None)
            if loc_meta:
                print(
                    f"   Fault Localization: suspect_surfaces={loc_meta.get('suspect_surfaces')}, "
                    f"suspect_sections={loc_meta.get('suspect_sections')}, "
                    f"context_reduction={loc_meta.get('reduction_pct')}%"
                )
            if self.proposer.last_diagnosis:
                print(f"   Orchestrator Diagnosis: {self.proposer.last_diagnosis}")

            (reports_dir / f"round_{round_idx}_proposer.json").write_text(
                json.dumps(
                    {
                        "round": round_idx,
                        "proposer_model": self.proposer.proposer_model_name,
                        "localization": loc_meta,
                        "diagnosis": self.proposer.last_diagnosis,
                        "candidates": [asdict(c) for c in candidates],
                        "raw_response": self.proposer.last_raw_response,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(f"\n🧪 [Phase 4/5] {round_label}Materializing and Screening Candidates...")
            accepted_in_round: List[
                tuple[HarnessCandidate, HarnessManifest, GateResult, List[ExecutionTrace]]
            ] = []
            round_results: List[Dict[str, Any]] = []

            for cand in candidates:
                print(f"\n   -> Candidate [{cand.candidate_id}]: {cand.hypothesis}")
                try:
                    cand_manifest = self.patcher.materialize(
                        baseline_manifest=current_manifest,
                        candidate=cand,
                        output_base_dir=candidates_dir,
                    )
                except Exception as ex:
                    print(f"      ❌ Failed to materialize patch: {ex}")
                    entry = {
                        "round": round_idx,
                        "candidate_id": cand.candidate_id,
                        "hypothesis": cand.hypothesis,
                        "patches": [asdict(p) for p in cand.patches],
                        "status": "materialize_error",
                        "error": str(ex),
                    }
                    round_results.append(entry)
                    round_history.append(entry)
                    continue

                cand_runs_dir = scratch_dir / f"round_{round_idx}_cand_{cand.candidate_id}"
                in_ckpt = cand_runs_dir / "held_in" / "checkpoint.jsonl"
                ho_ckpt = cand_runs_dir / "held_out" / "checkpoint.jsonl"

                # Optional 2-stage fast screening on failed held-in tasks first
                if fast_screen_failed_only:
                    failed_in_tasks = [
                        held_in_by_id[tr.task_id]
                        for tr in current_traces_in
                        if not tr.passed and tr.task_id in held_in_by_id
                    ]
                    if failed_in_tasks and len(failed_in_tasks) < len(split.held_in):
                        print(f"      ⚡ Stage A Fast Screen on {len(failed_in_tasks)} failed held-in tasks...")
                        screen_traces = self._run_benchmark_eval(
                            harness=cand_manifest,
                            tasks=failed_in_tasks,
                            work_base_dir=cand_runs_dir / "held_in",
                            checkpoint_file=in_ckpt,
                        )
                        fixed_count = sum(1 for tr in screen_traces if tr.passed)
                        if fixed_count == 0:
                            reason = (
                                f"Fast-screen rejected: fixed 0/{len(failed_in_tasks)} failed held-in tasks"
                            )
                            print(f"      Gate Decision: REJECTED ({reason})")
                            self.registry.record_run(
                                manifest=cand_manifest,
                                parent_version=current_manifest.version,
                                in_pass=current_metrics_in.pass_rate,
                                ho_pass=current_metrics_ho.pass_rate,
                                utility=-1.0,
                                hypothesis=cand.hypothesis,
                                promoted=False,
                            )
                            entry = {
                                "round": round_idx,
                                "candidate_id": cand.candidate_id,
                                "hypothesis": cand.hypothesis,
                                "patches": [asdict(p) for p in cand.patches],
                                "accepted": False,
                                "reason": reason,
                                "delta_held_in": 0.0,
                                "delta_held_out": 0.0,
                                "utility_score": -1.0,
                                "fast_screen_traces": self._serialize_traces(screen_traces),
                            }
                            round_results.append(entry)
                            round_history.append(
                                {
                                    "round": round_idx,
                                    "candidate_id": cand.candidate_id,
                                    "hypothesis": cand.hypothesis,
                                    "patches": [asdict(p) for p in cand.patches],
                                    "accepted": False,
                                    "reason": reason,
                                    "delta_held_in": 0.0,
                                    "delta_held_out": 0.0,
                                }
                            )
                            continue
                        print(
                            f"      ✅ Stage A fixed {fixed_count}/{len(failed_in_tasks)} failed tasks! "
                            "Proceeding to full held-in & held-out verification (reusing Stage A checkpoint)."
                        )

                cand_traces_in = self._run_benchmark_eval(
                    harness=cand_manifest,
                    tasks=split.held_in,
                    work_base_dir=cand_runs_dir / "held_in",
                    checkpoint_file=in_ckpt,
                )
                cand_traces_ho = self._run_benchmark_eval(
                    harness=cand_manifest,
                    tasks=split.held_out,
                    work_base_dir=cand_runs_dir / "held_out",
                    checkpoint_file=ho_ckpt,
                )

                metrics_cand_in = EvaluationMetrics.from_traces(cand_traces_in)
                metrics_cand_ho = EvaluationMetrics.from_traces(cand_traces_ho)

                gate_res: GateResult = self.gate.evaluate(
                    baseline_in=current_metrics_in,
                    candidate_in=metrics_cand_in,
                    baseline_ho=current_metrics_ho,
                    candidate_ho=metrics_cand_ho,
                )

                print(f"      Held-in:  {metrics_cand_in.pass_rate:.1%} (Δ {gate_res.delta_held_in:+.1%})")
                print(f"      Held-out: {metrics_cand_ho.pass_rate:.1%} (Δ {gate_res.delta_held_out:+.1%})")
                print(f"      Gate Decision: {'ACCEPTED' if gate_res.accepted else 'REJECTED'} ({gate_res.reason})")

                self.registry.record_run(
                    manifest=cand_manifest,
                    parent_version=current_manifest.version,
                    in_pass=metrics_cand_in.pass_rate,
                    ho_pass=metrics_cand_ho.pass_rate,
                    utility=gate_res.utility_score,
                    hypothesis=cand.hypothesis,
                    promoted=gate_res.accepted,
                )

                entry = {
                    "round": round_idx,
                    "candidate_id": cand.candidate_id,
                    "hypothesis": cand.hypothesis,
                    "patches": [asdict(p) for p in cand.patches],
                    "accepted": gate_res.accepted,
                    "reason": gate_res.reason,
                    "delta_held_in": gate_res.delta_held_in,
                    "delta_held_out": gate_res.delta_held_out,
                    "utility_score": gate_res.utility_score,
                    "metrics_in": self._serialize_metrics(metrics_cand_in),
                    "metrics_ho": self._serialize_metrics(metrics_cand_ho),
                    "traces_in": self._serialize_traces(cand_traces_in),
                    "traces_ho": self._serialize_traces(cand_traces_ho),
                }
                round_results.append(entry)
                round_history.append(
                    {
                        "round": round_idx,
                        "candidate_id": cand.candidate_id,
                        "hypothesis": cand.hypothesis,
                        "patches": [asdict(p) for p in cand.patches],
                        "accepted": gate_res.accepted,
                        "reason": gate_res.reason,
                        "delta_held_in": gate_res.delta_held_in,
                        "delta_held_out": gate_res.delta_held_out,
                    }
                )

                if gate_res.accepted:
                    accepted_in_round.append((cand, cand_manifest, gate_res, cand_traces_in))

            (reports_dir / f"round_{round_idx}_results.json").write_text(
                json.dumps(round_results, indent=2), encoding="utf-8"
            )

            if accepted_in_round:
                accepted_in_round.sort(key=lambda x: x[2].utility_score, reverse=True)
                best_cand, best_manifest, best_gate, best_traces_in = accepted_in_round[0]
                any_promoted = True
                best_overall_cand = best_cand
                best_overall_gate = best_gate

                if promoted_dir.exists():
                    shutil.rmtree(promoted_dir)
                assert best_manifest.base_dir is not None
                shutil.copytree(best_manifest.base_dir, promoted_dir)

                current_manifest = self._reload_manifest(promoted_dir)
                current_traces_in = best_traces_in
                current_metrics_in = best_gate.metrics_candidate_in
                current_metrics_ho = best_gate.metrics_candidate_ho

        print("\n🏆 [Phase 5/5] Promotion & Lineage Update...")
        if not any_promoted or best_overall_cand is None or best_overall_gate is None:
            print("   ⚠️ No candidate passed the non-regression gate. Retaining baseline.")
            summary_payload = {
                "status": "no_promotion",
                "baseline_metrics_in": self._serialize_metrics(metrics_baseline_in),
                "baseline_metrics_ho": self._serialize_metrics(metrics_baseline_ho),
                "rounds": round_history,
                "duration_seconds": round(time.time() - cycle_start, 2),
            }
            (reports_dir / "summary.json").write_text(
                json.dumps(summary_payload, indent=2), encoding="utf-8"
            )
            return {
                "status": "no_promotion",
                "promoted_manifest": baseline_manifest,
                "metrics_in": metrics_baseline_in,
                "metrics_ho": metrics_baseline_ho,
            }

        promoted_manifest = current_manifest
        duration = time.time() - cycle_start
        overall_delta_in = current_metrics_in.pass_rate - metrics_baseline_in.pass_rate
        overall_delta_ho = current_metrics_ho.pass_rate - metrics_baseline_ho.pass_rate

        summary_payload = {
            "status": "promoted",
            "candidate_id": best_overall_cand.candidate_id,
            "promoted_dir": str(promoted_dir),
            "baseline_metrics_in": self._serialize_metrics(metrics_baseline_in),
            "baseline_metrics_ho": self._serialize_metrics(metrics_baseline_ho),
            "promoted_metrics_in": self._serialize_metrics(current_metrics_in),
            "promoted_metrics_ho": self._serialize_metrics(current_metrics_ho),
            "delta_held_in": overall_delta_in,
            "delta_held_out": overall_delta_ho,
            "utility_score": best_overall_gate.utility_score,
            "rounds": round_history,
            "duration_seconds": round(duration, 2),
        }
        (reports_dir / "summary.json").write_text(
            json.dumps(summary_payload, indent=2), encoding="utf-8"
        )

        print("\n🎉 SUCCESS! Promoted new harness to separate directory:")
        print(f"   Location: {promoted_dir}")
        print(f"   Held-in:  {metrics_baseline_in.pass_rate:.1%} -> {current_metrics_in.pass_rate:.1%}")
        print(f"   Held-out: {metrics_baseline_ho.pass_rate:.1%} -> {current_metrics_ho.pass_rate:.1%}")
        print(f"   Cycle duration: {duration:.1f}s")

        return {
            "status": "promoted",
            "candidate_id": best_overall_cand.candidate_id,
            "promoted_manifest": promoted_manifest,
            "promoted_dir": str(promoted_dir),
            "delta_held_in": overall_delta_in,
            "delta_held_out": overall_delta_ho,
            "utility_score": best_overall_gate.utility_score,
        }
