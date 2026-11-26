"""Command-line interface for Harness Tuning & Optimization Tool."""

from __future__ import annotations
import argparse
import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from harness_tune.core.agy_client import AgyCliClient, DEFAULT_AGY_MODEL
from harness_tune.core.manifest import HarnessManifest
from harness_tune.core.proposer import DedicatedProposer
from harness_tune.core.regression_gate import EvaluationMetrics
from harness_tune.core.trace import ExecutionTrace
from harness_tune.agents.pi_agent import PiCodingAgent
from harness_tune.benchmarks.terminal_bench import TerminalBenchAdapter
from harness_tune.benchmarks.terminal_bench2_docker import (
    TerminalBench2DockerAdapter,
    trace_from_dict,
)
from harness_tune.benchmarks.swe_bench import SWEBenchAdapter
from harness_tune.engine import HarnessEvolutionEngine

BENCHMARK_CHOICES = ["terminal_bench", "terminal_bench_2_full", "swe_bench"]


def _add_shared_benchmark_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--benchmark",
        type=str,
        default="terminal_bench",
        choices=BENCHMARK_CHOICES,
        help="Benchmark adapter: terminal_bench (6 light tasks), terminal_bench_2_full (89 Docker tasks), or swe_bench",
    )
    p.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Custom JSON task file (terminal_bench) or local git repo path (terminal_bench_2_full)",
    )
    p.add_argument(
        "--docker-network",
        type=str,
        default="bridge",
        choices=["bridge", "host"],
        help="Docker network mode for terminal_bench_2_full (default: bridge)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of tasks to evaluate (for smoke testing terminal_bench_2_full)",
    )
    p.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Optional comma-separated list of task IDs to evaluate",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-task execution timeout in seconds (default: 300)",
    )
    p.add_argument(
        "--user-profile",
        action="store_true",
        default=False,
        help="Mount and load user's personal ~/.agents/skills and ~/.pi/agent packages (ponytail, codegraph, beads)",
    )
    p.add_argument(
        "--skills-dir",
        type=str,
        default=None,
        help="Optional override path for personal skills directory (default: ~/.agents/skills)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness-tune",
        description="Optimize AI Coding Agent Harnesses using empirical evolution algorithms.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: optimize
    opt_parser = subparsers.add_parser("optimize", help="Run harness optimization cycle.")
    opt_parser.add_argument("--manifest", type=str, required=True, help="Path to initial harness.manifest.json")
    opt_parser.add_argument("--agent", type=str, default="pi", choices=["pi"], help="Target coding agent (default: pi)")
    opt_parser.add_argument(
        "--agent-mode",
        type=str,
        default="real",
        choices=["real", "auto", "simulated"],
        help="Pi agent execution mode (default: real)",
    )
    opt_parser.add_argument("--target-model", type=str, default=None, help="Override target model name (e.g. arex-2)")
    opt_parser.add_argument("--provider", type=str, default=None, help="Override model provider (e.g. local)")
    opt_parser.add_argument("--thinking", type=str, default=None, help="Override Pi thinking level")
    opt_parser.add_argument(
        "--orchestrator",
        type=str,
        default="agy",
        choices=["agy", "heuristic"],
        help="Orchestrator backend for generating harness patches (default: agy)",
    )
    opt_parser.add_argument(
        "--proposer-model",
        type=str,
        default=DEFAULT_AGY_MODEL,
        help=f"Dedicated Proposer model via agy CLI (default: {DEFAULT_AGY_MODEL})",
    )
    opt_parser.add_argument(
        "--allow-heuristic-fallback",
        action="store_true",
        default=False,
        help="Fall back to heuristic proposer if agy orchestrator fails",
    )
    _add_shared_benchmark_args(opt_parser)
    opt_parser.add_argument(
        "--output-dir",
        type=str,
        default="./optimized_harnesses",
        help="Dedicated directory to save optimized harness versions without modifying baseline",
    )
    opt_parser.add_argument("--rounds", type=int, default=1, help="Number of optimization rounds (default: 1)")
    opt_parser.add_argument("--max-proposals", type=int, default=2, help="Max proposals per evolution round (default: 2)")
    opt_parser.add_argument(
        "--baseline-traces",
        type=str,
        default=None,
        help="Optional path to baseline_traces.json produced by `benchmark` to skip re-running baseline",
    )
    opt_parser.add_argument(
        "--fast-screen",
        action="store_true",
        default=False,
        help="Screen candidates on failed held-in tasks first before full held-in/held-out verification",
    )

    # Subcommand: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Evaluate harness against benchmark dataset.")
    bench_parser.add_argument("--manifest", type=str, required=True, help="Path to harness.manifest.json")
    bench_parser.add_argument("--agent", type=str, default="pi", choices=["pi"], help="Target coding agent (default: pi)")
    bench_parser.add_argument(
        "--agent-mode",
        type=str,
        default="real",
        choices=["real", "auto", "simulated"],
        help="Pi agent execution mode (default: real)",
    )
    bench_parser.add_argument("--target-model", type=str, default=None, help="Override target model name")
    bench_parser.add_argument("--provider", type=str, default=None, help="Override model provider (e.g. local)")
    bench_parser.add_argument("--thinking", type=str, default=None, help="Override Pi thinking level")
    _add_shared_benchmark_args(bench_parser)
    bench_parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_eval_tmp",
        help="Directory to store workspace logs, checkpoint.jsonl, and baseline_traces.json",
    )

    return parser


def _build_benchmark(args: argparse.Namespace):
    task_ids = [t.strip() for t in args.task_ids.split(",")] if getattr(args, "task_ids", None) else None
    if args.benchmark == "terminal_bench":
        return TerminalBenchAdapter(tasks_file=args.dataset_path, timeout_sec=args.timeout)
    if args.benchmark == "terminal_bench_2_full":
        return TerminalBench2DockerAdapter(
            dataset_dir=args.dataset_path,
            timeout_sec=args.timeout,
            docker_network=args.docker_network,
            limit=args.limit,
            task_ids=task_ids,
        )
    return SWEBenchAdapter()


def _load_precomputed_baseline(path_str: str) -> Dict[str, List[ExecutionTrace]]:
    raw = json.loads(Path(path_str).resolve().read_text(encoding="utf-8"))
    return {
        "held_in": [trace_from_dict(item) for item in raw.get("held_in", [])],
        "held_out": [trace_from_dict(item) for item in raw.get("held_out", [])],
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        print(f"Error: Manifest file not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = HarnessManifest.load(manifest_path)
    if getattr(args, "target_model", None):
        manifest.model_profile.model_name = args.target_model
    if getattr(args, "provider", None):
        manifest.model_profile.provider = args.provider
    if getattr(args, "thinking", None):
        manifest.model_profile.thinking_level = args.thinking

    agent = PiCodingAgent(
        mode=args.agent_mode,
        provider=manifest.model_profile.provider,
        thinking_level=args.thinking,
        use_user_profile=args.user_profile,
        user_skills_dir=args.skills_dir,
    )
    benchmark = _build_benchmark(args)

    if args.command == "optimize":
        llm_caller = None
        if args.orchestrator == "agy":
            agy_client = AgyCliClient(
                default_model=args.proposer_model,
                timeout_sec=max(120, args.timeout),
            )
            llm_caller = agy_client.as_llm_caller()

        proposer = DedicatedProposer(
            proposer_model_name=args.proposer_model,
            llm_caller=llm_caller,
            allow_heuristic_fallback=args.allow_heuristic_fallback,
        )
        engine = HarnessEvolutionEngine(
            agent=agent,
            benchmark=benchmark,
            proposer=proposer,
            output_dir=args.output_dir,
        )

        precomputed = (
            _load_precomputed_baseline(args.baseline_traces)
            if getattr(args, "baseline_traces", None)
            else None
        )
        result = engine.run_optimization_cycle(
            baseline_manifest=manifest,
            max_proposals=args.max_proposals,
            max_rounds=args.rounds,
            precomputed_baseline=precomputed,
            fast_screen_failed_only=args.fast_screen,
        )
        return 0 if result["status"] in ("promoted", "perfect_baseline") else 2

    elif args.command == "benchmark":
        split = benchmark.get_split()
        out_dir = Path(args.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = out_dir / "checkpoint.jsonl"

        print(
            f"Running benchmark '{args.benchmark}' on harness {manifest.harness_id} "
            f"(held_in={len(split.held_in)}, held_out={len(split.held_out)})...",
            flush=True,
        )

        sig = inspect.signature(benchmark.run_evaluation)
        def _eval(tasks, subdir):
            if not tasks:
                return []
            if "checkpoint_file" in sig.parameters:
                return benchmark.run_evaluation(
                    agent=agent,
                    harness=manifest,
                    tasks=tasks,
                    work_base_dir=str(out_dir / subdir),
                    checkpoint_file=ckpt_file,
                )
            return benchmark.run_evaluation(
                agent=agent,
                harness=manifest,
                tasks=tasks,
                work_base_dir=str(out_dir / subdir),
            )

        traces_in = _eval(split.held_in, "held_in")
        traces_ho = _eval(split.held_out, "held_out")
        all_traces = traces_in + traces_ho

        metrics_in = EvaluationMetrics.from_traces(traces_in)
        metrics_ho = EvaluationMetrics.from_traces(traces_ho)
        metrics_all = EvaluationMetrics.from_traces(all_traces)

        (out_dir / "baseline_traces.json").write_text(
            json.dumps(
                {
                    "held_in": [asdict(t) for t in traces_in],
                    "held_out": [asdict(t) for t in traces_ho],
                    "metrics_in": asdict(metrics_in),
                    "metrics_ho": asdict(metrics_ho),
                    "metrics_overall": asdict(metrics_all),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (out_dir / "summary.json").write_text(
            json.dumps(
                {
                    "harness_id": manifest.harness_id,
                    "benchmark": args.benchmark,
                    "overall_pass_rate": metrics_all.pass_rate,
                    "passed_runs": metrics_all.passed_runs,
                    "total_runs": metrics_all.total_runs,
                    "held_in": asdict(metrics_in),
                    "held_out": asdict(metrics_ho),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"\nResults: {metrics_all.passed_runs}/{metrics_all.total_runs} passed "
            f"({metrics_all.pass_rate:.1%}) | "
            f"Held-in: {metrics_in.passed_runs}/{metrics_in.total_runs} ({metrics_in.pass_rate:.1%}) | "
            f"Held-out: {metrics_ho.passed_runs}/{metrics_ho.total_runs} ({metrics_ho.pass_rate:.1%})"
        )
        print(f"Saved traces & checkpoint to: {out_dir}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
