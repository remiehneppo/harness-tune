"""Command-line interface for Harness Tuning & Optimization Tool."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

from harness_tune.core.manifest import HarnessManifest, ModelProfile
from harness_tune.core.proposer import DedicatedProposer
from harness_tune.agents.pi_agent import PiCodingAgent
from harness_tune.benchmarks.terminal_bench import TerminalBenchAdapter
from harness_tune.benchmarks.swe_bench import SWEBenchAdapter
from harness_tune.engine import HarnessEvolutionEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness-tune",
        description="Optimize AI Coding Agent Harnesses using empirical evolution algorithms.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: optimize
    opt_parser = subparsers.add_parser("optimize", help="Run harness optimization cycle.")
    opt_parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to initial harness.manifest.json",
    )
    opt_parser.add_argument(
        "--agent",
        type=str,
        default="pi",
        choices=["pi"],
        help="Target coding agent (default: pi)",
    )
    opt_parser.add_argument(
        "--target-model",
        type=str,
        default=None,
        help="Override target model name (e.g. arex-2, qwen-2.5-coder-32b)",
    )
    opt_parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Override model provider (e.g. local, anthropic, google)",
    )
    opt_parser.add_argument(
        "--proposer-model",
        type=str,
        default="claude-3-7-sonnet",
        help="Dedicated high-capacity Proposer model (default: claude-3-7-sonnet)",
    )
    opt_parser.add_argument(
        "--benchmark",
        type=str,
        default="terminal_bench",
        choices=["terminal_bench", "swe_bench"],
        help="Benchmark dataset adapter (terminal_bench or swe_bench)",
    )
    opt_parser.add_argument(
        "--output-dir",
        type=str,
        default="./optimized_harnesses",
        help="Dedicated directory to save optimized harness versions without modifying baseline",
    )
    opt_parser.add_argument(
        "--max-proposals",
        type=int,
        default=3,
        help="Max proposals per evolution cycle (default: 3)",
    )

    # Subcommand: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Evaluate harness against benchmark dataset.")
    bench_parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to harness.manifest.json",
    )
    bench_parser.add_argument(
        "--agent",
        type=str,
        default="pi",
        choices=["pi"],
        help="Target coding agent (default: pi)",
    )
    bench_parser.add_argument(
        "--target-model",
        type=str,
        default=None,
        help="Override target model name",
    )
    bench_parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Override model provider (e.g. local)",
    )
    bench_parser.add_argument(
        "--benchmark",
        type=str,
        default="terminal_bench",
        choices=["terminal_bench", "swe_bench"],
        help="Benchmark dataset adapter",
    )

    return parser


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

    # Instantiate Agent
    agent = PiCodingAgent()

    # Instantiate Benchmark
    if args.benchmark == "terminal_bench":
        benchmark = TerminalBenchAdapter()
    else:
        benchmark = SWEBenchAdapter()

    if args.command == "optimize":
        proposer = DedicatedProposer(proposer_model_name=args.proposer_model)
        engine = HarnessEvolutionEngine(
            agent=agent,
            benchmark=benchmark,
            proposer=proposer,
            output_dir=args.output_dir,
        )

        result = engine.run_optimization_cycle(
            baseline_manifest=manifest,
            max_proposals=args.max_proposals,
        )
        return 0 if result["status"] == "promoted" else 2

    elif args.command == "benchmark":
        split = benchmark.get_split()
        scratch_dir = Path("./benchmark_eval_tmp")
        scratch_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running benchmark '{args.benchmark}' on harness {manifest.harness_id}...")
        traces = benchmark.run_evaluation(
            agent=agent,
            harness=manifest,
            tasks=split.held_in + split.held_out,
            work_base_dir=str(scratch_dir),
        )
        passed = sum(1 for t in traces if t.passed)
        print(f"Results: {passed}/{len(traces)} passed ({passed / len(traces):.1%})")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
