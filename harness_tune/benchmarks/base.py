"""Abstract Benchmark interface and dataset split abstractions."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from harness_tune.core.trace import ExecutionTrace
from harness_tune.core.manifest import HarnessManifest
from harness_tune.agents.base import AbstractCodingAgent


@dataclass
class BenchmarkTask:
    task_id: str
    dataset_name: str
    instruction: str
    workspace_template: Optional[str] = None
    verification_script: Optional[str] = None
    setup_files: Dict[str, str] = field(default_factory=dict)
    setup_script: Optional[str] = None
    timeout_sec: int = 180
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSplit:
    held_in: List[BenchmarkTask]
    held_out: List[BenchmarkTask]
    untouched_audit: List[BenchmarkTask] = field(default_factory=list)


class AbstractBenchmark(ABC):
    """Abstract interface for standardized agent benchmarks (Terminal-Bench-2.0, SWE-bench)."""

    @abstractmethod
    def get_split(self) -> BenchmarkSplit:
        """Returns the benchmark split (held-in, held-out, audit)."""
        pass

    @abstractmethod
    def run_evaluation(
        self,
        agent: AbstractCodingAgent,
        harness: HarnessManifest,
        tasks: List[BenchmarkTask],
        work_base_dir: str,
    ) -> List[ExecutionTrace]:
        """Runs the agent across given tasks and collects execution traces."""
        pass
