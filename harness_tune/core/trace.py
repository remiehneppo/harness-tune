"""Execution traces, step recordings, and outcome data structures."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCallRecord:
    tool_name: str
    arguments: Dict[str, Any]
    output: str
    status: str  # "success" | "error" | "warning"
    duration_ms: float = 0.0


@dataclass
class StepTrace:
    step_number: int
    thought: Optional[str] = None
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    tokens_used: int = 0
    duration_ms: float = 0.0


@dataclass
class ExecutionTrace:
    task_id: str
    harness_id: str
    model_name: str
    passed: bool
    steps: List[StepTrace] = field(default_factory=list)
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    error_summary: Optional[str] = None
    verifier_output: Optional[str] = None

    @property
    def total_tool_calls(self) -> int:
        return sum(len(step.tool_calls) for step in self.steps)

    @property
    def failed_tool_calls(self) -> List[ToolCallRecord]:
        fails = []
        for step in self.steps:
            for tc in step.tool_calls:
                if tc.status == "error":
                    fails.append(tc)
        return fails

    def detect_consecutive_same_errors(self) -> int:
        max_streak = 0
        current_streak = 0
        last_signature = None

        for step in self.steps:
            for tc in step.tool_calls:
                if tc.status == "error":
                    sig = (tc.tool_name, tc.output[:100])
                    if sig == last_signature:
                        current_streak += 1
                    else:
                        current_streak = 1
                        last_signature = sig
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
                    last_signature = None
        return max_streak
