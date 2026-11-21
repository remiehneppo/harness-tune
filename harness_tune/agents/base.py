"""Abstract coding agent interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from harness_tune.core.manifest import HarnessManifest
from harness_tune.core.trace import ExecutionTrace


class AbstractCodingAgent(ABC):
    """Abstract interface for any coding agent being evaluated and tuned."""

    @abstractmethod
    def run_task(
        self,
        task_id: str,
        instruction: str,
        workspace_dir: str,
        harness: HarnessManifest,
        timeout_sec: int = 120,
    ) -> ExecutionTrace:
        """Executes the agent on the given task and returns a detailed execution trace."""
        pass
