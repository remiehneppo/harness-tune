"""Unit tests for FailureMiner."""

import unittest
from harness_tune.core.trace import ExecutionTrace, StepTrace, ToolCallRecord
from harness_tune.core.failure_miner import FailureMiner


class TestFailureMiner(unittest.TestCase):
    def test_detect_tool_loop(self):
        trace = ExecutionTrace(
            task_id="task_01",
            harness_id="pi-v1",
            model_name="qwen-35b",
            passed=False,
            steps=[
                StepTrace(
                    step_number=1,
                    tool_calls=[
                        ToolCallRecord("bash", {"cmd": "foo"}, "error: not found", "error"),
                    ],
                ),
                StepTrace(
                    step_number=2,
                    tool_calls=[
                        ToolCallRecord("bash", {"cmd": "foo"}, "error: not found", "error"),
                    ],
                ),
            ],
        )

        miner = FailureMiner()
        evidence = miner.analyze([trace])
        self.assertEqual(evidence.total_runs, 1)
        self.assertEqual(evidence.failed_runs, 1)
        self.assertTrue(any(c.cluster_id == "FC_TOOL_LOOP" for c in evidence.clusters))

    def test_detect_dependency_error(self):
        trace = ExecutionTrace(
            task_id="task_02",
            harness_id="pi-v1",
            model_name="qwen-35b",
            passed=False,
            steps=[
                StepTrace(
                    step_number=1,
                    tool_calls=[
                        ToolCallRecord("bash", {"cmd": "pkg"}, "bash: pkg: command not found", "error"),
                    ],
                ),
            ],
        )

        miner = FailureMiner()
        evidence = miner.analyze([trace])
        self.assertTrue(any(c.cluster_id == "FC_MISSING_DEP" for c in evidence.clusters))


if __name__ == "__main__":
    unittest.main()
