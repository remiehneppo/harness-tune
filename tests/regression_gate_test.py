"""Unit tests for RegressionGate."""

import unittest
from harness_tune.core.regression_gate import RegressionGate, EvaluationMetrics


class TestRegressionGate(unittest.TestCase):
    def setUp(self):
        self.gate = RegressionGate(non_regression_margin=0.0)

    def test_accept_improved_candidate(self):
        base_in = EvaluationMetrics(pass_rate=0.40, total_runs=10, passed_runs=4, avg_tokens=1000, avg_latency_ms=500, total_cost_usd=0.01)
        base_ho = EvaluationMetrics(pass_rate=0.40, total_runs=10, passed_runs=4, avg_tokens=1000, avg_latency_ms=500, total_cost_usd=0.01)

        cand_in = EvaluationMetrics(pass_rate=0.60, total_runs=10, passed_runs=6, avg_tokens=1000, avg_latency_ms=500, total_cost_usd=0.01)
        cand_ho = EvaluationMetrics(pass_rate=0.50, total_runs=10, passed_runs=5, avg_tokens=1000, avg_latency_ms=500, total_cost_usd=0.01)

        res = self.gate.evaluate(base_in, cand_in, base_ho, cand_ho)
        self.assertTrue(res.accepted)
        self.assertAlmostEqual(res.delta_held_in, 0.20)
        self.assertAlmostEqual(res.delta_held_out, 0.10)

    def test_reject_held_out_regression(self):
        base_in = EvaluationMetrics(pass_rate=0.50, total_runs=10, passed_runs=5, avg_tokens=1000, avg_latency_ms=500, total_cost_usd=0.01)
        base_ho = EvaluationMetrics(pass_rate=0.50, total_runs=10, passed_runs=5, avg_tokens=1000, avg_latency_ms=500, total_cost_usd=0.01)

        cand_in = EvaluationMetrics(pass_rate=0.80, total_runs=10, passed_runs=8, avg_tokens=1000, avg_latency_ms=500, total_cost_usd=0.01)
        cand_ho = EvaluationMetrics(pass_rate=0.40, total_runs=10, passed_runs=4, avg_tokens=1000, avg_latency_ms=500, total_cost_usd=0.01)

        res = self.gate.evaluate(base_in, cand_in, base_ho, cand_ho)
        self.assertFalse(res.accepted)
        self.assertIn("Held-out regression detected", res.reason)


if __name__ == "__main__":
    unittest.main()
