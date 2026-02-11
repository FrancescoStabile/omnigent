"""Tests for Cost Tracker."""

import pytest
from omnigent.cost_tracker import CostTracker, PRICING


class TestCostTracker:
    def test_initial_state(self):
        ct = CostTracker()
        assert ct.get_total_cost() == 0.0
        assert ct.get_total_tokens() == (0, 0)
        assert ct.tool_calls == 0

    def test_add_tokens(self):
        ct = CostTracker()
        ct.add_tokens("deepseek", 1000, 500)
        total_in, total_out = ct.get_total_tokens()
        assert total_in == 1000
        assert total_out == 500

    def test_cost_calculation(self):
        ct = CostTracker()
        ct.add_tokens("deepseek", 1_000_000, 1_000_000)
        cost = ct.get_provider_cost("deepseek")
        expected = PRICING["deepseek"].input_per_million + PRICING["deepseek"].output_per_million
        assert abs(cost - expected) < 0.01

    def test_multi_provider(self):
        ct = CostTracker()
        ct.add_tokens("deepseek", 100, 100)
        ct.add_tokens("claude", 100, 100)
        assert ct.get_total_cost() > 0
        # Claude is more expensive
        assert ct.get_provider_cost("claude") > ct.get_provider_cost("deepseek")

    def test_task_type_tracking(self):
        ct = CostTracker()
        ct.add_tokens("deepseek", 1000, 500, task_type="analysis")
        ct.add_tokens("deepseek", 2000, 1000, task_type="synthesis")
        assert "analysis" in ct.costs_by_task_type
        assert "synthesis" in ct.costs_by_task_type

    def test_tool_call_counter(self):
        ct = CostTracker()
        ct.add_tool_call()
        ct.add_tool_call()
        assert ct.tool_calls == 2

    def test_budget_check(self):
        ct = CostTracker(budget_limit=0.001)
        ct.add_tokens("claude", 1_000_000, 1_000_000)
        assert ct.is_over_budget() is True

    def test_budget_percentage(self):
        ct = CostTracker(budget_limit=10.0)
        ct.add_tokens("deepseek", 1_000_000, 1_000_000)
        pct = ct.get_budget_percentage()
        assert pct > 0
        assert pct < 100  # DeepSeek is cheap

    def test_ollama_is_free(self):
        ct = CostTracker()
        ct.add_tokens("ollama", 1_000_000, 1_000_000)
        assert ct.get_provider_cost("ollama") == 0.0

    def test_unknown_provider(self):
        ct = CostTracker()
        ct.add_tokens("unknown_provider", 1000, 1000)
        assert ct.get_provider_cost("unknown_provider") == 0.0

    def test_format_summary(self):
        ct = CostTracker()
        ct.add_tokens("deepseek", 1000, 500)
        ct.add_tool_call()
        summary = ct.format_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
