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


class TestCacheTokenTracking:
    """Tests for cache token support (cache_read, cache_creation)."""

    def test_cache_tokens_stored(self):
        ct = CostTracker()
        ct.add_tokens("claude", 1000, 500, cache_read_tokens=200, cache_creation_tokens=50)
        tokens = ct.tokens_by_provider["claude"]
        assert tokens["cache_read"] == 200
        assert tokens["cache_creation"] == 50

    def test_cache_tokens_default_zero(self):
        ct = CostTracker()
        ct.add_tokens("claude", 1000, 500)
        tokens = ct.tokens_by_provider["claude"]
        assert tokens["cache_read"] == 0
        assert tokens["cache_creation"] == 0

    def test_cache_tokens_accumulate(self):
        ct = CostTracker()
        ct.add_tokens("claude", 100, 50, cache_read_tokens=10, cache_creation_tokens=5)
        ct.add_tokens("claude", 100, 50, cache_read_tokens=20, cache_creation_tokens=10)
        tokens = ct.tokens_by_provider["claude"]
        assert tokens["cache_read"] == 30
        assert tokens["cache_creation"] == 15

    def test_cache_cost_included(self):
        ct = CostTracker()
        # Cost without cache
        ct_no_cache = CostTracker()
        ct_no_cache.add_tokens("claude", 1_000_000, 0)
        cost_no_cache = ct_no_cache.get_provider_cost("claude")

        # Cost with cache read tokens
        ct.add_tokens("claude", 1_000_000, 0, cache_read_tokens=1_000_000)
        cost_with_cache = ct.get_provider_cost("claude")

        assert cost_with_cache > cost_no_cache

    def test_cache_read_pricing_correct(self):
        ct = CostTracker()
        ct.add_tokens("claude", 0, 0, cache_read_tokens=1_000_000)
        cost = ct.get_provider_cost("claude")
        # cache_read_per_million for claude = 0.30
        assert abs(cost - 0.30) < 0.001

    def test_cache_creation_pricing_correct(self):
        ct = CostTracker()
        ct.add_tokens("claude", 0, 0, cache_creation_tokens=1_000_000)
        cost = ct.get_provider_cost("claude")
        # cache_creation_per_million for claude = 3.75
        assert abs(cost - 3.75) < 0.001

    def test_cache_info_in_format_summary(self):
        ct = CostTracker()
        ct.add_tokens("claude", 1000, 500, cache_read_tokens=200, cache_creation_tokens=50)
        summary = ct.format_summary()
        assert "cache" in summary
        assert "200" in summary
        assert "50" in summary

    def test_provider_pricing_has_cache_fields(self):
        """Claude pricing should include cache fields."""
        assert PRICING["claude"].cache_read_per_million == 0.30
        assert PRICING["claude"].cache_creation_per_million == 3.75

    def test_deepseek_no_cache_pricing(self):
        """DeepSeek pricing should have cache fields at 0."""
        assert PRICING["deepseek"].cache_read_per_million == 0.0
        assert PRICING["deepseek"].cache_creation_per_million == 0.0
