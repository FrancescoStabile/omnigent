"""
Omnigent — Cost Tracking System

Real-time token and cost tracking across LLM providers.
Display in CLI, warn on budget limits. Tracks per-task-type costs.

This module is 100% domain-agnostic.
"""

from dataclasses import dataclass


@dataclass
class ProviderPricing:
    """Pricing per token for a provider."""
    input_per_million: float
    output_per_million: float
    cache_read_per_million: float = 0.0
    cache_creation_per_million: float = 0.0


# Provider pricing (as of Feb 2026)
PRICING = {
    "deepseek": ProviderPricing(input_per_million=0.14, output_per_million=0.28),
    "claude": ProviderPricing(
        input_per_million=3.00, output_per_million=15.00,
        cache_read_per_million=0.30, cache_creation_per_million=3.75,
    ),
    "openai": ProviderPricing(input_per_million=0.15, output_per_million=0.60),
    "ollama": ProviderPricing(input_per_million=0.0, output_per_million=0.0),
}


class CostTracker:
    """
    Track costs across providers in real-time.

    Features:
    - Per-provider token counting
    - Per-task-type cost tracking
    - Real-time cost calculation
    - Budget warnings
    """

    def __init__(self, budget_limit: float = 10.0):
        self.budget_limit = budget_limit
        self.tokens_by_provider: dict[str, dict[str, int]] = {}
        self.costs_by_task_type: dict[str, float] = {}
        self.tool_calls = 0

    def add_tokens(
        self,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        task_type: str = "",
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ):
        """Add tokens for a provider, optionally tracking task type and cache tokens."""
        provider = provider.lower()
        if provider not in self.tokens_by_provider:
            self.tokens_by_provider[provider] = {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}
        self.tokens_by_provider[provider]["input"] += input_tokens
        self.tokens_by_provider[provider]["output"] += output_tokens
        self.tokens_by_provider[provider]["cache_read"] += cache_read_tokens
        self.tokens_by_provider[provider]["cache_creation"] += cache_creation_tokens
        if task_type:
            cost = self._compute_cost(provider, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens)
            self.costs_by_task_type[task_type] = self.costs_by_task_type.get(task_type, 0.0) + cost

    def _compute_cost(
        self, provider: str, input_tokens: int, output_tokens: int,
        cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
    ) -> float:
        """Compute cost for a specific token usage, including cache tokens."""
        if provider not in PRICING:
            return 0.0
        pricing = PRICING[provider]
        input_cost = (input_tokens / 1_000_000) * pricing.input_per_million
        output_cost = (output_tokens / 1_000_000) * pricing.output_per_million
        cache_read_cost = (cache_read_tokens / 1_000_000) * pricing.cache_read_per_million
        cache_creation_cost = (cache_creation_tokens / 1_000_000) * pricing.cache_creation_per_million
        return input_cost + output_cost + cache_read_cost + cache_creation_cost

    def add_tool_call(self):
        """Increment tool call counter."""
        self.tool_calls += 1

    def get_provider_cost(self, provider: str) -> float:
        """Get cost for a specific provider, including cache token costs."""
        provider = provider.lower()
        if provider not in self.tokens_by_provider or provider not in PRICING:
            return 0.0
        tokens = self.tokens_by_provider[provider]
        return self._compute_cost(
            provider, tokens["input"], tokens["output"],
            tokens.get("cache_read", 0), tokens.get("cache_creation", 0),
        )

    def get_total_cost(self) -> float:
        """Get total cost across all providers."""
        return sum(self.get_provider_cost(p) for p in self.tokens_by_provider)

    def get_total_tokens(self) -> tuple[int, int]:
        """Get total input and output tokens."""
        total_input = sum(t["input"] for t in self.tokens_by_provider.values())
        total_output = sum(t["output"] for t in self.tokens_by_provider.values())
        return total_input, total_output

    def is_over_budget(self) -> bool:
        return self.get_total_cost() > self.budget_limit

    def get_budget_percentage(self) -> float:
        if self.budget_limit <= 0:
            return 0.0
        return (self.get_total_cost() / self.budget_limit) * 100

    def format_summary(self) -> str:
        """Get formatted summary string."""
        total_cost = self.get_total_cost()
        total_in, total_out = self.get_total_tokens()
        lines = ["Session Cost\n"]
        for provider, tokens in self.tokens_by_provider.items():
            cost = self.get_provider_cost(provider)
            if cost > 0 or provider == "ollama":
                cache_info = ""
                cache_read = tokens.get("cache_read", 0)
                cache_creation = tokens.get("cache_creation", 0)
                if cache_read or cache_creation:
                    cache_info = f" | cache: {cache_read:,} read, {cache_creation:,} write"
                lines.append(
                    f"  {provider.title()}: ${cost:.4f} "
                    f"({tokens['input']:,} in / {tokens['output']:,} out{cache_info})"
                )
        if self.costs_by_task_type:
            lines.append("")
            for task, cost in sorted(self.costs_by_task_type.items(), key=lambda x: x[1], reverse=True):
                if cost > 0.0001:
                    lines.append(f"  {task}: ${cost:.4f}")
        lines.append(f"\n  Total: ${total_cost:.4f}")
        lines.append(f"  Tokens: {total_in + total_out:,}")
        lines.append(f"  Tools: {self.tool_calls} calls")
        if self.is_over_budget():
            lines.append(f"\n  [!] OVER BUDGET (${total_cost:.4f} / ${self.budget_limit:.2f})")
        else:
            pct = self.get_budget_percentage()
            if pct > 75:
                lines.append(f"\n  [!] {pct:.0f}% of budget used")
        return "\n".join(lines)

    def reset(self):
        """Reset all counters."""
        self.tokens_by_provider.clear()
        self.costs_by_task_type.clear()
        self.tool_calls = 0
