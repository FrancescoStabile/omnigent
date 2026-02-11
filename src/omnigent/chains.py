"""
Omnigent — Escalation Chains

When a finding is confirmed, suggest escalation chains (next steps).
Maps finding_category → ordered list of next steps.

Architecture:
  CHAINS is a dict of {category: [ChainStep, ...]}.
  Domain implementations populate this with their own chains.

Example (security domain):
  CHAINS["sqli"] = [
      ChainStep("Enumerate database type", "sqlmap"),
      ChainStep("Dump table names", "sqlmap"),
      ChainStep("Extract credentials", "sqlmap"),
  ]
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChainStep:
    """A single step in an escalation chain."""
    description: str
    tool_hint: str = ""
    knowledge_ref: str = ""  # Reference to knowledge file


# ═══════════════════════════════════════════════════════════════════════════
# Chain Registry — populate in your domain implementation
# ═══════════════════════════════════════════════════════════════════════════

# Structure:
#   {
#       "finding_category": [ChainStep(...), ...],
#   }
#
# Domain implementations import this and add their chains:
#   from omnigent.chains import CHAINS, ChainStep
#   CHAINS["sqli"] = [ChainStep("Dump database", "sqlmap"), ...]

CHAINS: dict[str, list[ChainStep]] = {}


# ═══════════════════════════════════════════════════════════════════════════
# Core API
# ═══════════════════════════════════════════════════════════════════════════


def get_escalation_chain(category: str) -> list[ChainStep] | None:
    """Get the escalation chain for a finding category.

    Matching strategy (in order):
    1. Direct match (exact key)
    2. Partial match (key in category or category in key)
    3. Keyword match (registered keywords)

    Returns None if no chain exists.
    """
    normalized = category.lower().strip().replace(" ", "_").replace("-", "_")

    # 1. Direct match
    if normalized in CHAINS:
        return CHAINS[normalized]

    # 2. Partial match
    for key, chain in CHAINS.items():
        if key in normalized or normalized in key:
            return chain

    return None


def format_chain_for_prompt(category: str) -> str:
    """Format an escalation chain as prompt text."""
    chain = get_escalation_chain(category)
    if not chain:
        return ""

    lines = [f"\n## Escalation Chain: {category}\n"]
    for i, step in enumerate(chain, 1):
        lines.append(f"  {i}. {step.description}")
        if step.tool_hint:
            lines.append(f"     → use: {step.tool_hint}")
    return "\n".join(lines)
