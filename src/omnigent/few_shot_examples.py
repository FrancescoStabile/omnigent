"""
Omnigent — Few-Shot Examples System

Provides concrete examples for each tool to improve LLM tool accuracy.
2-3 examples per tool (2 good + 1 anti-pattern).

Architecture:
  EXAMPLES is a dict of {tool_name: [ToolExample, ...]}.
  Domain implementations populate this with their own examples.

Example (security domain):
  EXAMPLES["nmap"] = [
      ToolExample(
          scenario="What services are running?",
          thinking="Need service detection scan...",
          tool_name="nmap",
          tool_args={"target": "10.0.0.5", "scan_type": "service"},
          expected_result="22/tcp ssh, 80/tcp http",
          is_good=True,
      ),
  ]
"""

from dataclasses import dataclass


@dataclass
class ToolExample:
    """A single few-shot example for a tool."""
    scenario: str      # User request context
    thinking: str      # Chain-of-thought reasoning
    tool_name: str
    tool_args: dict
    expected_result: str
    is_good: bool      # True = good example, False = anti-pattern


# ═══════════════════════════════════════════════════════════════════════════
# Examples Registry — populate in your domain implementation
# ═══════════════════════════════════════════════════════════════════════════

# Structure:
#   {
#       "tool_name": [ToolExample(...), ...],
#   }

EXAMPLES: dict[str, list[ToolExample]] = {}


# ═══════════════════════════════════════════════════════════════════════════
# Core API
# ═══════════════════════════════════════════════════════════════════════════


def get_examples(tool_name: str) -> list[ToolExample]:
    """Get few-shot examples for a tool."""
    return EXAMPLES.get(tool_name, [])


def format_examples_for_prompt(tool_name: str, max_examples: int = 3) -> str:
    """Format few-shot examples as prompt text for a tool.

    Returns:
        Formatted examples string, or empty string if no examples exist.
    """
    examples = get_examples(tool_name)
    if not examples:
        return ""

    lines = [f"\n## Examples for `{tool_name}`\n"]

    for ex in examples[:max_examples]:
        label = "✓ GOOD" if ex.is_good else "✗ BAD"
        lines.append(f"### {label}: {ex.scenario}")
        lines.append(f"**Thinking**: {ex.thinking.strip()}")
        lines.append(f"**Tool**: {ex.tool_name}({ex.tool_args})")
        lines.append(f"**Expected**: {ex.expected_result}")
        lines.append("")

    return "\n".join(lines)


def get_all_examples() -> dict[str, list[ToolExample]]:
    """Get all registered examples."""
    return EXAMPLES.copy()
