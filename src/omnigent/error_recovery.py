"""
Omnigent — Error Recovery Patterns

Tool-specific error patterns with recovery strategies.
When a tool fails, provide intelligent guidance for next action.

Architecture:
  ERROR_PATTERNS is a dict of {tool_name: {pattern_name: {indicators, strategy}}}.
  Domain implementations populate this with their own patterns.

  get_recovery_strategy() matches error output against patterns.
  inject_recovery_guidance() formats the guidance for LLM injection.

Example (security domain):
  ERROR_PATTERNS["nmap"] = {
      "timeout": {
          "indicators": ["timed out", "no response"],
          "strategy": RecoveryStrategy(
              guidance="Nmap timed out. Try with different scan type.",
              retry_tool="nmap",
              retry_args={"scan_type": "quick"},
          ),
      },
  }
"""

from dataclasses import dataclass


@dataclass
class RecoveryStrategy:
    """Strategy for recovering from a tool failure."""
    guidance: str                        # Human-readable explanation
    retry_tool: str | None = None        # Suggested tool to retry with
    retry_args: dict | None = None       # Suggested different arguments
    give_up: bool = False                # If True, don't retry this vector


# ═══════════════════════════════════════════════════════════════════════════
# Pattern Registry — populate in your domain implementation
# ═══════════════════════════════════════════════════════════════════════════

# Structure:
#   {
#       "tool_name": {
#           "pattern_name": {
#               "indicators": ["substring1", "substring2"],
#               "strategy": RecoveryStrategy(...),
#           },
#       },
#   }
#
# Domain implementations import this and add their patterns:
#   from omnigent.error_recovery import ERROR_PATTERNS, RecoveryStrategy
#   ERROR_PATTERNS["my_tool"] = { ... }

ERROR_PATTERNS: dict[str, dict] = {}


# ═══════════════════════════════════════════════════════════════════════════
# Core API
# ═══════════════════════════════════════════════════════════════════════════


def get_recovery_strategy(tool_name: str, error_result: str) -> RecoveryStrategy | None:
    """Match error result against known patterns and return recovery strategy.

    Args:
        tool_name: Name of the tool that failed
        error_result: The error output from the tool

    Returns:
        RecoveryStrategy if pattern matched, None otherwise
    """
    tool_patterns = ERROR_PATTERNS.get(tool_name, {})
    error_lower = error_result.lower()

    for pattern_name, pattern_data in tool_patterns.items():
        indicators = pattern_data["indicators"]
        if any(ind in error_lower for ind in indicators):
            return pattern_data["strategy"]

    return None


def inject_recovery_guidance(tool_name: str, error_result: str) -> str:
    """Generate recovery guidance prompt to inject after tool failure.

    Args:
        tool_name: Name of the tool that failed
        error_result: The error output from the tool

    Returns:
        Formatted prompt with recovery guidance
    """
    strategy = get_recovery_strategy(tool_name, error_result)

    if not strategy:
        # Generic failure guidance
        return f"""
Tool '{tool_name}' failed: {error_result[:100]}

This error is not recognized. Analyze the error message and:
1. Determine if you should retry with different parameters
2. Try a different tool
3. Or move on to a different approach
"""

    if strategy.give_up:
        return f"""
Tool '{tool_name}' failed: {error_result[:100]}

**Recovery Guidance**: {strategy.guidance}

This is expected. Move on to next approach.
"""

    guidance = f"""
Tool '{tool_name}' failed: {error_result[:100]}

**Recovery Guidance**: {strategy.guidance}
"""

    if strategy.retry_tool:
        guidance += f"\n\n**Suggested Next Step**: \nTool: {strategy.retry_tool}\n"
        if strategy.retry_args:
            guidance += f"Arguments: {strategy.retry_args}\n"

    return guidance
