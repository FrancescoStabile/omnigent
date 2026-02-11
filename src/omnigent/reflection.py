"""
Omnigent — Reflection Engine

Analyzes tool results and generates strategic insights.
Runs after each tool call to update hypotheses and guide next action.

Architecture:
  REFLECTORS is a dict of {tool_name: reflector_function}.
  Each reflector receives (result, args, profile, lines) and appends
  insight strings to `lines`.

  Domain implementations populate REFLECTORS with their own functions.

Example (security domain):
  def _reflect_nmap(result, args, profile, lines):
      open_ports = profile.metadata.get("open_ports", [])
      if open_ports:
          lines.append(f"**Scan complete**: {len(open_ports)} open ports found")
      ...
  REFLECTORS["nmap"] = _reflect_nmap
"""

from __future__ import annotations

from omnigent.domain_profile import DomainProfile


# ═══════════════════════════════════════════════════════════════════════════
# Reflector Registry — populate in your domain implementation
# ═══════════════════════════════════════════════════════════════════════════

# Structure:
#   {
#       "tool_name": callable(result: str, args: dict, profile: DomainProfile, lines: list[str]) -> None
#   }
#
# Each reflector inspects the tool result and appends insights to `lines`.

REFLECTORS: dict[str, callable] = {}


# ═══════════════════════════════════════════════════════════════════════════
# Core API
# ═══════════════════════════════════════════════════════════════════════════


def reflect_on_result(
    tool_name: str,
    tool_args: dict,
    result: str,
    profile: DomainProfile,
) -> str:
    """Generate a reflection summary after a tool execution.

    Returns a concise analysis string that gets injected as an assistant
    thought, guiding the LLM's next decision.
    """
    lines: list[str] = []

    # Tool-specific reflection
    reflector = REFLECTORS.get(tool_name)
    if reflector:
        try:
            reflector(result, tool_args, profile, lines)
        except Exception:
            pass  # Reflectors must never crash the agent

    # General intelligence: untested hypotheses
    untested = profile.get_untested_hypotheses()
    if untested:
        top3 = sorted(untested, key=lambda h: h.confidence, reverse=True)[:3]
        lines.append("\n**Top untested hypotheses:**")
        for h in top3:
            lines.append(f"  - [{h.confidence}] {h.hypothesis_type}: {h.evidence}")

    # General intelligence: confirmed findings
    confirmed = profile.get_confirmed()
    if confirmed:
        lines.append(f"\n**{len(confirmed)} confirmed findings** — consider escalation or deeper analysis.")

    if not lines:
        return ""

    return "\n".join(lines)
