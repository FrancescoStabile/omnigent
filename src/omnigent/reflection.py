"""
Omnigent — Reflection Engine

Analyzes tool results and generates strategic insights.
Runs after each tool call to update hypotheses and guide next action.

Architecture:
  REFLECTORS is a dict of {tool_name: reflector_function}.
  Each reflector receives (result, args, profile, lines) and appends
  insight strings to `lines`.

  Reflectors can be synchronous or asynchronous functions.
  Domain implementations populate REFLECTORS with their own functions.

Example (security domain):
  def _reflect_nmap(result, args, profile, lines):
      open_ports = profile.metadata.get("open_ports", [])
      if open_ports:
          lines.append(f"**Scan complete**: {len(open_ports)} open ports found")
      ...
  REFLECTORS["nmap"] = _reflect_nmap

  # Async reflectors are also supported:
  async def _reflect_with_db(result, args, profile, lines):
      data = await db.query(...)
      lines.append(f"DB match: {data}")
  REFLECTORS["db_tool"] = _reflect_with_db
"""

from __future__ import annotations

import asyncio
import inspect

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
# Reflectors can be sync or async functions.

REFLECTORS: dict[str, callable] = {}


# ═══════════════════════════════════════════════════════════════════════════
# Core API
# ═══════════════════════════════════════════════════════════════════════════


async def reflect_on_result_async(
    tool_name: str,
    tool_args: dict,
    result: str,
    profile: DomainProfile,
) -> str:
    """Generate a reflection summary after a tool execution (async version).

    Supports both sync and async reflectors. Preferred over reflect_on_result().
    """
    lines: list[str] = []

    reflector = REFLECTORS.get(tool_name)
    if reflector:
        try:
            if inspect.iscoroutinefunction(reflector):
                await reflector(result, tool_args, profile, lines)
            else:
                reflector(result, tool_args, profile, lines)
        except Exception:
            pass  # Reflectors must never crash the agent

    _append_general_intelligence(profile, lines)

    return "\n".join(lines) if lines else ""


def reflect_on_result(
    tool_name: str,
    tool_args: dict,
    result: str,
    profile: DomainProfile,
) -> str:
    """Generate a reflection summary after a tool execution (sync version).

    For backward compatibility. Wraps sync reflectors only.
    Async reflectors are skipped — use reflect_on_result_async() instead.
    """
    lines: list[str] = []

    reflector = REFLECTORS.get(tool_name)
    if reflector:
        try:
            if inspect.iscoroutinefunction(reflector):
                pass  # Skip async reflectors in sync context
            else:
                reflector(result, tool_args, profile, lines)
        except Exception:
            pass  # Reflectors must never crash the agent

    _append_general_intelligence(profile, lines)

    return "\n".join(lines) if lines else ""


def _append_general_intelligence(profile: DomainProfile, lines: list[str]) -> None:
    """Append general intelligence insights (untested hypotheses, confirmed findings)."""
    untested = profile.get_untested_hypotheses()
    if untested:
        top3 = sorted(untested, key=lambda h: h.confidence, reverse=True)[:3]
        lines.append("\n**Top untested hypotheses:**")
        for h in top3:
            lines.append(f"  - [{h.confidence}] {h.hypothesis_type}: {h.evidence}")

    confirmed = profile.get_confirmed()
    if confirmed:
        lines.append(f"\n**{len(confirmed)} confirmed findings** — consider escalation or deeper analysis.")
