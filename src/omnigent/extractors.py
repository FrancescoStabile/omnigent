"""
Omnigent — Result Extractors

After every tool call, the appropriate extractor runs to populate
the DomainProfile with structured data. This is the key architectural
piece — it converts raw tool output into structured knowledge.

Design principles:
1. Never crash — extractors catch all exceptions
2. Be permissive in parsing — tools return messy output
3. Extract ALL useful information, not just the obvious
4. Generate hypotheses when patterns suggest findings

Architecture:
  EXTRACTORS is a dict of {tool_name: extractor_function}.
  Each extractor receives (profile, result, args) and mutates the profile.

  Domain implementations populate EXTRACTORS with their own functions.

Example (security domain):
  def extract_nmap(profile, result, args):
      data = json.loads(result)
      for host in data.get("hosts", []):
          for port_data in host.get("ports", []):
              profile.metadata.setdefault("ports", []).append(port_data)
  EXTRACTORS["nmap"] = lambda profile, result, args: extract_nmap(profile, result, args)
"""

from __future__ import annotations

import logging
from typing import Any

from omnigent.domain_profile import DomainProfile

logger = logging.getLogger("omnigent.extractors")


# ═══════════════════════════════════════════════════════════════════════════
# Extractor Registry — populate in your domain implementation
# ═══════════════════════════════════════════════════════════════════════════

# Structure:
#   {
#       "tool_name": callable(profile: DomainProfile, result: str, args: dict) -> None
#   }
#
# Each extractor parses the tool result and updates the DomainProfile.

EXTRACTORS: dict[str, Any] = {}


# ═══════════════════════════════════════════════════════════════════════════
# Core API
# ═══════════════════════════════════════════════════════════════════════════


def run_extractor(tool_name: str, profile: DomainProfile, result: str, args: dict):
    """Run the appropriate extractor for a tool result.

    Never raises — all errors are caught and logged.
    """
    extractor = EXTRACTORS.get(tool_name)
    if not extractor:
        return

    try:
        extractor(profile, result, args)
    except Exception as e:
        logger.warning(f"Extractor failed for {tool_name}: {e}", exc_info=True)
