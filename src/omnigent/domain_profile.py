"""
Omnigent — Domain Profile (Structured Agent Memory)

Everything the agent knows about its subject, auto-populated from tool results.
Injected into LLM context as a structured summary.

This is the ABSTRACT base — domain implementations subclass this and add
their own data models (ports, endpoints for security; files, functions for code; etc.)

Override points:
  - Define your own dataclass fields
  - Override to_prompt_summary() for domain-specific LLM context
  - Override to_dict() / from_dict() for persistence
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Hypothesis:
    """A suspected finding/issue that needs investigation.

    Generic base for any domain — could be a vulnerability hypothesis,
    a code quality suspicion, a compliance gap guess, etc.
    """
    hypothesis_type: str       # "sqli", "god_object", "gdpr_gap", etc.
    location: str              # Where in the subject (URL, file path, etc.)
    evidence: str = ""         # Supporting evidence
    confidence: float = 0.5    # 0.0 = wild guess, 1.0 = confirmed
    tested: bool = False
    confirmed: bool = False
    tool_used: str = ""
    notes: str = ""
    step_ref: str = ""         # Link to TaskStep (phase:step_index)

    def __str__(self) -> str:
        status = "CONFIRMED" if self.confirmed else ("TESTED" if self.tested else "UNTESTED")
        return f"[{status}] {self.hypothesis_type} @ {self.location} (confidence: {self.confidence:.0%})"


@dataclass
class DomainProfile:
    """
    Base class for domain-specific structured memory.

    Everything the agent knows about its subject. Auto-populated by
    extractors after each tool call. Serialized with sessions for
    resume capability. Injected into LLM context as structured summary.

    Subclass this for your domain and add typed fields:
      - SecurityProfile: ports, endpoints, technologies, credentials
      - CodeProfile: files, functions, dependencies, complexity metrics
      - ComplianceProfile: controls, gaps, evidence documents
    """
    # Subject identity
    subject: str = ""
    scope: list[str] = field(default_factory=list)

    # Hypotheses tracking (generic)
    hypotheses: list[Hypothesis] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    # ── Mutation methods ──

    def add_hypothesis(self, hyp: Hypothesis) -> bool:
        """Add hypothesis if not duplicate. Returns True if new."""
        for h in self.hypotheses:
            if h.hypothesis_type == hyp.hypothesis_type and h.location == hyp.location:
                if hyp.confidence > h.confidence:
                    h.confidence = hyp.confidence
                    h.evidence = hyp.evidence
                if hyp.confirmed:
                    h.confirmed = True
                if hyp.tested:
                    h.tested = True
                self._touch()
                return False
        self.hypotheses.append(hyp)
        self._touch()
        return True

    def mark_hypothesis_tested(self, hyp_type: str, location: str, confirmed: bool, notes: str = ""):
        """Mark a hypothesis as tested."""
        for h in self.hypotheses:
            if h.hypothesis_type == hyp_type and h.location == location:
                h.tested = True
                h.confirmed = confirmed
                h.confidence = 1.0 if confirmed else 0.0
                if notes:
                    h.notes = notes
                self._touch()
                return

    # ── Query methods ──

    def get_untested_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses if not h.tested]

    def get_confirmed(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.confirmed]

    # ── Context injection ──

    def to_prompt_summary(
        self, max_confirmed: int = 15, max_untested: int = 10
    ) -> str:
        """
        Generate a structured summary for LLM context injection.
        Override this in your domain profile for richer output.

        Args:
            max_confirmed: Maximum confirmed findings to include (default 15).
            max_untested: Maximum untested hypotheses to include (default 10).
        """
        if not self.subject:
            return ""

        lines = [f"## Current Subject Knowledge: {self.subject}"]

        confirmed = self.get_confirmed()
        untested = self.get_untested_hypotheses()

        if confirmed:
            shown = confirmed[:max_confirmed]
            lines.append(f"\n### Confirmed Findings ({len(confirmed)})")
            for v in shown:
                lines.append(f"- {v}")
            if len(confirmed) > max_confirmed:
                lines.append(f"- ... and {len(confirmed) - max_confirmed} more")

        if untested:
            shown = untested[:max_untested]
            lines.append(f"\n### Untested Hypotheses ({len(untested)})")
            for h in shown:
                lines.append(f"- {h}")
            if len(untested) > max_untested:
                lines.append(f"- ... and {len(untested) - max_untested} more")

        if self.metadata:
            lines.append("\n### Metadata")
            for k, v in list(self.metadata.items())[:20]:
                lines.append(f"- {k}: {v}")

        return "\n".join(lines)

    # ── Serialization ──

    def to_dict(self) -> dict:
        """Serialize for session persistence."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> DomainProfile:
        """Deserialize from session data."""
        if not data:
            return cls()
        profile = cls()
        profile.subject = data.get("subject", "")
        profile.scope = data.get("scope", [])
        profile.metadata = data.get("metadata", {})
        profile.last_updated = data.get("last_updated", datetime.now().isoformat())
        for h in data.get("hypotheses", []):
            if isinstance(h, dict):
                profile.hypotheses.append(Hypothesis(**h))
        return profile

    def _touch(self):
        self.last_updated = datetime.now().isoformat()
