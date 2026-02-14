"""
Omnigent — Session State

State management: findings, history, domain profile, task plan.

Finding model uses Pydantic for validation:
  - Enforces specific, non-generic titles
  - Validates severity as enum
  - Auto-generates timestamps
  - Supports domain-specific enrichment via hook

The enrichment hook (`enrich_finding`) is called on every `add_finding()`.
Set it to your domain's enrichment function:
    state.enrich_fn = my_domain_enrichment
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from omnigent.domain_profile import DomainProfile
from omnigent.planner import TaskPlan

logger = logging.getLogger("omnigent.state")


# ═══════════════════════════════════════════════════════════════════════════
# Severity Enum
# ═══════════════════════════════════════════════════════════════════════════

class Severity(str, enum.Enum):
    """Standardized severity levels — generic across domains."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# ═══════════════════════════════════════════════════════════════════════════
# Finding — Pydantic-validated discovery
# ═══════════════════════════════════════════════════════════════════════════

_GENERIC_TITLES = frozenset({
    "issue", "finding", "problem", "bug", "error",
    "issue found", "problem found",
    "potential issue", "possible problem",
})


class Finding(BaseModel):
    """A validated finding/discovery.

    Pydantic enforces:
      - Title must be specific (not a generic word)
      - Severity must be a valid enum value
      - Timestamp auto-generated
      - Extensible metadata dict for domain-specific enrichment
    """
    title: str
    severity: str = "info"
    description: str = ""
    evidence: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)

    # Domain-specific enrichment fields — populated by enrich_fn
    # For security: cwe_id, owasp_category, cvss_score
    # For code quality: iso25010_characteristic, solid_violation
    # For compliance: regulation_article, control_id
    enrichment: dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {datetime: lambda v: v.isoformat()},
    }

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, v: str) -> str:
        """Normalize severity to lowercase and validate."""
        if isinstance(v, Severity):
            return v.value
        v = str(v).lower().strip()
        valid = {s.value for s in Severity}
        if v not in valid:
            for sev in valid:
                if sev.startswith(v[:3]):
                    return sev
            logger.warning(f"Invalid severity '{v}', defaulting to 'info'")
            return "info"
        return v

    @field_validator("title", mode="before")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Ensure title is specific enough to be useful."""
        v = str(v).strip()
        if not v:
            return "Untitled Finding"
        if v.lower() in _GENERIC_TITLES:
            return f"{v} (needs specificity)"
        return v

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "title": self.title,
            "severity": self.severity,
            "description": self.description,
            "evidence": self.evidence,
            "timestamp": self.timestamp.isoformat(),
            "enrichment": self.enrichment,
        }


# Type for enrichment function hook
EnrichFn = Callable[[Finding], None]


@dataclass
class State:
    """Session state for the agent."""

    # Conversation history (LLM format)
    messages: list[dict] = field(default_factory=list)

    # Discoveries
    findings: list[Finding] = field(default_factory=list)

    # Current subject
    subject: str | None = None

    # Structured domain knowledge
    profile: DomainProfile = field(default_factory=DomainProfile)

    # Task plan
    plan: TaskPlan = field(default_factory=lambda: TaskPlan(objective=""))

    # Session data (cookies, tokens, credentials, etc)
    session_data: dict[str, Any] = field(default_factory=dict)

    # Metadata
    started_at: datetime = field(default_factory=datetime.now)
    iteration: int = 0

    # Incremental token tracking
    _total_tokens: int = 0

    # Enrichment hook — set this to your domain's enrichment function
    enrich_fn: EnrichFn | None = None

    def add_message(self, role: str, content: str | list):
        """Add message to history with incremental token tracking."""
        from omnigent.context import estimate_tokens
        self.messages.append({"role": role, "content": content})
        self._total_tokens += estimate_tokens(content)
        self.iteration += 1

    @property
    def total_tokens(self) -> int:
        """Get accumulated token count (O(1) instead of recalculating)."""
        return self._total_tokens

    def add_finding(self, finding: Finding):
        """Add finding with optional auto-enrichment via hook."""
        if self.enrich_fn:
            try:
                self.enrich_fn(finding)
            except Exception as e:
                logger.debug(f"Finding enrichment failed: {e}")
        self.findings.append(finding)

    def get_findings_by_severity(self, severity: str) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]

    @property
    def critical_count(self) -> int:
        return len(self.get_findings_by_severity("critical"))

    @property
    def high_count(self) -> int:
        return len(self.get_findings_by_severity("high"))

    def clear(self):
        """Clear state for new session."""
        self.messages.clear()
        self.findings.clear()
        self.session_data.clear()
        self.profile = DomainProfile()
        self.plan = TaskPlan(objective="")
        self.iteration = 0
        self._total_tokens = 0
        self.started_at = datetime.now()
