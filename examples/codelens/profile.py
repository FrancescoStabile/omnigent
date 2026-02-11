"""
CodeLens domain profile — structured memory for code quality analysis.

Shows how to subclass DomainProfile and Hypothesis for a concrete domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omnigent.domain_profile import DomainProfile, Hypothesis


# ── Domain-Specific Hypothesis ──────────────────────────────────────────────

@dataclass
class CodeSmellHypothesis(Hypothesis):
    """A suspected code quality issue."""

    file_path: str = ""
    line_range: tuple[int, int] | None = None
    smell_type: str = ""  # e.g. "god_class", "long_method", "tight_coupling"
    metric_value: float | None = None  # measured complexity, coupling, etc.


# ── Domain-Specific Profile ─────────────────────────────────────────────────

@dataclass
class CodeProfile(DomainProfile):
    """Structured memory for code quality analysis of a repository."""

    # ── Discovery ────────────────────────────────────────────────
    repo_url: str = ""
    language: str = ""
    framework: str = ""
    loc: int = 0
    file_count: int = 0

    # ── Metrics ──────────────────────────────────────────────────
    avg_complexity: float = 0.0
    max_complexity: float = 0.0
    test_coverage: float = 0.0
    dependency_count: int = 0
    circular_deps: list[str] = field(default_factory=list)

    # ── Files Analysis ───────────────────────────────────────────
    files_scanned: list[str] = field(default_factory=list)
    hot_spots: list[dict] = field(default_factory=list)  # high-change + high-complexity files
    dead_code: list[str] = field(default_factory=list)

    # ── Architecture ─────────────────────────────────────────────
    entry_points: list[str] = field(default_factory=list)
    layer_violations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable profile summary for prompt injection."""
        lines = [f"Repository: {self.subject}"]
        if self.language:
            lines.append(f"Language: {self.language} | Framework: {self.framework}")
        if self.loc:
            lines.append(f"LOC: {self.loc:,} across {self.file_count} files")
        if self.avg_complexity:
            lines.append(f"Complexity: avg={self.avg_complexity:.1f}, max={self.max_complexity:.1f}")
        if self.test_coverage:
            lines.append(f"Test coverage: {self.test_coverage:.0%}")
        if self.hot_spots:
            lines.append(f"Hot spots: {len(self.hot_spots)}")
        if self.circular_deps:
            lines.append(f"Circular deps: {len(self.circular_deps)}")
        return "\n".join(lines)
