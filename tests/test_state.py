"""Tests for State, Finding, and Severity."""

import pytest
from omnigent.state import Finding, Severity, State, EnrichFn
from omnigent.planner import TaskPlan
from omnigent.domain_profile import DomainProfile


# ── Finding Tests ────────────────────────────────────────────────────────

class TestFinding:
    def test_basic_creation(self):
        f = Finding(title="SQL Injection in login", severity="high")
        assert f.title == "SQL Injection in login"
        assert f.severity == "high"
        assert f.timestamp is not None

    def test_severity_normalization(self):
        f = Finding(title="Test", severity="HIGH")
        assert f.severity == "high"

    def test_severity_prefix_match(self):
        f = Finding(title="Test", severity="cri")
        assert f.severity == "critical"

    def test_invalid_severity_defaults_to_info(self):
        f = Finding(title="Test", severity="banana")
        assert f.severity == "info"

    def test_generic_title_gets_suffix(self):
        f = Finding(title="issue")
        assert "needs specificity" in f.title

    def test_empty_title(self):
        f = Finding(title="")
        assert f.title == "Untitled Finding"

    def test_specific_title_unchanged(self):
        f = Finding(title="XSS in search parameter")
        assert f.title == "XSS in search parameter"

    def test_enrichment_dict(self):
        f = Finding(title="Test", enrichment={"cwe": 89})
        assert f.enrichment["cwe"] == 89

    def test_to_dict(self):
        f = Finding(title="Test Finding", severity="medium", description="Desc")
        d = f.to_dict()
        assert d["title"] == "Test Finding"
        assert d["severity"] == "medium"
        assert d["description"] == "Desc"
        assert "timestamp" in d
        assert "enrichment" in d

    def test_severity_enum_values(self):
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"
        assert Severity.INFO.value == "info"


# ── State Tests ──────────────────────────────────────────────────────────

class TestState:
    def test_default_state(self):
        s = State()
        assert s.messages == []
        assert s.findings == []
        assert s.subject is None
        assert s.iteration == 0

    def test_add_message(self):
        s = State()
        s.add_message("user", "hello")
        assert len(s.messages) == 1
        assert s.messages[0]["role"] == "user"
        assert s.messages[0]["content"] == "hello"
        assert s.iteration == 1

    def test_add_finding(self):
        s = State()
        f = Finding(title="Test Issue", severity="high")
        s.add_finding(f)
        assert len(s.findings) == 1
        assert s.findings[0].title == "Test Issue"

    def test_enrichment_hook(self):
        def my_enricher(finding: Finding) -> None:
            finding.enrichment["custom"] = "enriched"

        s = State(enrich_fn=my_enricher)
        f = Finding(title="Test", severity="low")
        s.add_finding(f)
        assert f.enrichment["custom"] == "enriched"

    def test_enrichment_failure_doesnt_crash(self):
        def bad_enricher(finding: Finding) -> None:
            raise ValueError("Enrichment failed!")

        s = State(enrich_fn=bad_enricher)
        f = Finding(title="Test", severity="low")
        s.add_finding(f)  # Should not raise
        assert len(s.findings) == 1

    def test_get_findings_by_severity(self):
        s = State()
        s.add_finding(Finding(title="A", severity="high"))
        s.add_finding(Finding(title="B", severity="low"))
        s.add_finding(Finding(title="C", severity="high"))
        assert len(s.get_findings_by_severity("high")) == 2
        assert len(s.get_findings_by_severity("low")) == 1
        assert len(s.get_findings_by_severity("critical")) == 0

    def test_critical_and_high_count(self):
        s = State()
        s.add_finding(Finding(title="A", severity="critical"))
        s.add_finding(Finding(title="B", severity="critical"))
        s.add_finding(Finding(title="C", severity="high"))
        assert s.critical_count == 2
        assert s.high_count == 1

    def test_clear(self):
        s = State(subject="test")
        s.add_message("user", "hello")
        s.add_finding(Finding(title="Test", severity="low"))
        s.clear()
        assert s.messages == []
        assert s.findings == []
        assert s.iteration == 0
