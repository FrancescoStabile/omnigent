"""Tests for DomainProfile and Hypothesis."""

import pytest
from omnigent.domain_profile import DomainProfile, Hypothesis


class TestHypothesis:
    def test_creation(self):
        h = Hypothesis(hypothesis_type="sqli", location="/login")
        assert h.hypothesis_type == "sqli"
        assert h.location == "/login"
        assert h.confidence == 0.5
        assert h.tested is False
        assert h.confirmed is False

    def test_str_untested(self):
        h = Hypothesis(hypothesis_type="xss", location="/search")
        assert "[UNTESTED]" in str(h)

    def test_str_tested(self):
        h = Hypothesis(hypothesis_type="xss", location="/search", tested=True)
        assert "[TESTED]" in str(h)

    def test_str_confirmed(self):
        h = Hypothesis(hypothesis_type="xss", location="/search", confirmed=True)
        assert "[CONFIRMED]" in str(h)


class TestDomainProfile:
    def test_default_profile(self):
        p = DomainProfile()
        assert p.subject == ""
        assert p.scope == []
        assert p.hypotheses == []
        assert p.metadata == {}

    def test_add_hypothesis_new(self):
        p = DomainProfile()
        h = Hypothesis(hypothesis_type="sqli", location="/login")
        assert p.add_hypothesis(h) is True
        assert len(p.hypotheses) == 1

    def test_add_hypothesis_duplicate(self):
        p = DomainProfile()
        h1 = Hypothesis(hypothesis_type="sqli", location="/login", confidence=0.3)
        h2 = Hypothesis(hypothesis_type="sqli", location="/login", confidence=0.8)
        p.add_hypothesis(h1)
        result = p.add_hypothesis(h2)
        assert result is False  # Duplicate
        assert len(p.hypotheses) == 1
        assert p.hypotheses[0].confidence == 0.8  # Updated to higher

    def test_add_hypothesis_duplicate_confirms(self):
        p = DomainProfile()
        h1 = Hypothesis(hypothesis_type="sqli", location="/login")
        h2 = Hypothesis(hypothesis_type="sqli", location="/login", confirmed=True)
        p.add_hypothesis(h1)
        p.add_hypothesis(h2)
        assert p.hypotheses[0].confirmed is True

    def test_add_hypothesis_different_types(self):
        p = DomainProfile()
        p.add_hypothesis(Hypothesis(hypothesis_type="sqli", location="/login"))
        p.add_hypothesis(Hypothesis(hypothesis_type="xss", location="/login"))
        assert len(p.hypotheses) == 2

    def test_to_dict(self):
        p = DomainProfile(subject="test.com", scope=["test.com"])
        p.add_hypothesis(Hypothesis(hypothesis_type="sqli", location="/login"))
        d = p.to_dict()
        assert d["subject"] == "test.com"
        assert len(d["hypotheses"]) == 1

    def test_from_dict(self):
        d = {
            "subject": "test.com",
            "scope": ["test.com"],
            "hypotheses": [
                {"hypothesis_type": "sqli", "location": "/login", "confidence": 0.7}
            ],
            "metadata": {"key": "value"},
        }
        p = DomainProfile.from_dict(d)
        assert p.subject == "test.com"
        assert len(p.hypotheses) == 1
        assert p.hypotheses[0].confidence == 0.7

    def test_subclass(self):
        """Test that DomainProfile can be subclassed."""
        from dataclasses import dataclass, field

        @dataclass
        class CodeProfile(DomainProfile):
            files_scanned: list[str] = field(default_factory=list)
            avg_complexity: float = 0.0

        cp = CodeProfile(subject="/repo")
        cp.files_scanned.append("main.py")
        cp.avg_complexity = 8.5
        assert cp.subject == "/repo"
        assert cp.files_scanned == ["main.py"]
        assert cp.avg_complexity == 8.5
        # Base methods still work
        cp.add_hypothesis(Hypothesis(hypothesis_type="god_class", location="main.py"))
        assert len(cp.hypotheses) == 1
