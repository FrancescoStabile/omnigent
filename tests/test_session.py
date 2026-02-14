"""Comprehensive tests for omnigent.session module.

Covers:
  - Session dataclass: from_dict/to_dict roundtrip, defaults
  - SessionManager: create, save_state, resume, list, delete, mark status
  - Export in md/json/html formats
  - derive_key determinism and randomness
"""

import json
import pytest
from pathlib import Path

from omnigent.session import Session, SessionManager
from omnigent.state import State, Finding


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_session_dict(**overrides) -> dict:
    """Build a valid Session dict with sensible defaults, applying overrides."""
    base = {
        "id": "test-session-001",
        "timestamp": "2025-01-15T10:30:00",
        "subject": "example.com",
        "messages": [{"role": "user", "content": "hello"}],
        "findings": [
            {
                "title": "SQL Injection in /login",
                "severity": "high",
                "description": "Parameter 'user' is injectable.",
                "evidence": "sqlmap output...",
            }
        ],
        "cost": 0.0042,
        "tokens_in": 1500,
        "tokens_out": 800,
        "status": "active",
    }
    base.update(overrides)
    return base


def _make_state(subject: str = "example.com", with_findings: bool = False) -> State:
    """Build a minimal State object for save_state tests."""
    state = State(subject=subject)
    state.messages = [{"role": "user", "content": "test message"}]
    if with_findings:
        state.findings.append(
            Finding(
                title="Open Port 22",
                severity="medium",
                description="SSH port is open.",
                evidence="nmap scan",
            )
        )
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# TestSession — dataclass serialization
# ═══════════════════════════════════════════════════════════════════════════════

class TestSession:
    """Session.from_dict / .to_dict roundtrip and default handling."""

    def test_to_dict_contains_all_fields(self):
        data = _make_session_dict()
        session = Session.from_dict(data)
        result = session.to_dict()

        assert result["id"] == "test-session-001"
        assert result["timestamp"] == "2025-01-15T10:30:00"
        assert result["subject"] == "example.com"
        assert result["status"] == "active"
        assert result["cost"] == 0.0042
        assert result["tokens_in"] == 1500
        assert result["tokens_out"] == 800
        assert len(result["messages"]) == 1
        assert len(result["findings"]) == 1

    def test_roundtrip_identity(self):
        """from_dict(to_dict(s)) should produce an equivalent Session."""
        original = _make_session_dict(
            domain_profile={"subject": "target.com"},
            task_plan={"objective": "Full audit"},
        )
        session = Session.from_dict(original)
        roundtripped = Session.from_dict(session.to_dict())

        assert roundtripped.id == session.id
        assert roundtripped.subject == session.subject
        assert roundtripped.domain_profile == session.domain_profile
        assert roundtripped.task_plan == session.task_plan
        assert roundtripped.status == session.status
        assert roundtripped.cost == session.cost

    def test_defaults_for_optional_fields(self):
        """domain_profile and task_plan default to None when absent."""
        data = _make_session_dict()
        session = Session.from_dict(data)
        assert session.domain_profile is None
        assert session.task_plan is None

    def test_from_dict_ignores_unknown_keys(self):
        """Extra keys in the dict should be silently ignored."""
        data = _make_session_dict(extra_field="should be dropped", another=42)
        session = Session.from_dict(data)
        assert not hasattr(session, "extra_field")
        assert session.id == "test-session-001"

    def test_from_dict_with_domain_profile_and_task_plan(self):
        data = _make_session_dict(
            domain_profile={"subject": "target.com", "scope": ["*.target.com"]},
            task_plan={"objective": "Scan all endpoints", "phases": []},
        )
        session = Session.from_dict(data)
        assert session.domain_profile["subject"] == "target.com"
        assert session.task_plan["objective"] == "Scan all endpoints"


# ═══════════════════════════════════════════════════════════════════════════════
# TestSessionManager — CRUD operations
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionManager:
    """SessionManager create / save / resume / list using tmp_path."""

    def test_create_session_returns_active_session(self, tmp_path: Path):
        mgr = SessionManager(sessions_dir=tmp_path)
        session = mgr.create_session("Test subject")

        assert session.status == "active"
        assert session.subject == "Test subject"
        assert session.id  # UUID should be non-empty
        assert mgr.current_session is session

    def test_create_session_persists_to_disk(self, tmp_path: Path):
        mgr = SessionManager(sessions_dir=tmp_path)
        session = mgr.create_session("Persist test")

        json_file = tmp_path / f"{session.id}.json"
        assert json_file.exists()
        data = json.loads(json_file.read_text())
        assert data["subject"] == "Persist test"
        assert data["status"] == "active"

    def test_save_state_updates_session(self, tmp_path: Path):
        mgr = SessionManager(sessions_dir=tmp_path)
        mgr.create_session("State test")

        state = _make_state("example.com", with_findings=True)
        mgr.save_state(state, cost=0.01, tokens_in=500, tokens_out=250)

        assert mgr.current_session.cost == 0.01
        assert mgr.current_session.tokens_in == 500
        assert mgr.current_session.tokens_out == 250
        assert mgr.current_session.subject == "example.com"
        assert len(mgr.current_session.findings) == 1
        assert mgr.current_session.findings[0]["title"] == "Open Port 22"

    def test_save_state_creates_session_if_none(self, tmp_path: Path):
        """save_state auto-creates a session when current_session is None."""
        mgr = SessionManager(sessions_dir=tmp_path)
        assert mgr.current_session is None

        state = _make_state("auto-create.com")
        mgr.save_state(state, cost=0.005, tokens_in=100, tokens_out=50)

        assert mgr.current_session is not None
        assert mgr.current_session.subject == "auto-create.com"

    def test_resume_session_roundtrip(self, tmp_path: Path):
        mgr = SessionManager(sessions_dir=tmp_path)
        original = mgr.create_session("Resume test")

        state = _make_state("resume.com", with_findings=True)
        mgr.save_state(state, cost=0.03, tokens_in=2000, tokens_out=1000)

        # Create a fresh manager (simulates restart)
        mgr2 = SessionManager(sessions_dir=tmp_path)
        resumed = mgr2.resume_session(original.id)

        assert resumed is not None
        assert resumed.id == original.id
        assert resumed.subject == "resume.com"
        assert resumed.cost == 0.03
        assert len(resumed.findings) == 1

    def test_resume_nonexistent_session_returns_none(self, tmp_path: Path):
        mgr = SessionManager(sessions_dir=tmp_path)
        assert mgr.resume_session("nonexistent-id") is None

    def test_list_sessions_returns_correct_count(self, tmp_path: Path):
        mgr = SessionManager(sessions_dir=tmp_path)
        ids = []
        for i in range(5):
            s = mgr.create_session(f"Subject {i}")
            ids.append(s.id)

        sessions = mgr.list_sessions(limit=10)
        assert len(sessions) == 5

    def test_list_sessions_respects_limit(self, tmp_path: Path):
        mgr = SessionManager(sessions_dir=tmp_path)
        for i in range(5):
            mgr.create_session(f"Subject {i}")

        sessions = mgr.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_list_sessions_empty_dir(self, tmp_path: Path):
        mgr = SessionManager(sessions_dir=tmp_path)
        sessions = mgr.list_sessions()
        assert sessions == []


# ═══════════════════════════════════════════════════════════════════════════════
# TestSessionExport — md / json / html
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionExport:
    """Export in markdown, JSON, and HTML formats."""

    def _create_session_with_findings(self, tmp_path: Path) -> tuple[SessionManager, str]:
        """Helper: create and save a session with findings, return (mgr, session_id)."""
        mgr = SessionManager(sessions_dir=tmp_path)
        session = mgr.create_session("Export target")

        state = _make_state("export.com", with_findings=True)
        mgr.save_state(state, cost=0.025, tokens_in=3000, tokens_out=1500)
        return mgr, session.id

    def test_export_markdown(self, tmp_path: Path):
        mgr, sid = self._create_session_with_findings(tmp_path)
        md = mgr.export_session(sid, format="md")

        assert "# Omnigent Session Report" in md
        assert sid in md
        assert "export.com" in md
        assert "Open Port 22" in md
        assert "MEDIUM" in md
        assert "$0.0250" in md

    def test_export_json(self, tmp_path: Path):
        mgr, sid = self._create_session_with_findings(tmp_path)
        raw = mgr.export_session(sid, format="json")

        data = json.loads(raw)
        assert data["id"] == sid
        assert data["subject"] == "export.com"
        assert len(data["findings"]) == 1
        assert data["cost"] == 0.025

    def test_export_html(self, tmp_path: Path):
        mgr, sid = self._create_session_with_findings(tmp_path)
        html = mgr.export_session(sid, format="html")

        assert "<!DOCTYPE html>" in html
        assert "Omnigent" in html
        assert sid in html
        assert "Open Port 22" in html
        assert "export.com" in html

    def test_export_html_escapes_special_chars(self, tmp_path: Path):
        """Ensure HTML export escapes <, >, &, etc."""
        mgr = SessionManager(sessions_dir=tmp_path)
        session = mgr.create_session("<script>alert('xss')</script>")

        state = State(subject="<script>alert('xss')</script>")
        state.messages = [{"role": "user", "content": "test"}]
        state.findings.append(
            Finding(
                title="XSS in <div> tag",
                severity="high",
                description="Input <b>not</b> sanitized & vulnerable",
                evidence="<script>alert(1)</script>",
            )
        )
        mgr.save_state(state)

        html = mgr.export_session(session.id, format="html")
        # The raw <script> tags should be escaped
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_export_nonexistent_session_returns_empty(self, tmp_path: Path):
        mgr = SessionManager(sessions_dir=tmp_path)
        assert mgr.export_session("no-such-id", format="md") == ""
        assert mgr.export_session("no-such-id", format="json") == ""
        assert mgr.export_session("no-such-id", format="html") == ""

    def test_export_unknown_format_returns_empty(self, tmp_path: Path):
        mgr, sid = self._create_session_with_findings(tmp_path)
        assert mgr.export_session(sid, format="pdf") == ""


# ═══════════════════════════════════════════════════════════════════════════════
# TestSessionDelete
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionDelete:
    """delete_session removes both .json and .enc files."""

    def test_delete_removes_json_file(self, tmp_path: Path):
        mgr = SessionManager(sessions_dir=tmp_path)
        session = mgr.create_session("Delete me")
        sid = session.id

        json_file = tmp_path / f"{sid}.json"
        assert json_file.exists()

        mgr.delete_session(sid)
        assert not json_file.exists()

    def test_delete_removes_enc_file(self, tmp_path: Path):
        """If an .enc file exists for a session, it should also be removed."""
        mgr = SessionManager(sessions_dir=tmp_path)
        sid = "fake-encrypted-session"

        # Manually create an .enc file to simulate encrypted session
        enc_file = tmp_path / f"{sid}.enc"
        enc_file.write_text("encrypted-data-placeholder")
        assert enc_file.exists()

        mgr.delete_session(sid)
        assert not enc_file.exists()

    def test_delete_removes_both_json_and_enc(self, tmp_path: Path):
        """If both .json and .enc exist (edge case), both are removed."""
        mgr = SessionManager(sessions_dir=tmp_path)
        sid = "dual-file-session"

        json_file = tmp_path / f"{sid}.json"
        enc_file = tmp_path / f"{sid}.enc"
        json_file.write_text("{}")
        enc_file.write_text("encrypted")

        mgr.delete_session(sid)
        assert not json_file.exists()
        assert not enc_file.exists()

    def test_delete_nonexistent_session_does_not_raise(self, tmp_path: Path):
        mgr = SessionManager(sessions_dir=tmp_path)
        # Should not raise any exception
        mgr.delete_session("does-not-exist")


# ═══════════════════════════════════════════════════════════════════════════════
# TestSessionMarkStatus
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionMarkStatus:
    """mark_complete and mark_paused update status and persist."""

    def test_mark_complete(self, tmp_path: Path):
        mgr = SessionManager(sessions_dir=tmp_path)
        session = mgr.create_session("Complete me")
        assert session.status == "active"

        mgr.mark_complete()
        assert mgr.current_session.status == "complete"

        # Verify persistence
        data = json.loads((tmp_path / f"{session.id}.json").read_text())
        assert data["status"] == "complete"

    def test_mark_paused(self, tmp_path: Path):
        mgr = SessionManager(sessions_dir=tmp_path)
        session = mgr.create_session("Pause me")
        assert session.status == "active"

        mgr.mark_paused()
        assert mgr.current_session.status == "paused"

        # Verify persistence
        data = json.loads((tmp_path / f"{session.id}.json").read_text())
        assert data["status"] == "paused"

    def test_mark_complete_without_session_is_noop(self, tmp_path: Path):
        mgr = SessionManager(sessions_dir=tmp_path)
        assert mgr.current_session is None
        # Should not raise
        mgr.mark_complete()
        assert mgr.current_session is None

    def test_mark_paused_without_session_is_noop(self, tmp_path: Path):
        mgr = SessionManager(sessions_dir=tmp_path)
        assert mgr.current_session is None
        mgr.mark_paused()
        assert mgr.current_session is None

    def test_mark_complete_then_resume_preserves_status(self, tmp_path: Path):
        mgr = SessionManager(sessions_dir=tmp_path)
        session = mgr.create_session("Status roundtrip")
        mgr.mark_complete()

        mgr2 = SessionManager(sessions_dir=tmp_path)
        resumed = mgr2.resume_session(session.id)
        assert resumed.status == "complete"


# ═══════════════════════════════════════════════════════════════════════════════
# TestDeriveKey
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeriveKey:
    """SessionManager.derive_key determinism and randomness."""

    def test_same_passphrase_and_salt_produces_same_key(self):
        key1, salt1 = SessionManager.derive_key("my-secret")
        key2, _ = SessionManager.derive_key("my-secret", salt=salt1)
        assert key1 == key2

    def test_different_passphrases_produce_different_keys(self):
        key1, salt = SessionManager.derive_key("passphrase-one")
        key2, _ = SessionManager.derive_key("passphrase-two", salt=salt)
        assert key1 != key2

    def test_random_salt_when_not_provided(self):
        _, salt1 = SessionManager.derive_key("same-pass")
        _, salt2 = SessionManager.derive_key("same-pass")
        # Two calls without an explicit salt should produce different salts
        assert salt1 != salt2

    def test_key_is_url_safe_base64(self):
        key, _ = SessionManager.derive_key("test-passphrase")
        assert isinstance(key, bytes)
        # Fernet keys are 32 bytes base64-encoded → 44 chars with padding
        assert len(key) == 44

    def test_salt_is_16_bytes(self):
        _, salt = SessionManager.derive_key("test-passphrase")
        assert isinstance(salt, bytes)
        assert len(salt) == 16

    def test_explicit_salt_is_returned_unchanged(self):
        explicit_salt = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10"
        _, returned_salt = SessionManager.derive_key("any-pass", salt=explicit_salt)
        assert returned_salt == explicit_salt
