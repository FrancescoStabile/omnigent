"""
Omnigent — Session Persistence

Auto-save sessions, resume work, never lose progress.
Storage: ~/.omnigent/sessions/ as JSON files (or encrypted .enc files).
Includes DomainProfile + TaskPlan for smart resume.

Encryption: Pass `encryption_key` to SessionManager for Fernet-based
encryption of session files. Requires `cryptography` package (optional).
"""

import base64
import hashlib
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from omnigent.state import State

logger = logging.getLogger("omnigent.session")


@dataclass
class Session:
    """Persistent session data."""
    id: str
    timestamp: str
    subject: str | None
    messages: list[dict]
    findings: list[dict]
    cost: float
    tokens_in: int
    tokens_out: int
    status: str  # "active", "paused", "complete"
    domain_profile: dict | None = None
    task_plan: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        known_fields = {
            'id', 'timestamp', 'subject', 'messages', 'findings',
            'cost', 'tokens_in', 'tokens_out', 'status',
            'domain_profile', 'task_plan',
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        filtered.setdefault('domain_profile', None)
        filtered.setdefault('task_plan', None)
        return cls(**filtered)


class SessionManager:
    """Manage persistent sessions.

    Features:
    - Auto-save after every finding
    - Resume interrupted sessions
    - List all sessions
    - Export session history (markdown, JSON, HTML)
    """

    def __init__(
        self,
        sessions_dir: Path | None = None,
        encryption_key: bytes | None = None,
        passphrase: str | None = None,
    ):
        self.sessions_dir = sessions_dir or Path.home() / ".omnigent" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Session | None = None
        self._fernet = None
        self._passphrase = passphrase
        if encryption_key:
            try:
                from cryptography.fernet import Fernet
                self._fernet = Fernet(encryption_key)
                logger.debug("Session encryption enabled (pre-derived key)")
            except ImportError:
                logger.warning("cryptography package not installed, sessions will be stored unencrypted")
        elif passphrase:
            try:
                from cryptography.fernet import Fernet  # noqa: F811
                # For passphrase mode, Fernet is created per-file with random salt
                logger.debug("Session encryption enabled (passphrase mode)")
            except ImportError:
                self._passphrase = None
                logger.warning("cryptography package not installed, sessions will be stored unencrypted")

    SALT_SIZE = 16  # 16-byte random salt

    @staticmethod
    def derive_key(passphrase: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
        """Derive a Fernet-compatible encryption key from a passphrase.

        Uses PBKDF2-HMAC-SHA256 with 100k iterations and a random salt.

        Args:
            passphrase: User passphrase.
            salt: Optional salt bytes. If None, generates a random 16-byte salt.

        Returns:
            Tuple of (fernet_key, salt) — salt must be stored alongside ciphertext.
        """
        if salt is None:
            salt = os.urandom(SessionManager.SALT_SIZE)
        key = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, 100_000)
        return base64.urlsafe_b64encode(key), salt

    def create_session(self, subject: str | None = None) -> Session:
        """Create new session."""
        session = Session(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            subject=subject,
            messages=[],
            findings=[],
            cost=0.0,
            tokens_in=0,
            tokens_out=0,
            status="active",
        )
        self.current_session = session
        self._save(session)
        return session

    def save_state(self, state: State, cost: float = 0.0, tokens_in: int = 0, tokens_out: int = 0):
        """Save current agent state to session."""
        if not self.current_session:
            self.create_session(state.subject)

        self.current_session.subject = state.subject
        self.current_session.messages = [
            {"role": m["role"], "content": self._serialize_content(m["content"])}
            for m in state.messages
        ]
        self.current_session.findings = [
            {
                "title": f.title,
                "severity": f.severity,
                "description": f.description,
                "evidence": f.evidence,
            }
            for f in state.findings
        ]
        self.current_session.cost = cost
        self.current_session.tokens_in = tokens_in
        self.current_session.tokens_out = tokens_out

        if state.profile:
            self.current_session.domain_profile = state.profile.to_dict()
        if state.plan and state.plan.objective:
            self.current_session.task_plan = state.plan.to_dict()

        self._save(self.current_session)

    def _serialize_content(self, content) -> str | list:
        """Serialize message content."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            return [
                {k: v for k, v in item.items() if k in ["type", "id", "name", "input", "tool_call_id", "content"]}
                for item in content
            ]
        else:
            return str(content)

    def resume_session(self, session_id: str) -> Session | None:
        """Resume a previous session. Supports both encrypted and plaintext files."""
        enc_file = self.sessions_dir / f"{session_id}.enc"
        if enc_file.exists():
            # Try pre-derived key first
            if self._fernet:
                try:
                    encrypted = enc_file.read_text()
                    decrypted = self._fernet.decrypt(encrypted.encode()).decode()
                    data = json.loads(decrypted)
                    self.current_session = Session.from_dict(data)
                    return self.current_session
                except Exception as e:
                    logger.error(f"Failed to decrypt session {session_id} with key: {e}")

            # Try passphrase mode (salt prefix)
            if self._passphrase:
                try:
                    from cryptography.fernet import Fernet
                    raw = enc_file.read_bytes()
                    salt = raw[:self.SALT_SIZE]
                    encrypted_data = raw[self.SALT_SIZE:]
                    fernet_key, _ = self.derive_key(self._passphrase, salt=salt)
                    fernet = Fernet(fernet_key)
                    decrypted = fernet.decrypt(encrypted_data).decode()
                    data = json.loads(decrypted)
                    self.current_session = Session.from_dict(data)
                    return self.current_session
                except Exception as e:
                    logger.error(f"Failed to decrypt session {session_id} with passphrase: {e}")
                    return None

        # Fallback to plaintext
        json_file = self.sessions_dir / f"{session_id}.json"
        if not json_file.exists():
            return None

        data = json.loads(json_file.read_text())
        self.current_session = Session.from_dict(data)
        return self.current_session

    def list_sessions(self, limit: int = 10) -> list[Session]:
        """List recent sessions (supports both .json and .enc files)."""
        sessions = []
        all_files = list(self.sessions_dir.glob("*.json")) + list(self.sessions_dir.glob("*.enc"))
        for file in sorted(all_files, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            try:
                if file.suffix == ".enc":
                    if self._fernet:
                        encrypted = file.read_text()
                        decrypted = self._fernet.decrypt(encrypted.encode()).decode()
                        data = json.loads(decrypted)
                    elif self._passphrase:
                        from cryptography.fernet import Fernet
                        raw = file.read_bytes()
                        salt = raw[:self.SALT_SIZE]
                        encrypted_data = raw[self.SALT_SIZE:]
                        fernet_key, _ = self.derive_key(self._passphrase, salt=salt)
                        fernet = Fernet(fernet_key)
                        decrypted = fernet.decrypt(encrypted_data).decode()
                        data = json.loads(decrypted)
                    else:
                        continue
                elif file.suffix == ".json":
                    data = json.loads(file.read_text())
                else:
                    continue
                sessions.append(Session.from_dict(data))
            except Exception:
                continue
        return sessions

    def get_last_session(self) -> Session | None:
        """Get most recent session."""
        sessions = self.list_sessions(limit=1)
        return sessions[0] if sessions else None

    def mark_complete(self):
        """Mark current session as complete."""
        if self.current_session:
            self.current_session.status = "complete"
            self._save(self.current_session)

    def mark_paused(self):
        """Mark current session as paused."""
        if self.current_session:
            self.current_session.status = "paused"
            self._save(self.current_session)

    def delete_session(self, session_id: str):
        """Delete a session (both encrypted and plaintext)."""
        for ext in (".json", ".enc"):
            session_file = self.sessions_dir / f"{session_id}{ext}"
            if session_file.exists():
                session_file.unlink()

    def _save(self, session: Session):
        """Save session to disk. Uses encryption if configured."""
        data = json.dumps(session.to_dict(), indent=2)
        if self._fernet:
            encrypted = self._fernet.encrypt(data.encode()).decode()
            session_file = self.sessions_dir / f"{session.id}.enc"
            session_file.write_text(encrypted)
            plain_file = self.sessions_dir / f"{session.id}.json"
            if plain_file.exists():
                plain_file.unlink()
        elif self._passphrase:
            try:
                from cryptography.fernet import Fernet
                fernet_key, salt = self.derive_key(self._passphrase)
                fernet = Fernet(fernet_key)
                encrypted = fernet.encrypt(data.encode())
                # Store salt (16 bytes) + encrypted data
                session_file = self.sessions_dir / f"{session.id}.enc"
                session_file.write_bytes(salt + encrypted)
                plain_file = self.sessions_dir / f"{session.id}.json"
                if plain_file.exists():
                    plain_file.unlink()
            except ImportError:
                session_file = self.sessions_dir / f"{session.id}.json"
                session_file.write_text(data)
        else:
            session_file = self.sessions_dir / f"{session.id}.json"
            session_file.write_text(data)

    def export_session(self, session_id: str, format: str = "md") -> str:
        """Export session in various formats."""
        session = self.resume_session(session_id)
        if not session:
            return ""

        if format == "md":
            return self._export_markdown(session)
        elif format == "json":
            return json.dumps(session.to_dict(), indent=2)
        elif format == "html":
            return self._export_html(session)
        return ""

    def _export_markdown(self, session: Session) -> str:
        """Export as Markdown."""
        md = "# Omnigent Session Report\n\n"
        md += f"**Session ID**: {session.id}\n"
        md += f"**Date**: {session.timestamp}\n"
        md += f"**Subject**: {session.subject or 'N/A'}\n"
        md += f"**Status**: {session.status}\n"
        md += f"**Cost**: ${session.cost:.4f}\n"
        md += f"**Tokens**: {session.tokens_in:,} in / {session.tokens_out:,} out\n\n"

        md += f"## Findings ({len(session.findings)})\n\n"

        for i, finding in enumerate(session.findings, 1):
            md += f"### {i}. [{finding['severity'].upper()}] {finding['title']}\n\n"
            md += f"**Description**: {finding['description']}\n\n"
            if finding.get('evidence'):
                md += f"**Evidence**:\n```\n{finding['evidence']}\n```\n\n"
            md += "---\n\n"

        return md

    @staticmethod
    def _escape(text: str) -> str:
        """Escape HTML special characters."""
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )

    # ──────────────────────────────────────────────────────────
    # Checkpoint / Replay
    # ──────────────────────────────────────────────────────────

    def save_checkpoint(self, state: State, iteration: int, label: str = "") -> str:
        """Save a mid-execution checkpoint for replay capability.

        Checkpoints are snapshots of the full agent state at a given iteration.
        They allow resuming from a specific point or rolling back on failure.

        Args:
            state: Current agent state to snapshot.
            iteration: Current iteration number.
            label: Optional human-readable label.

        Returns:
            Checkpoint ID.
        """
        session_id = self.current_session.id if self.current_session else "orphan"
        checkpoint_id = f"{session_id}_cp{iteration}"

        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "session_id": session_id,
            "iteration": iteration,
            "label": label or f"Iteration {iteration}",
            "timestamp": datetime.now().isoformat(),
            "messages": [
                {"role": m["role"], "content": self._serialize_content(m["content"])}
                for m in state.messages
            ],
            "findings": [
                {
                    "title": f.title,
                    "severity": f.severity,
                    "description": f.description,
                    "evidence": f.evidence,
                }
                for f in state.findings
            ],
            "profile": state.profile.to_dict() if state.profile else {},
            "plan": state.plan.to_dict() if state.plan and state.plan.objective else {},
            "iteration_count": state.iteration,
        }

        checkpoint_dir = self.sessions_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = checkpoint_dir / f"{checkpoint_id}.json"
        checkpoint_file.write_text(json.dumps(checkpoint, indent=2))

        logger.debug(f"Checkpoint saved: {checkpoint_id} ({label})")
        return checkpoint_id

    def list_checkpoints(self, session_id: str | None = None) -> list[dict]:
        """List all checkpoints, optionally filtered by session.

        Returns list of dicts with: checkpoint_id, session_id, iteration, label, timestamp.
        """
        checkpoint_dir = self.sessions_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return []

        checkpoints = []
        for f in sorted(checkpoint_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
            try:
                data = json.loads(f.read_text())
                if session_id and data.get("session_id") != session_id:
                    continue
                checkpoints.append({
                    "checkpoint_id": data.get("checkpoint_id", ""),
                    "session_id": data.get("session_id", ""),
                    "iteration": data.get("iteration", 0),
                    "label": data.get("label", ""),
                    "timestamp": data.get("timestamp", ""),
                })
            except Exception:
                continue
        return checkpoints

    def restore_checkpoint(self, checkpoint_id: str, state: State) -> bool:
        """Restore agent state from a checkpoint.

        Args:
            checkpoint_id: The checkpoint to restore.
            state: The State object to restore into.

        Returns:
            True if successful, False otherwise.
        """
        from omnigent.domain_profile import DomainProfile
        from omnigent.planner import TaskPlan
        from omnigent.state import Finding

        checkpoint_dir = self.sessions_dir / "checkpoints"
        checkpoint_file = checkpoint_dir / f"{checkpoint_id}.json"

        if not checkpoint_file.exists():
            logger.error(f"Checkpoint not found: {checkpoint_id}")
            return False

        try:
            data = json.loads(checkpoint_file.read_text())

            # Restore messages
            state.messages = data.get("messages", [])

            # Restore findings
            state.findings = []
            for f_data in data.get("findings", []):
                state.findings.append(Finding(
                    title=f_data.get("title", ""),
                    severity=f_data.get("severity", "info"),
                    description=f_data.get("description", ""),
                    evidence=f_data.get("evidence", ""),
                ))

            # Restore profile
            profile_data = data.get("profile", {})
            if profile_data:
                state.profile = DomainProfile.from_dict(profile_data)
            else:
                state.profile = DomainProfile()

            # Restore plan
            plan_data = data.get("plan", {})
            if plan_data:
                state.plan = TaskPlan.from_dict(plan_data)

            state.iteration = data.get("iteration_count", 0)

            logger.info(f"Checkpoint restored: {checkpoint_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
            return False

    def delete_checkpoints(self, session_id: str) -> int:
        """Delete all checkpoints for a session. Returns count deleted."""
        checkpoint_dir = self.sessions_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return 0

        count = 0
        for f in checkpoint_dir.glob(f"{session_id}_cp*.json"):
            f.unlink()
            count += 1
        return count

    def _export_html(self, session: Session) -> str:
        """Export as HTML."""
        esc = self._escape
        subject = esc(session.subject or 'N/A')
        sid = esc(session.id)
        ts = esc(str(session.timestamp))

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Omnigent Report - {subject}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
        .header {{ border-bottom: 2px solid #4cc9f0; padding-bottom: 10px; margin-bottom: 20px; }}
        .finding {{ border: 1px solid #4cc9f0; padding: 15px; margin: 10px 0; border-radius: 4px; }}
        .critical {{ border-color: #ff0051; }}
        .high {{ border-color: #ff6b35; }}
        .medium {{ border-color: #ffd700; }}
        .severity {{ font-weight: bold; font-size: 1.2em; }}
        pre {{ background: #111; padding: 10px; border-left: 3px solid #4cc9f0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Omnigent Session Report</h1>
        <p><strong>Session ID:</strong> {sid}</p>
        <p><strong>Date:</strong> {ts}</p>
        <p><strong>Subject:</strong> {subject}</p>
        <p><strong>Cost:</strong> ${session.cost:.4f}</p>
    </div>
    <h2>Findings ({len(session.findings)})</h2>
"""

        for i, finding in enumerate(session.findings, 1):
            severity_class = esc(finding['severity'].lower())
            title = esc(finding['title'])
            desc = esc(finding['description'])
            sev_label = esc(finding['severity'].upper())
            html += f"""
    <div class="finding {severity_class}">
        <div class="severity">[{sev_label}]</div>
        <h3>{i}. {title}</h3>
        <p><strong>Description:</strong> {desc}</p>
"""
            if finding.get('evidence'):
                html += f"<p><strong>Evidence:</strong></p><pre>{esc(finding['evidence'])}</pre>"
            html += "    </div>\n"

        html += "\n</body>\n</html>"
        return html
