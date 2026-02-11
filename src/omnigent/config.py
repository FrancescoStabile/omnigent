"""
Omnigent — Config Management

Unified configuration loading from multiple sources:
1. ~/.omnigent/config.yaml (persistent, recommended)
2. .env file (project-local)
3. Environment variables (override)

Priority: ENV > .env > config.yaml
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# ═══════════════════════════════════════════════════════════════════════════
# Config Paths
# ═══════════════════════════════════════════════════════════════════════════

OMNIGENT_HOME = Path.home() / ".omnigent"
CONFIG_FILE = OMNIGENT_HOME / "config.yaml"
SESSIONS_DIR = OMNIGENT_HOME / "sessions"


# ═══════════════════════════════════════════════════════════════════════════
# Config Loader
# ═══════════════════════════════════════════════════════════════════════════


class Config:
    """Unified configuration management."""

    # API key names to auto-detect.  Domain implementations can extend this.
    API_KEY_NAMES: list[str] = [
        "DEEPSEEK_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ]

    def __init__(self):
        self.data: dict[str, Any] = {}
        self._load()

    def _load(self):
        """Load config from all sources (priority: ENV > .env > config.yaml)."""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                self.data = yaml.safe_load(f) or {}

        self._load_dotenv()
        self._apply_env_overrides()

    def _load_dotenv(self):
        """Load .env file from cwd or parent directories."""
        check = Path.cwd()
        for _ in range(5):
            env_file = check / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if "=" in line and not line.strip().startswith("#"):
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key not in os.environ:
                            self.data.setdefault(key, value)
                return
            check = check.parent

    def _apply_env_overrides(self):
        """Environment variables override config file."""
        for key in self.API_KEY_NAMES:
            if key in os.environ:
                self.data[key] = os.environ[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        """Set config value (in-memory only)."""
        self.data[key] = value

    def save(self):
        """Save config to ~/.omnigent/config.yaml."""
        try:
            OMNIGENT_HOME.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, "w") as f:
                yaml.dump(self.data, f, default_flow_style=False)
        except (OSError, PermissionError) as e:
            import sys
            print(f"[!] Warning: Could not save config to {CONFIG_FILE}: {e}", file=sys.stderr)
            print(f"[i] Config will work for this session only.", file=sys.stderr)

    def has_api_key(self) -> bool:
        """Check if at least one API key is configured."""
        return any(self.get(key) for key in self.API_KEY_NAMES)

    def get_api_keys(self) -> dict[str, str]:
        """Get all configured API keys."""
        keys = {}
        for key in self.API_KEY_NAMES:
            value = self.get(key)
            if value:
                keys[key] = value
        return keys


# ═══════════════════════════════════════════════════════════════════════════
# Interactive Setup
# ═══════════════════════════════════════════════════════════════════════════


def interactive_setup() -> Config:
    """Interactive first-time setup."""
    from rich.console import Console
    from rich.prompt import Prompt

    console = Console(color_system="truecolor")
    config = Config()

    console.print("\n[bold cyan]Omnigent Setup[/]")
    console.print("One API key is needed to start.\n")

    console.print("[bold #00ff41]DeepSeek[/]")
    console.print("[dim]  → https://platform.deepseek.com/api_keys[/]\n")
    deepseek_key = Prompt.ask(
        "DeepSeek API Key",
        default=config.get("DEEPSEEK_API_KEY", ""),
        password=True,
    )
    if deepseek_key:
        config.set("DEEPSEEK_API_KEY", deepseek_key)

    console.print("\n[bold yellow]Anthropic Claude[/]")
    console.print("[dim]  → https://console.anthropic.com/settings/keys[/]\n")
    claude_key = Prompt.ask(
        "Claude API Key (Enter to skip)",
        default=config.get("ANTHROPIC_API_KEY", ""),
        password=True,
    )
    if claude_key:
        config.set("ANTHROPIC_API_KEY", claude_key)

    console.print("\n[bold blue]OpenAI[/]")
    console.print("[dim]  → https://platform.openai.com/api-keys[/]\n")
    openai_key = Prompt.ask(
        "OpenAI API Key (Enter to skip)",
        default=config.get("OPENAI_API_KEY", ""),
        password=True,
    )
    if openai_key:
        config.set("OPENAI_API_KEY", openai_key)

    if not config.has_api_key():
        console.print("\n[bold yellow]No API keys configured.[/]")
        console.print("Set a key: [dim]export DEEPSEEK_API_KEY=\"sk-...\"[/]\n")
        return config

    config.save()

    if CONFIG_FILE.exists():
        console.print(f"\n[bold #00ff41]✓ Config saved:[/] {CONFIG_FILE}")
        console.print("Edit anytime: [cyan]~/.omnigent/config.yaml[/]\n")
    else:
        console.print(f"\n[bold yellow]✓ Config loaded (session-only)[/]")
        console.print("Use environment variables for persistent config.\n")

    return config


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def load_config() -> Config:
    """Load config from all sources."""
    return Config()


def ensure_config() -> Config:
    """Ensure config exists, run interactive setup if needed."""
    config = Config()

    if config.has_api_key():
        return config

    config = interactive_setup()
    return config
