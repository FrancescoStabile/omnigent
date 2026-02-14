"""
Omnigent — Plugin System

Extensible architecture for loading custom tools, extractors, and knowledge.

Plugin types:
  - tool: Adds new tool functions + schemas to ToolRegistry
  - extractor: Adds new extractors to the extraction pipeline
  - knowledge: Adds knowledge files for context injection

Discovery:
  - ~/.omnigent/plugins/ directory
  - Each plugin is a Python package (directory with __init__.py)
  - Plugin metadata in plugin.json or __init__.py PLUGIN_META dict

Plugin structure example:
  ~/.omnigent/plugins/
    my_tool/
      __init__.py          # Must define PLUGIN_META dict
      plugin.json          # Optional: metadata override
      tool.py              # Tool implementation
      extractor.py         # Optional: custom extractor
      knowledge/           # Optional: knowledge files
        my_cheatsheet.md
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("omnigent.plugins")

# Default plugin directory
PLUGIN_DIR = Path.home() / ".omnigent" / "plugins"


# ═══════════════════════════════════════════════════════════════════════════
# Plugin Metadata
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PluginMeta:
    """Plugin metadata."""
    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    plugin_type: str = "tool"  # tool | extractor | knowledge
    enabled: bool = True
    path: Path = field(default_factory=lambda: Path("."))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "type": self.plugin_type,
            "enabled": self.enabled,
            "path": str(self.path),
        }


@dataclass
class LoadedPlugin:
    """A loaded plugin with its components."""
    meta: PluginMeta
    module: Any = None
    tools: dict[str, Callable] = field(default_factory=dict)
    tool_schemas: dict[str, dict] = field(default_factory=dict)
    extractors: dict[str, Callable] = field(default_factory=dict)
    knowledge_files: list[Path] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Plugin Manager
# ═══════════════════════════════════════════════════════════════════════════


class PluginManager:
    """Discovers, loads, and manages Omnigent plugins.

    Usage:
        pm = PluginManager()
        pm.discover()
        pm.load_all()

        # Register tools
        for plugin in pm.loaded:
            for name, func in plugin.tools.items():
                tool_registry.register(name, func, plugin.tool_schemas.get(name, {}))
    """

    def __init__(self, plugin_dir: Path | str | None = None, strict_checksums: bool = False):
        self.plugin_dir = Path(plugin_dir) if plugin_dir else PLUGIN_DIR
        self.strict_checksums = strict_checksums
        self.discovered: list[PluginMeta] = []
        self.loaded: list[LoadedPlugin] = []
        self._errors: list[tuple[str, str]] = []

    # Required fields in plugin.json for validation
    REQUIRED_MANIFEST_FIELDS = {"name", "version", "type"}
    VALID_PLUGIN_TYPES = {"tool", "extractor", "knowledge", "all"}

    def ensure_plugin_dir(self) -> Path:
        """Create plugin directory if it doesn't exist."""
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        return self.plugin_dir

    def _validate_manifest(self, data: dict, plugin_dir: Path) -> bool:
        """Validate plugin manifest has required fields and valid values."""
        missing = self.REQUIRED_MANIFEST_FIELDS - set(data.keys())
        if missing:
            logger.warning(f"Plugin {plugin_dir.name}: manifest missing required fields: {missing}")
            return False
        ptype = data.get("type", "")
        if ptype not in self.VALID_PLUGIN_TYPES:
            logger.warning(f"Plugin {plugin_dir.name}: invalid type '{ptype}', expected one of {self.VALID_PLUGIN_TYPES}")
            return False
        return True

    def _verify_checksums(self, plugin_dir: Path) -> bool:
        """Verify plugin file SHA-256 checksums if checksums.json exists.

        In strict mode (strict_checksums=True), plugins without checksums.json
        are REJECTED. In normal mode, they are loaded with a warning (backward compatible).
        """
        checksums_file = plugin_dir / "checksums.json"
        if not checksums_file.exists():
            if self.strict_checksums:
                logger.warning(
                    f"Plugin {plugin_dir.name}: no checksums.json found "
                    f"(strict mode enabled — plugin rejected)"
                )
                return False
            return True  # No checksums = trust (backward compatible)

        try:
            checksums = json.loads(checksums_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Invalid checksums.json in {plugin_dir.name}: {e}")
            return False

        if not isinstance(checksums, dict):
            logger.warning(f"Plugin {plugin_dir.name}: checksums.json must be a dict")
            return False

        for filename, expected_hash in checksums.items():
            file_path = plugin_dir / filename
            if not file_path.exists():
                logger.warning(f"Plugin {plugin_dir.name}: missing file referenced in checksums: {filename}")
                return False
            actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                logger.error(
                    f"Plugin {plugin_dir.name}: SHA-256 hash mismatch for {filename} "
                    f"(expected {expected_hash[:16]}..., got {actual_hash[:16]}...)"
                )
                return False

        logger.debug(f"Plugin {plugin_dir.name}: checksum verification passed ({len(checksums)} files)")
        return True

    def discover(self) -> list[PluginMeta]:
        """Discover plugins in the plugin directory."""
        self.discovered.clear()

        if not self.plugin_dir.exists():
            logger.debug(f"Plugin directory does not exist: {self.plugin_dir}")
            return self.discovered

        for item in sorted(self.plugin_dir.iterdir()):
            if not item.is_dir():
                continue

            init_file = item / "__init__.py"
            if not init_file.exists():
                continue

            try:
                meta = self._read_plugin_meta(item)
                if meta:
                    self.discovered.append(meta)
                    logger.info(f"Discovered plugin: {meta.name} v{meta.version} ({meta.plugin_type})")
            except Exception as e:
                logger.warning(f"Failed to read plugin metadata from {item.name}: {e}")
                self._errors.append((item.name, str(e)))

        return self.discovered

    def _read_plugin_meta(self, plugin_dir: Path) -> PluginMeta | None:
        """Read plugin metadata from plugin.json or __init__.py."""
        # Try plugin.json first
        json_file = plugin_dir / "plugin.json"
        if json_file.exists():
            try:
                data = json.loads(json_file.read_text())
                if not self._validate_manifest(data, plugin_dir):
                    logger.warning(f"Plugin {plugin_dir.name}: manifest validation failed, skipping")
                    return None
                return PluginMeta(
                    name=data.get("name", plugin_dir.name),
                    version=data.get("version", "0.1.0"),
                    description=data.get("description", ""),
                    author=data.get("author", ""),
                    plugin_type=data.get("type", "tool"),
                    enabled=data.get("enabled", True),
                    path=plugin_dir,
                )
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid plugin.json in {plugin_dir.name}: {e}")

        # Fallback: __init__.py PLUGIN_META
        init_file = plugin_dir / "__init__.py"
        try:
            spec = importlib.util.spec_from_file_location(
                f"omnigent_plugin_{plugin_dir.name}", init_file
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                plugin_meta = getattr(module, "PLUGIN_META", None)
                if plugin_meta and isinstance(plugin_meta, dict):
                    return PluginMeta(
                        name=plugin_meta.get("name", plugin_dir.name),
                        version=plugin_meta.get("version", "0.1.0"),
                        description=plugin_meta.get("description", ""),
                        author=plugin_meta.get("author", ""),
                        plugin_type=plugin_meta.get("type", "tool"),
                        enabled=plugin_meta.get("enabled", True),
                        path=plugin_dir,
                    )
        except Exception as e:
            logger.debug(f"Could not load __init__.py for {plugin_dir.name}: {e}")

        # Minimal metadata from directory name
        return PluginMeta(name=plugin_dir.name, path=plugin_dir)

    def load_all(self) -> list[LoadedPlugin]:
        """Load all discovered and enabled plugins."""
        if not self.discovered:
            self.discover()

        self.loaded.clear()

        for meta in self.discovered:
            if not meta.enabled:
                logger.info(f"Skipping disabled plugin: {meta.name}")
                continue

            try:
                plugin = self._load_plugin(meta)
                if plugin:
                    self.loaded.append(plugin)
                    logger.info(
                        f"Loaded plugin: {meta.name} "
                        f"(tools: {len(plugin.tools)}, "
                        f"extractors: {len(plugin.extractors)}, "
                        f"knowledge: {len(plugin.knowledge_files)})"
                    )
            except Exception as e:
                logger.error(f"Failed to load plugin {meta.name}: {e}", exc_info=True)
                self._errors.append((meta.name, str(e)))

        return self.loaded

    def _load_plugin(self, meta: PluginMeta) -> LoadedPlugin | None:
        """Load a single plugin."""
        # Verify checksums before loading any code
        if not self._verify_checksums(meta.path):
            logger.error(f"Checksum verification failed for plugin {meta.name}, skipping")
            self._errors.append((meta.name, "Checksum verification failed"))
            return None

        plugin = LoadedPlugin(meta=meta)

        plugin_path = str(meta.path)
        if plugin_path not in sys.path:
            sys.path.insert(0, plugin_path)

        try:
            if meta.plugin_type in ("tool", "all"):
                self._load_tool_module(meta, plugin)
            if meta.plugin_type in ("extractor", "all"):
                self._load_extractor_module(meta, plugin)
            if meta.plugin_type in ("knowledge", "all"):
                self._load_knowledge_files(meta, plugin)
            self._load_from_init(meta, plugin)
        finally:
            if plugin_path in sys.path:
                sys.path.remove(plugin_path)

        return plugin

    def _load_tool_module(self, meta: PluginMeta, plugin: LoadedPlugin):
        """Load tool.py from plugin directory."""
        tool_file = meta.path / "tool.py"
        if not tool_file.exists():
            return

        try:
            spec = importlib.util.spec_from_file_location(
                f"omnigent_plugin_{meta.name}_tool", tool_file
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                plugin.module = module

                tools = getattr(module, "TOOLS", {})
                if isinstance(tools, dict):
                    plugin.tools.update(tools)

                schemas = getattr(module, "TOOL_SCHEMAS", {})
                if isinstance(schemas, dict):
                    plugin.tool_schemas.update(schemas)

        except Exception as e:
            logger.warning(f"Failed to load tool.py from {meta.name}: {e}")

    def _load_extractor_module(self, meta: PluginMeta, plugin: LoadedPlugin):
        """Load extractor.py from plugin directory."""
        extractor_file = meta.path / "extractor.py"
        if not extractor_file.exists():
            return

        try:
            spec = importlib.util.spec_from_file_location(
                f"omnigent_plugin_{meta.name}_extractor", extractor_file
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                extractors = getattr(module, "EXTRACTORS", {})
                if isinstance(extractors, dict):
                    plugin.extractors.update(extractors)

        except Exception as e:
            logger.warning(f"Failed to load extractor.py from {meta.name}: {e}")

    def _load_knowledge_files(self, meta: PluginMeta, plugin: LoadedPlugin):
        """Discover knowledge markdown files."""
        knowledge_dir = meta.path / "knowledge"
        if knowledge_dir.exists():
            for md_file in sorted(knowledge_dir.glob("**/*.md")):
                plugin.knowledge_files.append(md_file)

    def _load_from_init(self, meta: PluginMeta, plugin: LoadedPlugin):
        """Load tools/extractors exported from __init__.py."""
        init_file = meta.path / "__init__.py"
        if not init_file.exists():
            return

        try:
            spec = importlib.util.spec_from_file_location(
                f"omnigent_plugin_{meta.name}_init", init_file
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                register_fn = getattr(module, "register", None)
                if callable(register_fn):
                    result = register_fn()
                    if isinstance(result, dict):
                        if "tools" in result:
                            plugin.tools.update(result["tools"])
                        if "schemas" in result:
                            plugin.tool_schemas.update(result["schemas"])
                        if "extractors" in result:
                            plugin.extractors.update(result["extractors"])

        except Exception as e:
            logger.debug(f"Could not load from __init__.py for {meta.name}: {e}")

    # ──────────────────────────────────────────────────────────
    # Registration helpers
    # ──────────────────────────────────────────────────────────

    def register_tools(self, tool_registry) -> int:
        """Register all plugin tools into the ToolRegistry."""
        count = 0
        for plugin in self.loaded:
            for name, func in plugin.tools.items():
                schema = plugin.tool_schemas.get(name, {})
                tool_registry.register(name, func, schema)
                logger.info(f"Registered plugin tool: {name} (from {plugin.meta.name})")
                count += 1
        return count

    def register_extractors(self, extractors_dict: dict) -> int:
        """Register all plugin extractors into the extractors dispatch dict."""
        count = 0
        for plugin in self.loaded:
            for tool_name, extractor_fn in plugin.extractors.items():
                extractors_dict[tool_name] = extractor_fn
                logger.info(f"Registered plugin extractor: {tool_name} (from {plugin.meta.name})")
                count += 1
        return count

    def get_knowledge_files(self) -> list[Path]:
        """Get all knowledge files from all loaded plugins."""
        files = []
        for plugin in self.loaded:
            files.extend(plugin.knowledge_files)
        return files

    def get_errors(self) -> list[tuple[str, str]]:
        """Get list of (plugin_name, error_message) for failed plugins."""
        return self._errors.copy()

    def list_plugins(self) -> list[dict]:
        """List all discovered plugins with their status."""
        result = []
        loaded_names = {p.meta.name for p in self.loaded}

        for meta in self.discovered:
            status = "loaded" if meta.name in loaded_names else "disabled" if not meta.enabled else "error"
            result.append({**meta.to_dict(), "status": status})

        return result


# ═══════════════════════════════════════════════════════════════════════════
# Plugin Scaffold Helper
# ═══════════════════════════════════════════════════════════════════════════


def scaffold_plugin(name: str, plugin_type: str = "tool", plugin_dir: Path | None = None) -> Path:
    """Create a new plugin scaffold.

    Args:
        name: Plugin name
        plugin_type: "tool", "extractor", or "knowledge"
        plugin_dir: Override plugin directory

    Returns:
        Path to created plugin directory
    """
    base = plugin_dir or PLUGIN_DIR
    base.mkdir(parents=True, exist_ok=True)
    pdir = base / name

    if pdir.exists():
        raise FileExistsError(f"Plugin directory already exists: {pdir}")

    pdir.mkdir()

    # plugin.json
    meta = {
        "name": name,
        "version": "0.1.0",
        "description": f"Omnigent plugin: {name}",
        "author": "",
        "type": plugin_type,
        "enabled": True,
    }
    (pdir / "plugin.json").write_text(json.dumps(meta, indent=2))

    # __init__.py
    (pdir / "__init__.py").write_text(f'''"""
Omnigent Plugin: {name}
"""

PLUGIN_META = {{
    "name": "{name}",
    "version": "0.1.0",
    "description": "Omnigent plugin: {name}",
    "type": "{plugin_type}",
}}
''')

    if plugin_type in ("tool", "all"):
        (pdir / "tool.py").write_text(f'''"""
Tool implementation for {name} plugin.
"""


async def my_tool(param1: str, param2: str = "") -> str:
    """Your tool implementation."""
    return f"Result from {name}: {{param1}}"


TOOLS = {{
    "{name}": my_tool,
}}

TOOL_SCHEMAS = {{
    "{name}": {{
        "name": "{name}",
        "description": "Description of what {name} does",
        "input_schema": {{
            "type": "object",
            "properties": {{
                "param1": {{"type": "string", "description": "First parameter"}},
                "param2": {{"type": "string", "description": "Optional second parameter"}},
            }},
            "required": ["param1"],
        }},
    }},
}}
''')

    if plugin_type in ("extractor", "all"):
        (pdir / "extractor.py").write_text(f'''"""
Extractor implementation for {name} plugin.
"""


def extract_{name}(raw_output: str, profile) -> None:
    """Extract structured data from {name} tool output."""
    pass


EXTRACTORS = {{
    "{name}": extract_{name},
}}
''')

    if plugin_type in ("knowledge", "all"):
        kdir = pdir / "knowledge"
        kdir.mkdir()
        (kdir / f"{name}_cheatsheet.md").write_text(f"""# {name.title()} Cheatsheet

## Quick Reference

- TODO: Add your knowledge here
""")

    logger.info(f"Plugin scaffold created: {pdir}")
    return pdir


# ═══════════════════════════════════════════════════════════════════════════
# Convenience
# ═══════════════════════════════════════════════════════════════════════════


def load_plugins(plugin_dir: Path | str | None = None) -> PluginManager:
    """Load all plugins from directory."""
    pm = PluginManager(plugin_dir)
    pm.discover()
    pm.load_all()
    return pm
