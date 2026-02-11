"""Tests for Plugin System."""

import pytest
from pathlib import Path
from omnigent.plugins import PluginMeta, LoadedPlugin, PluginManager, PLUGIN_DIR


class TestPluginMeta:
    def test_creation(self):
        meta = PluginMeta(name="test_plugin")
        assert meta.name == "test_plugin"
        assert meta.version == "0.1.0"
        assert meta.enabled is True

    def test_to_dict(self):
        meta = PluginMeta(name="test", version="1.0.0", description="A test plugin")
        d = meta.to_dict()
        assert d["name"] == "test"
        assert d["version"] == "1.0.0"


class TestLoadedPlugin:
    def test_creation(self):
        meta = PluginMeta(name="test")
        lp = LoadedPlugin(meta=meta)
        assert lp.tools == {}
        assert lp.extractors == {}
        assert lp.knowledge_files == []


class TestPluginManager:
    def test_creation(self):
        pm = PluginManager()
        assert isinstance(pm, PluginManager)

    def test_default_plugin_dir(self):
        assert PLUGIN_DIR == Path.home() / ".omnigent" / "plugins"

    def test_discover_empty_dir(self, tmp_path):
        """Should not crash on empty/nonexistent plugin dir."""
        pm = PluginManager(plugin_dir=tmp_path / "nonexistent")
        pm.discover()
        assert len(pm.discovered) == 0
