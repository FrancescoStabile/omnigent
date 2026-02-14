"""Tests for Plugin System."""

import json

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


class TestManifestValidation:
    """Tests for plugin.json manifest validation."""

    def test_valid_manifest(self, tmp_path):
        pm = PluginManager(plugin_dir=tmp_path)
        data = {"name": "test", "version": "1.0.0", "type": "tool"}
        assert pm._validate_manifest(data, tmp_path) is True

    def test_missing_required_field(self, tmp_path):
        pm = PluginManager(plugin_dir=tmp_path)
        data = {"name": "test"}  # missing version, type
        assert pm._validate_manifest(data, tmp_path) is False

    def test_invalid_type(self, tmp_path):
        pm = PluginManager(plugin_dir=tmp_path)
        data = {"name": "test", "version": "1.0.0", "type": "invalid_type"}
        assert pm._validate_manifest(data, tmp_path) is False

    def test_all_valid_types(self, tmp_path):
        pm = PluginManager(plugin_dir=tmp_path)
        for ptype in ("tool", "extractor", "knowledge", "all"):
            data = {"name": "test", "version": "1.0.0", "type": ptype}
            assert pm._validate_manifest(data, tmp_path) is True

    def test_discover_skips_invalid_manifest(self, tmp_path):
        """Plugin with invalid manifest should be skipped during discovery."""
        plugin_dir = tmp_path / "bad_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("PLUGIN_META = None\n")
        (plugin_dir / "plugin.json").write_text('{"name": "bad"}')  # missing version, type

        pm = PluginManager(plugin_dir=tmp_path)
        discovered = pm.discover()
        names = [d.name for d in discovered]
        assert "bad_plugin" not in names


class TestChecksumVerification:
    """Tests for SHA-256 checksum verification."""

    def test_no_checksums_file_passes(self, tmp_path):
        """Without checksums.json, verification should pass (backward compat)."""
        pm = PluginManager(plugin_dir=tmp_path)
        assert pm._verify_checksums(tmp_path) is True

    def test_valid_checksums_pass(self, tmp_path):
        """Files matching their checksums should pass."""
        import hashlib
        content = b"print('hello')"
        (tmp_path / "tool.py").write_bytes(content)
        expected_hash = hashlib.sha256(content).hexdigest()
        checksums = {"tool.py": expected_hash}
        (tmp_path / "checksums.json").write_text(json.dumps(checksums))

        pm = PluginManager(plugin_dir=tmp_path)
        assert pm._verify_checksums(tmp_path) is True

    def test_invalid_checksum_fails(self, tmp_path):
        """Modified file should fail checksum verification."""
        (tmp_path / "tool.py").write_bytes(b"modified content")
        checksums = {"tool.py": "0" * 64}  # wrong hash
        (tmp_path / "checksums.json").write_text(json.dumps(checksums))

        pm = PluginManager(plugin_dir=tmp_path)
        assert pm._verify_checksums(tmp_path) is False

    def test_missing_file_fails(self, tmp_path):
        """Checksums referencing non-existent file should fail."""
        checksums = {"missing.py": "abc123"}
        (tmp_path / "checksums.json").write_text(json.dumps(checksums))

        pm = PluginManager(plugin_dir=tmp_path)
        assert pm._verify_checksums(tmp_path) is False

    def test_malformed_checksums_json_fails(self, tmp_path):
        """Invalid JSON in checksums.json should fail."""
        (tmp_path / "checksums.json").write_text("{bad json")

        pm = PluginManager(plugin_dir=tmp_path)
        assert pm._verify_checksums(tmp_path) is False

    def test_non_dict_checksums_fails(self, tmp_path):
        """checksums.json must be a dict, not a list."""
        (tmp_path / "checksums.json").write_text('["not", "a", "dict"]')

        pm = PluginManager(plugin_dir=tmp_path)
        assert pm._verify_checksums(tmp_path) is False

    def test_checksum_blocks_plugin_load(self, tmp_path):
        """Plugin with failed checksums should not be loaded."""
        import hashlib

        plugin_dir = tmp_path / "evil_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text(
            'PLUGIN_META = {"name": "evil", "version": "1.0.0", "type": "tool"}\n'
        )
        (plugin_dir / "plugin.json").write_text(
            '{"name": "evil", "version": "1.0.0", "type": "tool"}'
        )
        # Write a file and set wrong checksum
        (plugin_dir / "tool.py").write_bytes(b"tampered content")
        checksums = {"tool.py": "0" * 64}
        (plugin_dir / "checksums.json").write_text(json.dumps(checksums))

        pm = PluginManager(plugin_dir=tmp_path)
        pm.discover()
        pm.load_all()
        loaded_names = [p.meta.name for p in pm.loaded]
        assert "evil" not in loaded_names
