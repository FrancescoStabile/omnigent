"""Tests for Tool Registry."""

import asyncio
import json
import pytest
from omnigent.tools import ToolRegistry, CREATE_FINDING_SCHEMA


class TestToolRegistry:
    def test_register_and_list(self):
        reg = ToolRegistry()
        async def my_tool(**kwargs):
            return "result"
        reg.register("my_tool", my_tool, {"description": "Test tool"})
        assert "my_tool" in reg.list_tools()

    @pytest.mark.asyncio
    async def test_call_async_tool(self):
        reg = ToolRegistry()
        async def my_tool(arg1: str = ""):
            return f"got {arg1}"
        reg.register("my_tool", my_tool, {"description": "Test"})
        result = await reg.call("my_tool", {"arg1": "hello"})
        assert "got hello" in result

    @pytest.mark.asyncio
    async def test_call_sync_tool(self):
        reg = ToolRegistry()
        def my_tool(arg1: str = ""):
            return f"got {arg1}"
        reg.register("my_tool", my_tool, {"description": "Test"})
        result = await reg.call("my_tool", {"arg1": "hello"})
        assert "got hello" in result

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self):
        reg = ToolRegistry()
        result = await reg.call("nonexistent", {})
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_call_tool_error(self):
        reg = ToolRegistry()
        async def bad_tool():
            raise RuntimeError("broken")
        reg.register("bad_tool", bad_tool, {"description": "Broken"})
        result = await reg.call("bad_tool", {})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_get_schemas(self):
        reg = ToolRegistry()
        async def my_tool():
            return "ok"
        reg.register("my_tool", my_tool, {"description": "A test tool"})
        schemas = reg.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "my_tool"

    def test_scope_check_default_allows_all(self):
        reg = ToolRegistry()
        assert reg._check_scope("any_tool", {"url": "http://anything.com"}) is None

    def test_is_in_scope_exact_match(self):
        reg = ToolRegistry(allowed_targets=["example.com"])
        assert reg._is_in_scope("http://example.com/path") is True

    def test_is_in_scope_subdomain(self):
        reg = ToolRegistry(allowed_targets=["example.com"])
        assert reg._is_in_scope("http://sub.example.com") is True

    def test_is_in_scope_no_match(self):
        reg = ToolRegistry(allowed_targets=["example.com"])
        assert reg._is_in_scope("http://other.com") is False

    def test_is_in_scope_ip_range(self):
        reg = ToolRegistry(allowed_targets=["10.0.0.0/24"])
        assert reg._is_in_scope("10.0.0.5") is True
        assert reg._is_in_scope("10.0.1.5") is False

    def test_set_scope(self):
        reg = ToolRegistry()
        reg.set_scope(["new-target.com"])
        assert reg.allowed_targets == ["new-target.com"]

    def test_create_finding_schema_exists(self):
        assert "description" in CREATE_FINDING_SCHEMA
        assert "parameters" in CREATE_FINDING_SCHEMA
