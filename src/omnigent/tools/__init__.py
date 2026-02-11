"""
Omnigent — Tool Registry

Core tool registry with scope checking and few-shot examples.
Domain implementations register their own tools.

Architecture:
  ToolRegistry — holds tools, schemas, scope checking.
  Domain implementations do:
    registry = ToolRegistry()
    registry.register("my_tool", my_func, MY_SCHEMA)
"""

import asyncio
import json
import logging
from typing import Any

from omnigent.few_shot_examples import get_examples

logger = logging.getLogger("omnigent.tools")


# ═══════════════════════════════════════════════════════════════════════════
# Tool Registry
# ═══════════════════════════════════════════════════════════════════════════


class ToolRegistry:
    """Direct tool registry with scope checking.

    Usage:
        registry = ToolRegistry()
        registry.register("my_tool", my_async_func, {"description": "...", "input_schema": {...}})
        result = await registry.call("my_tool", {"arg1": "value"})
        schemas = registry.get_schemas()  # For LLM tool definitions

    Scope checking:
        registry.set_scope(["target1.com", "10.0.0.0/24"])
        # Tool calls that target out-of-scope hosts will be blocked.
        # Override _check_scope() and _is_in_scope() for domain-specific logic.
    """

    def __init__(self, allowed_targets: list[str] | None = None):
        self.tools: dict[str, dict] = {}
        self.allowed_targets = allowed_targets or []
        self._logger = logging.getLogger("omnigent.tools")

    def register(self, name: str, func: callable = None, schema: dict = None, *, handler: callable = None):
        """Register a tool.

        Accepts either positional: register("name", func, schema)
        or keyword: register(name="name", schema={...}, handler=func)
        """
        fn = func or handler
        if fn is None:
            raise ValueError("Must provide func or handler")
        self.tools[name] = {"func": fn, "schema": schema or {}}

    async def close(self):
        """Cleanup async resources. Override in domain implementation."""
        pass

    def set_scope(self, targets: list[str]):
        """Set allowed targets for scope checking."""
        self.allowed_targets = targets

    def _check_scope(self, name: str, args: dict) -> str | None:
        """Check if tool arguments are within allowed scope.

        Returns error message if out of scope, None if OK.
        Override in domain implementation for tool-specific scope rules.
        """
        if not self.allowed_targets:
            return None  # No scope = allow everything
        return None  # Base implementation allows everything; override in subclass

    def _is_in_scope(self, target_value: str) -> bool:
        """Check if a target value matches any allowed scope entry.

        Uses proper hostname extraction and CIDR matching.
        """
        from urllib.parse import urlparse
        import ipaddress

        def _extract_host(value: str) -> str:
            v = value.strip()
            if "://" not in v:
                v = "http://" + v
            parsed = urlparse(v)
            host = parsed.hostname or ""
            return host.lower().rstrip(".")

        target_host = _extract_host(target_value)
        if not target_host:
            return False

        for allowed in self.allowed_targets:
            allowed_raw = allowed.strip()

            # Try CIDR matching first (before hostname extraction mangles the /prefix)
            try:
                target_ip = ipaddress.ip_address(target_host)
                try:
                    scope_net = ipaddress.ip_network(allowed_raw, strict=False)
                    if target_ip in scope_net:
                        return True
                except ValueError:
                    pass
            except ValueError:
                pass

            scope_host = _extract_host(allowed_raw)
            if not scope_host:
                continue

            # Exact match
            if target_host == scope_host:
                return True

            # Subdomain match
            if target_host.endswith("." + scope_host):
                return True

            # IP match (non-CIDR)
            try:
                target_ip = ipaddress.ip_address(target_host)
                scope_ip = ipaddress.ip_address(scope_host)
                if target_ip == scope_ip:
                    return True
            except ValueError:
                pass

        return False

    async def call(self, name: str, args: dict) -> str:
        """Call a tool directly."""
        tool = self.tools.get(name)
        if not tool:
            return json.dumps({"error": f"Unknown tool: {name}"})

        # Scope check
        scope_error = self._check_scope(name, args)
        if scope_error:
            self._logger.warning(f"Scope violation: {name} → {args}")
            return json.dumps({"error": scope_error})

        try:
            func = tool["func"]
            self._logger.debug(f"Tool call: {name} with args: {args}")

            if asyncio.iscoroutinefunction(func):
                result = await func(**args)
            else:
                result = func(**args)

            if isinstance(result, str):
                return result
            return json.dumps(result, indent=2)
        except Exception as e:
            self._logger.error(f"Tool execution error for {name}: {e}", exc_info=True)
            return json.dumps({"error": str(e)})

    def get_schemas(self) -> list[dict]:
        """Get all tool schemas for LLM with few-shot examples."""
        schemas = []
        for name, tool in self.tools.items():
            schema = {
                "type": "function",
                "function": {
                    "name": name,
                    **tool["schema"],
                },
            }

            # Add few-shot examples
            examples = get_examples(name)
            if examples:
                desc = schema["function"].get("description", "")
                desc += "\n\n**Examples:**"
                for i, ex in enumerate(examples, 1):
                    outcome_emoji = "+" if ex.is_good else "-"
                    desc += f"\n{outcome_emoji} Example {i}: {ex.scenario}"
                    if ex.thinking:
                        desc += f"\n   Thinking: {ex.thinking[:100]}..."
                    desc += f"\n   Expected: {ex.expected_result[:80]}..."
                schema["function"]["description"] = desc

            schemas.append(schema)

        return schemas

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self.tools.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Built-in: create_finding tool (always available)
# ═══════════════════════════════════════════════════════════════════════════

CREATE_FINDING_SCHEMA = {
    "description": (
        "Register a finding/issue discovered during analysis. "
        "Use this to formally report discovered issues with severity and evidence."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short title of the finding"},
            "severity": {
                "type": "string",
                "enum": ["critical", "high", "medium", "low", "info"],
                "description": "Severity level",
            },
            "description": {"type": "string", "description": "Detailed description"},
            "evidence": {"type": "string", "description": "Proof/evidence of the finding"},
        },
        "required": ["title", "severity", "description"],
    },
}
