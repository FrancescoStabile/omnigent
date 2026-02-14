"""
Omnigent — Domain Registry

Centralises all domain-specific registries into a single injectable dataclass.
Eliminates global singleton state so multiple Agent instances (or test cases)
can each carry their own registries without leaking into one another.

Usage:
    # Default (uses module-level dicts as fallback)
    agent = Agent()  # uses DomainRegistry.default()

    # Per-instance
    reg = DomainRegistry(
        extractors={"nmap": my_extractor},
        plan_templates={"web": my_template},
    )
    agent = Agent(registry=reg)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DomainRegistry:
    """Holds all domain-specific registries for an Agent instance.

    Each field mirrors a former module-level dict. Passing this to
    Agent avoids global state and enables multi-agent scenarios.
    """

    # Extractors: tool_name → callable(profile, result, args)
    extractors: dict[str, Any] = field(default_factory=dict)

    # Reflectors: tool_name → callable(result, args, profile, lines)
    reflectors: dict[str, Callable] = field(default_factory=dict)

    # Escalation chains: category → list[ChainStep]
    chains: dict[str, list] = field(default_factory=dict)

    # Plan templates: template_key → list[phase_defs]
    plan_templates: dict[str, list[dict]] = field(default_factory=dict)

    # Error recovery patterns: tool_name → {pattern_name: {indicators, strategy}}
    error_patterns: dict[str, dict] = field(default_factory=dict)

    # Few-shot examples: tool_name → list[ToolExample]
    examples: dict[str, list] = field(default_factory=dict)

    # Knowledge map: key → list[Entry]
    knowledge_map: dict[str, list] = field(default_factory=dict)

    # Tool-specific timeouts: tool_name → seconds
    tool_timeouts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def default(cls) -> DomainRegistry:
        """Create a registry pre-populated from module-level dicts.

        This is the backward-compatible path: if no registry is passed to
        Agent, it reads the module-level dicts that domain implementations
        already populate at import time.
        """
        from omnigent.agent import TOOL_TIMEOUTS
        from omnigent.chains import CHAINS
        from omnigent.error_recovery import ERROR_PATTERNS
        from omnigent.extractors import EXTRACTORS
        from omnigent.few_shot_examples import EXAMPLES
        from omnigent.knowledge_loader import KNOWLEDGE_MAP
        from omnigent.planner import PLAN_TEMPLATES
        from omnigent.reflection import REFLECTORS

        return cls(
            extractors=EXTRACTORS,
            reflectors=REFLECTORS,
            chains=CHAINS,
            plan_templates=PLAN_TEMPLATES,
            error_patterns=ERROR_PATTERNS,
            examples=EXAMPLES,
            knowledge_map=KNOWLEDGE_MAP,
            tool_timeouts=TOOL_TIMEOUTS,
        )

    def merge(self, other: DomainRegistry) -> None:
        """Merge another registry into this one (other takes precedence)."""
        self.extractors.update(other.extractors)
        self.reflectors.update(other.reflectors)
        self.chains.update(other.chains)
        self.plan_templates.update(other.plan_templates)
        self.error_patterns.update(other.error_patterns)
        self.examples.update(other.examples)
        self.knowledge_map.update(other.knowledge_map)
        self.tool_timeouts.update(other.tool_timeouts)
