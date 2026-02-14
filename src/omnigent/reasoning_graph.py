"""
Omnigent — Reasoning Graph (Multi-Step Chain Reasoning)

Directed graph of capability/reasoning chains. When a finding is registered,
the graph activates downstream escalation paths. The agent reads
the graph to decide what to pursue next.

THIS IS THE DIFFERENTIATOR — it separates an autonomous agent from a
simple tool-caller. Instead of finding individual issues, the agent
chains them into multi-step reasoning paths.

Architecture:
  Nodes = capabilities/states (e.g. "data_access", "root_cause_found", etc.)
  Edges = reasoning/exploitation steps connecting capabilities
  When a capability is confirmed, downstream edges become available.

Domain examples:
  Security: SQLi → DB Dump → Credential Extraction → Admin Access → RCE
  Code Quality: God Object → High Coupling → Low Testability → Regression Risk
  Incident Response: Alert → Log Correlation → Root Cause → Blast Radius
  Compliance: Gap Found → Control Missing → Risk Assessment → Remediation Plan

The graph is populated at construction time. Override _build_default_graph()
in your domain subclass or use register_node()/register_edge() dynamically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("omnigent.reasoning_graph")


# ═══════════════════════════════════════════════════════════════════════════
# Node & Edge Types
# ═══════════════════════════════════════════════════════════════════════════

class NodeState(str, Enum):
    """State of a reasoning graph node."""
    UNKNOWN = "unknown"         # Not yet investigated
    SUSPECTED = "suspected"     # Hypothesis, not confirmed
    CONFIRMED = "confirmed"     # Confirmed via investigation
    EXPLOITED = "exploited"     # Successfully acted upon
    FAILED = "failed"           # Investigated and not present


@dataclass
class ReasoningNode:
    """A capability or state in the reasoning graph."""
    id: str                                       # Unique identifier
    capability: str                                # Capability type
    label: str                                     # Human-readable label
    state: NodeState = NodeState.UNKNOWN
    finding_title: str = ""                        # Link to Finding
    location: str = ""                             # Where found
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, ReasoningNode):
            return self.id == other.id
        return False


@dataclass
class ReasoningEdge:
    """A reasoning step connecting two capabilities."""
    source: str                     # Source node ID
    target: str                     # Target node ID
    technique: str                  # Technique/method name
    description: str                # Human-readable step description
    tool_hint: str = ""             # Suggested tool to use
    knowledge_ref: str = ""         # Knowledge base reference
    priority: int = 5               # 1 = highest priority
    requires_all: list[str] = field(default_factory=list)

    def __hash__(self):
        return hash((self.source, self.target, self.technique))


@dataclass
class ReasoningPath:
    """A sequence of edges forming a complete reasoning chain."""
    name: str
    edges: list[ReasoningEdge]
    impact: str                     # "critical", "high", "medium"
    description: str

    @property
    def steps(self) -> int:
        return len(self.edges)


# ═══════════════════════════════════════════════════════════════════════════
# Reasoning Graph
# ═══════════════════════════════════════════════════════════════════════════

class ReasoningGraph:
    """Directed graph of reasoning/escalation paths.

    Core API:
      - register_node(id, capability, label) — add a node
      - register_edge(source, target, ...) — add an edge
      - register_path(name, node_ids, impact, desc) — define a named chain
      - mark_discovered(capability, finding) — register a confirmed capability
      - get_next_steps() — return highest-impact unexplored paths
      - to_prompt_context() — generate prompt text for LLM context injection

    Override _build_default_graph() in a subclass to pre-populate
    with domain-specific nodes, edges, and chains.
    """

    def __init__(self):
        self.nodes: dict[str, ReasoningNode] = {}
        self.edges: list[ReasoningEdge] = []
        self._paths: list[ReasoningPath] = []
        self._aliases: dict[str, str] = {}  # alias → node_id

        # Populate with domain-specific chains (override in subclass)
        self._build_default_graph()

    def _build_default_graph(self) -> None:
        """Override in domain subclass to pre-populate nodes, edges, paths.

        Example for security domain:
            self.register_node("sqli", "sqli", "SQL Injection")
            self.register_node("db_access", "db_access", "Database Access")
            self.register_edge("sqli", "db_access", "db_dump", "Dump database via SQLi")
            self.register_path("SQLi → DB", ["sqli", "db_access"], "critical", "Chain SQLi to DB access")
        """
        pass  # Empty by default — domain subclass populates

    # ═══════════════════════════════════════════════════════════════════
    # Registration API
    # ═══════════════════════════════════════════════════════════════════

    def register_node(self, node_id: str, capability: str, label: str, **metadata) -> ReasoningNode:
        """Register a new node in the graph."""
        node = ReasoningNode(id=node_id, capability=capability, label=label, metadata=metadata)
        self.nodes[node_id] = node
        return node

    def register_edge(
        self,
        source: str,
        target: str,
        technique: str,
        description: str,
        tool_hint: str = "",
        knowledge_ref: str = "",
        priority: int = 5,
    ) -> ReasoningEdge:
        """Register a new edge in the graph."""
        edge = ReasoningEdge(
            source=source, target=target, technique=technique,
            description=description, tool_hint=tool_hint,
            knowledge_ref=knowledge_ref, priority=priority,
        )
        self.edges.append(edge)
        return edge

    def register_path(
        self,
        name: str,
        node_ids: list[str],
        impact: str,
        description: str,
    ) -> ReasoningPath | None:
        """Register a named reasoning path from a sequence of node IDs."""
        edges = []
        for i in range(len(node_ids) - 1):
            src, tgt = node_ids[i], node_ids[i + 1]
            for edge in self.edges:
                if edge.source == src and edge.target == tgt:
                    edges.append(edge)
                    break
        if edges:
            path = ReasoningPath(name=name, edges=edges, impact=impact, description=description)
            self._paths.append(path)
            return path
        return None

    def register_alias(self, alias: str, node_id: str):
        """Register an alias for fuzzy matching (e.g., 'sql injection' → 'sqli')."""
        self._aliases[alias.lower()] = node_id

    def register_aliases(self, aliases: dict[str, str]):
        """Register multiple aliases at once."""
        for alias, node_id in aliases.items():
            self._aliases[alias.lower()] = node_id

    # ═══════════════════════════════════════════════════════════════════
    # Core API
    # ═══════════════════════════════════════════════════════════════════

    def mark_discovered(
        self,
        capability: str,
        finding_title: str = "",
        location: str = "",
        state: NodeState = NodeState.CONFIRMED,
    ) -> list[ReasoningPath]:
        """Register a confirmed capability, activate downstream paths.

        Uses 5-tier fuzzy matching:
        1. Exact node ID
        2. Case-insensitive ID
        3. Alias lookup (registered via register_alias)
        4. Label exact match
        5. Conservative prefix match (>4 chars)

        Returns:
            List of newly available reasoning paths
        """
        node = self.nodes.get(capability)
        if not node:
            cap_lower = capability.lower()

            # 1. Exact case-insensitive
            if cap_lower in self.nodes:
                node = self.nodes[cap_lower]

            # 2. Alias lookup
            if not node:
                for alias, nid in self._aliases.items():
                    if alias in cap_lower and nid in self.nodes:
                        node = self.nodes[nid]
                        break

            # 3. Label exact match
            if not node:
                for nid, n in self.nodes.items():
                    if n.label.lower() == cap_lower:
                        node = n
                        break

            # 4. Conservative prefix (>4 chars only)
            if not node and len(cap_lower) > 4:
                for nid, n in self.nodes.items():
                    if cap_lower == nid.lower() or nid.lower().startswith(cap_lower):
                        node = n
                        break

        if not node:
            logger.debug(f"Unknown capability: {capability}")
            return []

        node.state = state
        node.finding_title = finding_title
        node.location = location

        logger.info(f"Reasoning graph: {capability} marked as {state.value}")
        return self.get_available_paths()

    def get_available_paths(self) -> list[ReasoningPath]:
        """Return paths where first node is confirmed but later nodes are unexplored."""
        available = []
        for path in self._paths:
            if not path.edges:
                continue
            first_source = path.edges[0].source
            first_node = self.nodes.get(first_source)
            if not first_node or first_node.state not in (NodeState.CONFIRMED, NodeState.EXPLOITED):
                continue
            has_unexplored = False
            for edge in path.edges:
                target_node = self.nodes.get(edge.target)
                if target_node and target_node.state in (NodeState.UNKNOWN, NodeState.SUSPECTED):
                    has_unexplored = True
                    break
            if has_unexplored:
                available.append(path)

        impact_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        available.sort(key=lambda p: (impact_order.get(p.impact, 99), p.steps))
        return available

    def get_next_steps(self, limit: int = 5) -> list[ReasoningEdge]:
        """Return highest-priority next steps (confirmed source → unknown target).

        Respects requires_all: an edge is only available if ALL nodes in its
        requires_all list are confirmed/exploited.
        """
        next_edges: list[ReasoningEdge] = []
        for edge in self.edges:
            source_node = self.nodes.get(edge.source)
            target_node = self.nodes.get(edge.target)
            if not source_node or not target_node:
                continue
            if source_node.state not in (NodeState.CONFIRMED, NodeState.EXPLOITED):
                continue
            if target_node.state not in (NodeState.UNKNOWN, NodeState.SUSPECTED):
                continue
            # Validate requires_all prerequisites
            if edge.requires_all:
                all_met = all(
                    self.nodes.get(req) is not None
                    and self.nodes[req].state in (NodeState.CONFIRMED, NodeState.EXPLOITED)
                    for req in edge.requires_all
                )
                if not all_met:
                    continue
            next_edges.append(edge)
        next_edges.sort(key=lambda e: e.priority)
        return next_edges[:limit]

    def get_confirmed_nodes(self) -> list[ReasoningNode]:
        """Return all confirmed/exploited nodes."""
        return [
            n for n in self.nodes.values()
            if n.state in (NodeState.CONFIRMED, NodeState.EXPLOITED)
        ]

    def to_prompt_context(self) -> str:
        """Generate prompt context for LLM injection.

        Returns structured summary of confirmed capabilities,
        available escalation paths, and suggested next steps.
        """
        confirmed = self.get_confirmed_nodes()
        if not confirmed:
            return ""

        lines = [
            "## Reasoning Graph — Confirmed Capabilities & Escalation Paths",
            "",
            "### Confirmed",
        ]

        for node in confirmed:
            loc = f" at {node.location}" if node.location else ""
            lines.append(f"- ✅ **{node.label}**{loc}")
            if node.finding_title:
                lines.append(f"  (Finding: {node.finding_title})")

        paths = self.get_available_paths()
        if paths:
            lines.append("")
            lines.append("### Available Escalation Paths")
            lines.append("")
            for path in paths[:5]:
                lines.append(f"**{path.name}** ({path.impact} impact, {path.steps} steps)")
                for i, edge in enumerate(path.edges, 1):
                    source = self.nodes.get(edge.source)
                    state_icon = "✅" if source and source.state in (NodeState.CONFIRMED, NodeState.EXPLOITED) else "⬜"
                    lines.append(f"  {i}. {state_icon} {edge.description}")
                    if edge.tool_hint:
                        lines.append(f"     → Tool: `{edge.tool_hint}`")
                    if edge.knowledge_ref:
                        lines.append(f"     → Knowledge: `{edge.knowledge_ref}`")
                lines.append("")

        next_steps = self.get_next_steps(limit=3)
        if next_steps:
            lines.append("### Suggested Next Actions")
            lines.append("")
            for step in next_steps:
                lines.append(f"1. **{step.description}**")
                if step.tool_hint:
                    lines.append(f"   Use: `{step.tool_hint}`")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph state for session persistence."""
        return {
            "nodes": {
                nid: {
                    "state": node.state.value,
                    "finding_title": node.finding_title,
                    "location": node.location,
                }
                for nid, node in self.nodes.items()
                if node.state != NodeState.UNKNOWN
            },
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore graph state from serialized data."""
        for nid, node_data in data.get("nodes", {}).items():
            if nid in self.nodes:
                self.nodes[nid].state = NodeState(node_data.get("state", "unknown"))
                self.nodes[nid].finding_title = node_data.get("finding_title", "")
                self.nodes[nid].location = node_data.get("location", "")
