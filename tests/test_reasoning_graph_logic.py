"""Comprehensive tests for reasoning graph logic in omnigent.reasoning_graph.

Covers:
  - Chain activation (A->B->C, confirm A, check next_steps)
  - Path availability after partial confirmation
  - Multi-path priority sorting by impact and step count
  - requires_all prerequisite gating on edges
  - Fuzzy matching (alias, label, prefix)
  - Prompt context generation
  - Unconfirmed nodes do not generate paths
"""

import pytest

from omnigent.reasoning_graph import (
    NodeState,
    ReasoningEdge,
    ReasoningGraph,
    ReasoningNode,
    ReasoningPath,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_abc_graph() -> ReasoningGraph:
    """Build a simple three-node chain: A -> B -> C."""
    g = ReasoningGraph()
    g.register_node("a", "cap_a", "Node A")
    g.register_node("b", "cap_b", "Node B")
    g.register_node("c", "cap_c", "Node C")
    g.register_edge("a", "b", "step_ab", "Go from A to B", tool_hint="tool_ab")
    g.register_edge("b", "c", "step_bc", "Go from B to C", tool_hint="tool_bc")
    g.register_path("Chain A->C", ["a", "b", "c"], "high", "Full chain A to C")
    return g


# ═══════════════════════════════════════════════════════════════════════════════
# TestChainActivation
# ═══════════════════════════════════════════════════════════════════════════════

class TestChainActivation:
    """Register A->B->C, confirm A, verify B's edge appears in next_steps."""

    def test_confirming_a_yields_ab_edge_in_next_steps(self):
        g = _build_abc_graph()
        g.mark_discovered("a", finding_title="Found A", location="/endpoint")

        steps = g.get_next_steps()
        assert len(steps) >= 1
        sources_and_targets = [(e.source, e.target) for e in steps]
        assert ("a", "b") in sources_and_targets

    def test_bc_edge_not_in_next_steps_before_b_confirmed(self):
        """Edge B->C should NOT appear until B itself is confirmed."""
        g = _build_abc_graph()
        g.mark_discovered("a", finding_title="Found A")

        steps = g.get_next_steps()
        sources_and_targets = [(e.source, e.target) for e in steps]
        assert ("b", "c") not in sources_and_targets

    def test_confirming_b_yields_bc_edge(self):
        """After confirming both A and B, the B->C edge should appear."""
        g = _build_abc_graph()
        g.mark_discovered("a", finding_title="Found A")
        g.mark_discovered("b", finding_title="Found B")

        steps = g.get_next_steps()
        sources_and_targets = [(e.source, e.target) for e in steps]
        assert ("b", "c") in sources_and_targets

    def test_no_next_steps_when_nothing_confirmed(self):
        g = _build_abc_graph()
        steps = g.get_next_steps()
        assert steps == []

    def test_confirmed_source_with_already_confirmed_target_not_in_steps(self):
        """If both source and target are confirmed, the edge is not a 'next step'."""
        g = _build_abc_graph()
        g.mark_discovered("a", finding_title="Found A")
        g.mark_discovered("b", finding_title="Found B")

        steps = g.get_next_steps()
        # a->b should NOT appear since b is already confirmed
        sources_and_targets = [(e.source, e.target) for e in steps]
        assert ("a", "b") not in sources_and_targets


# ═══════════════════════════════════════════════════════════════════════════════
# TestPathAvailability
# ═══════════════════════════════════════════════════════════════════════════════

class TestPathAvailability:
    """Register path [A,B,C], confirm A, verify path is available."""

    def test_path_available_after_first_node_confirmed(self):
        g = _build_abc_graph()
        paths = g.mark_discovered("a", finding_title="Found A")

        assert len(paths) >= 1
        path_names = [p.name for p in paths]
        assert "Chain A->C" in path_names

    def test_path_not_available_when_nothing_confirmed(self):
        g = _build_abc_graph()
        paths = g.get_available_paths()
        assert len(paths) == 0

    def test_path_not_available_when_all_nodes_confirmed(self):
        """If all nodes in a path are confirmed, the path has no unexplored targets."""
        g = _build_abc_graph()
        g.mark_discovered("a", finding_title="Found A")
        g.mark_discovered("b", finding_title="Found B")
        g.mark_discovered("c", finding_title="Found C")

        paths = g.get_available_paths()
        # All targets are confirmed, so no unexplored edge targets remain
        assert len(paths) == 0

    def test_path_available_with_partial_confirmation(self):
        """Path still available when first source confirmed and later nodes unknown."""
        g = _build_abc_graph()
        g.mark_discovered("a", finding_title="Found A")
        g.mark_discovered("b", finding_title="Found B")

        # C is still unknown, so the chain should still be available
        paths = g.get_available_paths()
        path_names = [p.name for p in paths]
        assert "Chain A->C" in path_names


# ═══════════════════════════════════════════════════════════════════════════════
# TestMultiPathPriority
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiPathPriority:
    """Multiple paths with different impact, verify sorting."""

    def test_critical_path_comes_before_high(self):
        g = ReasoningGraph()
        g.register_node("x", "cap_x", "Node X")
        g.register_node("y", "cap_y", "Node Y")
        g.register_node("z", "cap_z", "Node Z")
        g.register_node("w", "cap_w", "Node W")

        g.register_edge("x", "y", "xy", "X to Y")
        g.register_edge("x", "z", "xz", "X to Z")
        g.register_edge("x", "w", "xw", "X to W")

        g.register_path("High path", ["x", "y"], "high", "High impact chain")
        g.register_path("Critical path", ["x", "z"], "critical", "Critical impact chain")
        g.register_path("Medium path", ["x", "w"], "medium", "Medium impact chain")

        g.mark_discovered("x", finding_title="Found X")
        paths = g.get_available_paths()

        assert len(paths) == 3
        assert paths[0].impact == "critical"
        assert paths[1].impact == "high"
        assert paths[2].impact == "medium"

    def test_same_impact_sorted_by_step_count(self):
        """When impact is the same, shorter paths come first."""
        g = ReasoningGraph()
        g.register_node("a", "cap_a", "A")
        g.register_node("b", "cap_b", "B")
        g.register_node("c", "cap_c", "C")
        g.register_node("d", "cap_d", "D")

        g.register_edge("a", "b", "ab", "A to B")
        g.register_edge("a", "c", "ac", "A to C")
        g.register_edge("c", "d", "cd", "C to D")

        # Short path: A -> B (1 step)
        g.register_path("Short", ["a", "b"], "high", "Short chain")
        # Long path: A -> C -> D (2 steps)
        g.register_path("Long", ["a", "c", "d"], "high", "Long chain")

        g.mark_discovered("a", finding_title="Found A")
        paths = g.get_available_paths()

        assert len(paths) == 2
        assert paths[0].name == "Short"
        assert paths[0].steps == 1
        assert paths[1].name == "Long"
        assert paths[1].steps == 2


# ═══════════════════════════════════════════════════════════════════════════════
# TestRequiresAll
# ═══════════════════════════════════════════════════════════════════════════════

class TestRequiresAll:
    """Edge with requires_all=[A,B]: only available when ALL prerequisites met."""

    def _build_requires_all_graph(self) -> ReasoningGraph:
        """Graph where edge X->Z requires both A and B to be confirmed."""
        g = ReasoningGraph()
        g.register_node("a", "cap_a", "Prereq A")
        g.register_node("b", "cap_b", "Prereq B")
        g.register_node("x", "cap_x", "Source X")
        g.register_node("z", "cap_z", "Target Z")

        # Normal edge from x to z, but with requires_all
        edge = g.register_edge("x", "z", "xz", "X to Z (needs A and B)")
        # register_edge does not expose requires_all, so set it directly
        edge.requires_all = ["a", "b"]

        return g

    def test_edge_blocked_when_only_one_prerequisite_met(self):
        g = self._build_requires_all_graph()
        g.mark_discovered("x", finding_title="Found X")
        g.mark_discovered("a", finding_title="Found A")
        # B is NOT confirmed

        steps = g.get_next_steps()
        targets = [e.target for e in steps]
        assert "z" not in targets

    def test_edge_available_when_all_prerequisites_met(self):
        g = self._build_requires_all_graph()
        g.mark_discovered("x", finding_title="Found X")
        g.mark_discovered("a", finding_title="Found A")
        g.mark_discovered("b", finding_title="Found B")

        steps = g.get_next_steps()
        targets = [e.target for e in steps]
        assert "z" in targets

    def test_edge_blocked_when_no_prerequisites_met(self):
        g = self._build_requires_all_graph()
        g.mark_discovered("x", finding_title="Found X")
        # Neither A nor B confirmed

        steps = g.get_next_steps()
        targets = [e.target for e in steps]
        assert "z" not in targets

    def test_edge_with_empty_requires_all_is_unconditional(self):
        """An edge with no requires_all should work normally."""
        g = _build_abc_graph()
        g.mark_discovered("a", finding_title="Found A")

        steps = g.get_next_steps()
        assert len(steps) >= 1
        assert steps[0].source == "a"
        assert steps[0].target == "b"

    def test_requires_all_with_exploited_state(self):
        """EXPLOITED state should also satisfy requires_all."""
        g = self._build_requires_all_graph()
        g.mark_discovered("x", finding_title="Found X")
        g.mark_discovered("a", finding_title="Found A", state=NodeState.EXPLOITED)
        g.mark_discovered("b", finding_title="Found B", state=NodeState.CONFIRMED)

        steps = g.get_next_steps()
        targets = [e.target for e in steps]
        assert "z" in targets


# ═══════════════════════════════════════════════════════════════════════════════
# TestFuzzyMatching
# ═══════════════════════════════════════════════════════════════════════════════

class TestFuzzyMatching:
    """Alias lookup, label match, prefix match all work for mark_discovered."""

    def test_alias_match(self):
        g = ReasoningGraph()
        g.register_node("sqli", "sqli", "SQL Injection")
        g.register_alias("sql injection", "sqli")

        g.mark_discovered("sql injection", finding_title="SQLi found")
        assert g.nodes["sqli"].state == NodeState.CONFIRMED

    def test_label_exact_match(self):
        """Matching by exact label (case-insensitive)."""
        g = ReasoningGraph()
        g.register_node("rce", "rce", "Remote Code Execution")

        g.mark_discovered("remote code execution", finding_title="RCE found")
        assert g.nodes["rce"].state == NodeState.CONFIRMED

    def test_prefix_match(self):
        """Matching by prefix (only when capability >4 chars)."""
        g = ReasoningGraph()
        g.register_node("buffer_overflow", "buffer_overflow", "Buffer Overflow")

        # "buffer_overflow" starts with "buffer_over"
        g.mark_discovered("buffer_over", finding_title="Partial match")
        assert g.nodes["buffer_overflow"].state == NodeState.CONFIRMED

    def test_short_prefix_does_not_match(self):
        """Prefixes <= 4 chars should NOT trigger fuzzy matching."""
        g = ReasoningGraph()
        g.register_node("xss", "xss", "Cross-Site Scripting")

        result = g.mark_discovered("xs", finding_title="Too short")
        # "xs" is only 2 chars, should not match "xss"
        assert g.nodes["xss"].state == NodeState.UNKNOWN
        assert result == []

    def test_case_insensitive_exact_id_match(self):
        g = ReasoningGraph()
        g.register_node("sqli", "sqli", "SQL Injection")

        g.mark_discovered("SQLI", finding_title="Case test")
        # Case-insensitive ID match: "sqli" matches "SQLI".lower()
        assert g.nodes["sqli"].state == NodeState.CONFIRMED

    def test_unknown_capability_returns_empty_list(self):
        g = ReasoningGraph()
        g.register_node("a", "cap_a", "Node A")

        result = g.mark_discovered("completely_nonexistent", finding_title="Nothing")
        assert result == []
        assert g.nodes["a"].state == NodeState.UNKNOWN

    def test_alias_partial_match_in_capability(self):
        """Alias matching uses 'alias in cap_lower', so partial inclusion works."""
        g = ReasoningGraph()
        g.register_node("sqli", "sqli", "SQL Injection")
        g.register_alias("sql", "sqli")

        # "found sql injection" contains "sql"
        g.mark_discovered("found sql injection", finding_title="Alias partial")
        assert g.nodes["sqli"].state == NodeState.CONFIRMED


# ═══════════════════════════════════════════════════════════════════════════════
# TestPromptContext
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromptContext:
    """Verify to_prompt_context() generates proper text with confirmed nodes."""

    def test_empty_graph_returns_empty_string(self):
        g = ReasoningGraph()
        assert g.to_prompt_context() == ""

    def test_no_confirmed_nodes_returns_empty_string(self):
        g = _build_abc_graph()
        assert g.to_prompt_context() == ""

    def test_confirmed_node_appears_in_context(self):
        g = _build_abc_graph()
        g.mark_discovered("a", finding_title="Found A", location="/api/v1")

        ctx = g.to_prompt_context()
        assert "Confirmed" in ctx
        assert "Node A" in ctx
        assert "/api/v1" in ctx
        assert "Found A" in ctx

    def test_available_paths_appear_in_context(self):
        g = _build_abc_graph()
        g.mark_discovered("a", finding_title="Found A")

        ctx = g.to_prompt_context()
        assert "Escalation Paths" in ctx
        assert "Chain A->C" in ctx
        assert "high" in ctx.lower()

    def test_next_steps_appear_in_context(self):
        g = _build_abc_graph()
        g.mark_discovered("a", finding_title="Found A")

        ctx = g.to_prompt_context()
        assert "Suggested Next Actions" in ctx
        assert "Go from A to B" in ctx

    def test_tool_hints_in_context(self):
        g = _build_abc_graph()
        g.mark_discovered("a", finding_title="Found A")

        ctx = g.to_prompt_context()
        assert "tool_ab" in ctx

    def test_multiple_confirmed_nodes_all_listed(self):
        g = _build_abc_graph()
        g.mark_discovered("a", finding_title="Found A")
        g.mark_discovered("b", finding_title="Found B")

        ctx = g.to_prompt_context()
        assert "Node A" in ctx
        assert "Node B" in ctx


# ═══════════════════════════════════════════════════════════════════════════════
# TestNoPathForUnconfirmed
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoPathForUnconfirmed:
    """Unconfirmed nodes do not generate paths or next steps."""

    def test_suspected_source_does_not_activate_path(self):
        """SUSPECTED state should not activate downstream paths."""
        g = _build_abc_graph()
        g.mark_discovered("a", finding_title="Maybe A", state=NodeState.SUSPECTED)

        paths = g.get_available_paths()
        assert len(paths) == 0

    def test_failed_source_does_not_activate_path(self):
        g = _build_abc_graph()
        g.mark_discovered("a", finding_title="Failed A", state=NodeState.FAILED)

        paths = g.get_available_paths()
        assert len(paths) == 0

    def test_unknown_source_has_no_next_steps(self):
        g = _build_abc_graph()
        # All nodes are UNKNOWN by default
        steps = g.get_next_steps()
        assert steps == []

    def test_suspected_source_has_no_next_steps(self):
        g = _build_abc_graph()
        g.mark_discovered("a", finding_title="Suspected A", state=NodeState.SUSPECTED)

        steps = g.get_next_steps()
        assert steps == []

    def test_exploited_source_activates_path(self):
        """EXPLOITED state SHOULD activate paths, since it's 'more than confirmed'."""
        g = _build_abc_graph()
        g.mark_discovered("a", finding_title="Exploited A", state=NodeState.EXPLOITED)

        paths = g.get_available_paths()
        assert len(paths) >= 1
        path_names = [p.name for p in paths]
        assert "Chain A->C" in path_names

    def test_exploited_source_has_next_steps(self):
        g = _build_abc_graph()
        g.mark_discovered("a", finding_title="Exploited A", state=NodeState.EXPLOITED)

        steps = g.get_next_steps()
        assert len(steps) >= 1
        assert steps[0].source == "a"
        assert steps[0].target == "b"
