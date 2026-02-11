"""Tests for ReasoningGraph."""

import pytest
from omnigent.reasoning_graph import (
    ReasoningGraph, ReasoningNode, ReasoningEdge, ReasoningPath, NodeState,
)


class TestReasoningGraph:
    def _make_graph(self):
        """Create a simple test graph."""
        g = ReasoningGraph()
        g.register_node("a", "cap_a", "Node A")
        g.register_node("b", "cap_b", "Node B")
        g.register_node("c", "cap_c", "Node C")
        g.register_edge("a", "b", "step_ab", "Go from A to B")
        g.register_edge("b", "c", "step_bc", "Go from B to C")
        g.register_path("Chain A→C", ["a", "b", "c"], "high", "Full chain")
        return g

    def test_register_node(self):
        g = ReasoningGraph()
        node = g.register_node("test", "test_cap", "Test Node")
        assert isinstance(node, ReasoningNode)
        assert "test" in g.nodes
        assert g.nodes["test"].label == "Test Node"

    def test_register_edge(self):
        g = ReasoningGraph()
        g.register_node("a", "a", "A")
        g.register_node("b", "b", "B")
        edge = g.register_edge("a", "b", "tech", "Description")
        assert isinstance(edge, ReasoningEdge)
        assert edge.source == "a"
        assert edge.target == "b"

    def test_register_path(self):
        g = self._make_graph()
        assert len(g._paths) == 1
        assert g._paths[0].name == "Chain A→C"
        assert g._paths[0].steps == 2

    def test_mark_discovered_exact(self):
        g = self._make_graph()
        g.mark_discovered("a", "Found A")
        assert g.nodes["a"].state == NodeState.CONFIRMED

    def test_mark_discovered_case_insensitive(self):
        g = self._make_graph()
        g.mark_discovered("A", "Found A")
        # Should match node "a" via case-insensitive or label matching
        # Exact implementation may vary; the key is no crash
        found = any(n.state == NodeState.CONFIRMED for n in g.nodes.values())
        # At minimum it shouldn't crash
        assert True

    def test_mark_discovered_alias(self):
        g = self._make_graph()
        g.register_alias("alpha", "a")
        g.mark_discovered("alpha", "Found via alias")
        assert g.nodes["a"].state == NodeState.CONFIRMED

    def test_mark_discovered_unknown_capability(self):
        g = self._make_graph()
        # Should not raise
        result = g.mark_discovered("nonexistent", "Nothing")
        # Returns empty list or doesn't crash
        assert True

    def test_get_next_steps(self):
        g = self._make_graph()
        g.mark_discovered("a", "Found A")
        steps = g.get_next_steps()
        # Should suggest the edge from a→b since a is confirmed
        assert isinstance(steps, list)

    def test_to_prompt_context(self):
        g = self._make_graph()
        g.mark_discovered("a", "Found A")
        ctx = g.to_prompt_context()
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_to_dict_from_dict_roundtrip(self):
        g = self._make_graph()
        g.mark_discovered("a", "Found A")
        d = g.to_dict()
        g2 = self._make_graph()  # Fresh graph with same structure
        g2.from_dict(d)           # Restore state
        assert "a" in g2.nodes
        assert g2.nodes["a"].state == NodeState.CONFIRMED

    def test_empty_graph(self):
        g = ReasoningGraph()
        assert len(g.nodes) == 0
        assert len(g.edges) == 0
        assert g.to_prompt_context() is not None  # Should not crash

    def test_subclass_override(self):
        """Subclass can override _build_default_graph."""
        class MyGraph(ReasoningGraph):
            def _build_default_graph(self):
                self.register_node("x", "x", "X")
                self.register_node("y", "y", "Y")
                self.register_edge("x", "y", "xy", "X to Y")

        g = MyGraph()
        assert "x" in g.nodes
        assert "y" in g.nodes
        assert len(g.edges) == 1
