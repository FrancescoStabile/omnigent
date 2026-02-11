"""
CodeLens reasoning graph — chains of analysis that build on each other.

Shows how to subclass ReasoningGraph to model domain-specific
reasoning chains (e.g. complexity → coupling → testability).
"""

from __future__ import annotations

from omnigent.reasoning_graph import ReasoningGraph


class CodeQualityGraph(ReasoningGraph):
    """Graph modelling code quality analysis chains."""

    def _build_default_graph(self) -> None:
        # ── Nodes ────────────────────────────────────────────────
        #  (node_id, category, label)

        # Discovery layer
        self.register_node("project_structure", "discovery", "Project Structure")
        self.register_node("dependency_map", "discovery", "Dependency Map")
        self.register_node("framework_detection", "discovery", "Framework Detection")

        # Metric layer
        self.register_node("high_complexity", "metrics", "High Complexity")
        self.register_node("code_duplication", "metrics", "Code Duplication")
        self.register_node("large_files", "metrics", "Large Files")

        # Pattern layer
        self.register_node("god_class", "patterns", "God Class")
        self.register_node("tight_coupling", "patterns", "Tight Coupling")
        self.register_node("circular_dependency", "patterns", "Circular Dependency")
        self.register_node("dead_code", "patterns", "Dead Code")

        # Impact layer
        self.register_node("low_testability", "impact", "Low Testability")
        self.register_node("maintenance_risk", "impact", "Maintenance Risk")
        self.register_node("security_exposure", "impact", "Security Exposure")

        # ── Edges (transitions between nodes) ────────────────────
        # Discovery → Metrics
        self.register_edge(
            "project_structure", "high_complexity", "complexity_scan",
            "Scan discovered files for cyclomatic complexity",
        )
        self.register_edge(
            "project_structure", "large_files", "size_scan",
            "Identify oversized files from structure scan",
        )
        self.register_edge(
            "dependency_map", "circular_dependency", "cycle_detection",
            "Check import graph for circular dependencies",
        )
        self.register_edge(
            "dependency_map", "tight_coupling", "coupling_analysis",
            "Measure coupling from dependency graph",
        )

        # Metrics → Patterns
        self.register_edge(
            "high_complexity", "god_class", "class_analysis",
            "Analyze complex classes for god-class patterns",
        )
        self.register_edge(
            "large_files", "code_duplication", "duplication_scan",
            "Check large files for copied code blocks",
        )
        self.register_edge(
            "high_complexity", "dead_code", "dead_code_scan",
            "Check complex modules for unreachable code",
        )

        # Patterns → Impact
        self.register_edge(
            "god_class", "low_testability", "testability_check",
            "Assess test difficulty for god classes",
        )
        self.register_edge(
            "tight_coupling", "maintenance_risk", "maintainability_check",
            "Rate maintenance risk from coupling",
        )
        self.register_edge(
            "circular_dependency", "maintenance_risk", "dep_risk",
            "Rate risk from circular dependencies",
        )
        self.register_edge(
            "code_duplication", "maintenance_risk", "dup_risk",
            "Rate maintenance risk from duplication",
        )
        self.register_edge(
            "dead_code", "security_exposure", "dead_code_risk",
            "Check if dead code has security implications",
        )

        # ── Paths (named multi-step chains) ──────────────────────
        self.register_path(
            "Complexity Impact Chain",
            ["project_structure", "high_complexity", "god_class", "low_testability"],
            "high",
            "Trace how structural complexity leads to untestable code",
        )
        self.register_path(
            "Dependency Risk Chain",
            ["dependency_map", "circular_dependency", "maintenance_risk"],
            "high",
            "Trace dependency problems to maintenance risk",
        )
        self.register_path(
            "Coupling Impact Chain",
            ["dependency_map", "tight_coupling", "maintenance_risk"],
            "medium",
            "Trace tight coupling to maintenance risk",
        )
        self.register_path(
            "Duplication Debt Chain",
            ["project_structure", "large_files", "code_duplication", "maintenance_risk"],
            "medium",
            "Trace large files through duplication to tech debt",
        )
