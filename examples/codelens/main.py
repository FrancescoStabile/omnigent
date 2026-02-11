"""
CodeLens main — entry point showing how to wire everything together.

This file demonstrates the complete bootstrap sequence:
1. Import registries (populates all empty dicts)
2. Create domain profile + reasoning graph
3. Create state with enrichment function
4. Build agent with all components
5. Run the analysis loop
"""

from __future__ import annotations

import asyncio
import sys

from omnigent.agent import Agent
from omnigent.planner import generate_plan
from omnigent.router import LLMRouter, Provider
from omnigent.state import Finding, State
from omnigent.tools import ToolRegistry

# Step 1: Import registries — this populates all the empty dicts
from examples.codelens import registries  # noqa: F401
from examples.codelens.graph import CodeQualityGraph
from examples.codelens.profile import CodeProfile
from examples.codelens.tools import register_codelens_tools


# ── Domain Enrichment Function ───────────────────────────────────────────────

def enrich_finding(finding: Finding) -> Finding:
    """
    Enrich a finding with domain-specific metadata.

    In NumaSec this adds CWE IDs, OWASP categories, and CVSS scores.
    For CodeLens we add software quality metrics.
    """
    severity_to_priority = {
        "critical": "P0 — fix immediately",
        "high": "P1 — fix this sprint",
        "medium": "P2 — plan for next sprint",
        "low": "P3 — backlog",
        "info": "P4 — informational",
    }

    finding.enrichment["priority"] = severity_to_priority.get(
        finding.severity, "P3 — backlog"
    )

    # Map finding types to quality dimensions
    title_lower = finding.title.lower()
    if "complexity" in title_lower or "god class" in title_lower:
        finding.enrichment["quality_dimension"] = "Maintainability"
        finding.enrichment["iso_25010"] = "Maintainability > Modularity"
    elif "coupling" in title_lower or "circular" in title_lower:
        finding.enrichment["quality_dimension"] = "Modularity"
        finding.enrichment["iso_25010"] = "Maintainability > Modularity"
    elif "coverage" in title_lower or "test" in title_lower:
        finding.enrichment["quality_dimension"] = "Reliability"
        finding.enrichment["iso_25010"] = "Reliability > Maturity"
    elif "duplication" in title_lower:
        finding.enrichment["quality_dimension"] = "Maintainability"
        finding.enrichment["iso_25010"] = "Maintainability > Reusability"

    return finding


# ── Main ─────────────────────────────────────────────────────────────────────

async def run_analysis(repo_path: str) -> None:
    """Run CodeLens analysis on a repository."""

    # Step 2: Create domain objects
    profile = CodeProfile()
    profile.subject = repo_path
    profile.scope = [repo_path]

    graph = CodeQualityGraph()

    # Step 3: Create state with enrichment
    state = State(subject=repo_path, enrich_fn=enrich_finding)
    state.profile = profile

    # Step 4: Build components
    router = LLMRouter(
        primary=Provider.DEEPSEEK,
        fallback=Provider.CLAUDE,
    )
    tools = ToolRegistry(allowed_targets=[repo_path])

    # Step 5: Register domain tools
    register_codelens_tools(tools)

    # Step 6: Build agent
    agent = Agent(
        router=router,
        tools=tools,
        reasoning_graph=graph,
    )
    agent.state = state  # Inject domain-configured state

    # Step 7: Run
    print(f"🔍 CodeLens — Analyzing: {repo_path}\n")

    async for event in agent.run(
        f"Perform a comprehensive code quality analysis of {repo_path}"
    ):
        if event.type == "text":
            print(event.content, end="", flush=True)
        elif event.type == "tool_call":
            print(f"\n🔧 {event.tool_name}({event.arguments})")
        elif event.type == "finding":
            f = event.finding
            priority = f.enrichment.get("priority", "?")
            dimension = f.enrichment.get("quality_dimension", "General")
            print(f"\n{'='*60}")
            print(f"[{f.severity.upper()}] {f.title}")
            print(f"Priority: {priority} | Dimension: {dimension}")
            print(f"{'='*60}")

    # Summary
    print("\n\n📊 Analysis Complete")
    print(f"Findings: {len(state.findings)}")
    print(f"Messages: {len(state.messages)}")
    for sev in ["critical", "high", "medium", "low", "info"]:
        count = sum(1 for f in state.findings if f.severity == sev)
        if count:
            print(f"  {sev}: {count}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m examples.codelens.main <repo_path>")
        print("Example: python -m examples.codelens.main /path/to/your/project")
        sys.exit(1)

    asyncio.run(run_analysis(sys.argv[1]))


if __name__ == "__main__":
    main()
