"""
CodeLens registry population — wires all domain-specific data into Omnigent.

Import this module once at startup to populate all empty registries.
"""

from __future__ import annotations

from omnigent.chains import CHAINS, ChainStep
from omnigent.error_recovery import ERROR_PATTERNS, RecoveryStrategy
from omnigent.extractors import EXTRACTORS
from omnigent.few_shot_examples import EXAMPLES, ToolExample
from omnigent.knowledge_loader import KNOWLEDGE_MAP, PHASE_BUDGETS
from omnigent.planner import PLAN_TEMPLATES
from omnigent.reflection import REFLECTORS


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. PLAN TEMPLATES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PLAN_TEMPLATES["python_project"] = [
    {
        "name": "Discovery",
        "objective": "Map project structure, dependencies, and framework",
        "steps": [
            ("Scan project tree and count files/LOC", "file_scanner"),
            ("Parse requirements and imports", "dependency_scanner"),
            ("Detect framework and language version", "framework_detector"),
        ],
    },
    {
        "name": "Metrics Collection",
        "objective": "Collect quantitative code quality metrics",
        "steps": [
            ("Measure cyclomatic complexity per file", "complexity_analyzer"),
            ("Detect code duplication", "duplication_scanner"),
            ("Measure test coverage", "coverage_analyzer"),
        ],
    },
    {
        "name": "Pattern Analysis",
        "objective": "Identify code smells and anti-patterns",
        "steps": [
            ("Identify god classes and long methods", "ast_analyzer"),
            ("Check for circular dependencies", "dependency_scanner"),
            ("Find dead/unreachable code", "dead_code_scanner"),
        ],
    },
    {
        "name": "Impact Assessment",
        "objective": "Rate findings by maintainability and risk impact",
        "steps": [
            ("Cross-reference hot spots with git churn", "git_analyzer"),
            ("Assess testability of flagged modules", "testability_scorer"),
            ("Generate prioritised recommendations", ""),
        ],
    },
]

PLAN_TEMPLATES["javascript_project"] = [
    {
        "name": "Discovery",
        "objective": "Map JS/TS project structure",
        "steps": [
            ("Scan project tree", "file_scanner"),
            ("Parse package.json dependencies", "dependency_scanner"),
            ("Detect bundler and framework", "framework_detector"),
        ],
    },
    {
        "name": "Analysis",
        "objective": "Identify quality issues",
        "steps": [
            ("Run ESLint analysis", "linter_runner"),
            ("Measure bundle size", "bundle_analyzer"),
            ("Check for unused dependencies", "dependency_scanner"),
        ],
    },
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. CHAINS (escalation paths when a finding is confirmed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHAINS["god_class"] = [
    ChainStep("Identify god class methods and responsibilities", "ast_analyzer"),
    ChainStep("Map incoming/outgoing dependencies", "dependency_scanner"),
    ChainStep("Calculate single-responsibility violations", ""),
    ChainStep("Generate decomposition recommendations", ""),
]

CHAINS["circular_dependency"] = [
    ChainStep("Trace full dependency cycle", "dependency_scanner"),
    ChainStep("Identify weakest link in cycle", ""),
    ChainStep("Suggest dependency inversion points", ""),
]

CHAINS["high_complexity"] = [
    ChainStep("Break down complex function into branches", "ast_analyzer"),
    ChainStep("Identify extract-method opportunities", ""),
    ChainStep("Estimate refactoring effort in story points", ""),
]

CHAINS["low_coverage"] = [
    ChainStep("Identify untested public methods", "coverage_analyzer"),
    ChainStep("Prioritise by complexity and change frequency", "git_analyzer"),
    ChainStep("Generate test stubs for critical paths", ""),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. EXTRACTORS (auto-parse tool results into CodeProfile)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_file_scanner(profile, result: str, args: dict) -> None:
    """Extract project structure from file_scanner output."""
    import re
    # Example: "Total: 142 files, 12,345 LOC"
    m = re.search(r"(\d[\d,]*)\s*files.*?([\d,]+)\s*LOC", result)
    if m:
        profile.file_count = int(m.group(1).replace(",", ""))
        profile.loc = int(m.group(2).replace(",", ""))
    # Collect scanned file paths
    for line in result.splitlines():
        line = line.strip()
        if line.startswith("- ") or line.startswith("• "):
            profile.files_scanned.append(line.lstrip("-• ").strip())


def _extract_complexity(profile, result: str, args: dict) -> None:
    """Extract complexity metrics."""
    import re
    m = re.search(r"avg.*?(\d+\.?\d*)", result, re.IGNORECASE)
    if m:
        profile.avg_complexity = float(m.group(1))
    m = re.search(r"max.*?(\d+\.?\d*)", result, re.IGNORECASE)
    if m:
        profile.max_complexity = float(m.group(1))


def _extract_dependencies(profile, result: str, args: dict) -> None:
    """Extract dependency info."""
    import re
    # Count dependencies
    m = re.search(r"(\d+)\s*dependenc", result, re.IGNORECASE)
    if m:
        profile.dependency_count = int(m.group(1))
    # Detect circular deps
    for line in result.splitlines():
        if "circular" in line.lower() or "cycle" in line.lower():
            profile.circular_deps.append(line.strip())


EXTRACTORS["file_scanner"] = _extract_file_scanner
EXTRACTORS["complexity_analyzer"] = _extract_complexity
EXTRACTORS["dependency_scanner"] = _extract_dependencies


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. REFLECTORS (strategic analysis after each tool call)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _reflect_on_complexity(result: str, args: dict, profile, lines: list[str]) -> None:
    """After complexity analysis, generate strategic reflection."""
    if profile.max_complexity and profile.max_complexity > 20:
        lines.append(
            f"⚠️ CRITICAL: Max complexity is {profile.max_complexity:.0f} "
            f"(threshold: 20). Prioritise god-class and extract-method analysis. "
            f"High complexity correlates with 4x more bugs (McCabe 1976)."
        )


def _reflect_on_circular_deps(result: str, args: dict, profile, lines: list[str]) -> None:
    """After dependency scan, flag circular dependencies."""
    if profile.circular_deps:
        lines.append(
            f"🔄 Found {len(profile.circular_deps)} circular dependencies. "
            f"These block independent testing and deployment. "
            f"Escalate to dependency inversion chain."
        )


REFLECTORS["complexity_analyzer"] = _reflect_on_complexity
REFLECTORS["dependency_scanner"] = _reflect_on_circular_deps


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. ERROR PATTERNS (recovery guidance when tools fail)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ERROR_PATTERNS["file_scanner"] = {
    "permission_denied": {
        "indicators": ["permission denied"],
        "strategy": RecoveryStrategy(
            guidance="The file or directory is not readable. Try running with elevated permissions or skip this file.",
            give_up=True,
        ),
    },
    "binary_file": {
        "indicators": ["binary file", "cannot read binary"],
        "strategy": RecoveryStrategy(
            guidance="This is a binary file, skip it and move to the next source file.",
            give_up=True,
        ),
    },
}

ERROR_PATTERNS["complexity_analyzer"] = {
    "parse_error": {
        "indicators": ["syntax error", "parse error", "unexpected token"],
        "strategy": RecoveryStrategy(
            guidance="File has syntax errors. Note this as a finding and continue analysis.",
            give_up=True,
        ),
    },
    "timeout": {
        "indicators": ["timeout", "timed out"],
        "strategy": RecoveryStrategy(
            guidance="Tool timed out. Try with a smaller scope — analyze one directory at a time.",
        ),
    },
}

ERROR_PATTERNS["dependency_scanner"] = {
    "large_repo": {
        "indicators": ["too many files", "memory", "out of memory"],
        "strategy": RecoveryStrategy(
            guidance="Repository is too large for full scan. Use sampling — analyze top-level modules individually.",
        ),
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. KNOWLEDGE MAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KNOWLEDGE_MAP["complexity"] = "knowledge/complexity_cheatsheet.md"
KNOWLEDGE_MAP["refactoring"] = "knowledge/refactoring_patterns.md"
KNOWLEDGE_MAP["testing"] = "knowledge/testing_strategies.md"
KNOWLEDGE_MAP["architecture"] = "knowledge/architecture_patterns.md"

PHASE_BUDGETS["Discovery"] = 1500
PHASE_BUDGETS["Metrics Collection"] = 2000
PHASE_BUDGETS["Pattern Analysis"] = 2500
PHASE_BUDGETS["Impact Assessment"] = 2000


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. FEW-SHOT EXAMPLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLES["file_scanner"] = [
    ToolExample(
        scenario="Scan a Python project structure",
        thinking="Need to understand project layout before analysis",
        tool_name="file_scanner",
        tool_args={"path": "/opt/repo", "extensions": [".py"]},
        expected_result="Found 87 Python files across 12 packages, 9,412 LOC",
        is_good=True,
    ),
]

EXAMPLES["complexity_analyzer"] = [
    ToolExample(
        scenario="Measure complexity of a module",
        thinking="High-complexity files are refactoring candidates",
        tool_name="complexity_analyzer",
        tool_args={"file": "/opt/repo/src/core/engine.py"},
        expected_result="engine.py: avg CC=8.2, max CC=34 (process_request), 12 functions",
        is_good=True,
    ),
]

EXAMPLES["dependency_scanner"] = [
    ToolExample(
        scenario="Map imports and find cycles",
        thinking="Circular dependencies block independent deployment",
        tool_name="dependency_scanner",
        tool_args={"path": "/opt/repo/src", "check_circular": True},
        expected_result="142 imports, 3 circular dependencies: core↔utils, db↔models, api↔auth",
        is_good=True,
    ),
]
