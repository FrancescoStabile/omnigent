"""
CodeLens tools — domain-specific tools for code quality analysis.

Shows how to build tools and register them with the ToolRegistry.
These are example stubs that demonstrate the pattern — a real implementation
would use AST parsing, radon, pylint, etc.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from omnigent.tools import ToolRegistry


def register_codelens_tools(registry: ToolRegistry) -> None:
    """Register all CodeLens tools with the tool registry."""

    # ── file_scanner ─────────────────────────────────────────────
    registry.register(
        name="file_scanner",
        schema={
            "description": (
                "Scan a directory to discover source files, count lines of code, "
                "and map project structure. Returns file list with LOC per file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Root directory to scan",
                    },
                    "extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File extensions to include (e.g. ['.py', '.js'])",
                    },
                },
                "required": ["path"],
            },
        },
        handler=_file_scanner,
    )

    # ── complexity_analyzer ──────────────────────────────────────
    registry.register(
        name="complexity_analyzer",
        schema={
            "description": (
                "Measure cyclomatic complexity for a file or directory. "
                "Reports per-function complexity, averages, and hot spots."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "File or directory path to analyze",
                    },
                },
                "required": ["file"],
            },
        },
        handler=_complexity_analyzer,
    )

    # ── dependency_scanner ───────────────────────────────────────
    registry.register(
        name="dependency_scanner",
        schema={
            "description": (
                "Parse import statements and build a dependency graph. "
                "Optionally check for circular dependencies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Root directory to scan",
                    },
                    "check_circular": {
                        "type": "boolean",
                        "description": "Whether to detect circular imports",
                        "default": True,
                    },
                },
                "required": ["path"],
            },
        },
        handler=_dependency_scanner,
    )


# ── Tool Implementations (stubs — replace with real analysis) ────────────────


async def _file_scanner(path: str, extensions: list[str] | None = None, **kwargs) -> str:
    """Scan project structure. Real impl would use pathlib + AST."""
    path = Path(path)
    extensions = extensions or [".py"]

    if not path.exists():
        return f"Error: {path} does not exist"

    files = []
    total_loc = 0
    for ext in extensions:
        for f in path.rglob(f"*{ext}"):
            if ".git" in f.parts or "__pycache__" in f.parts:
                continue
            try:
                loc = len(f.read_text().splitlines())
            except (OSError, UnicodeDecodeError):
                loc = 0
            files.append((str(f.relative_to(path)), loc))
            total_loc += loc

    # Format output
    lines = [f"Project scan: {path}"]
    lines.append(f"Total: {len(files)} files, {total_loc:,} LOC")
    lines.append("")
    for fname, loc in sorted(files, key=lambda x: -x[1])[:20]:
        lines.append(f"  • {fname} ({loc} LOC)")
    if len(files) > 20:
        lines.append(f"  ... and {len(files) - 20} more files")

    return "\n".join(lines)


async def _complexity_analyzer(file: str, **kwargs) -> str:
    """Measure complexity. Real impl would use radon or ast module."""
    target = Path(file)

    if not target.exists():
        return f"Error: {target} does not exist"

    # Try using radon if available
    try:
        result = subprocess.run(
            ["radon", "cc", str(target), "-a", "-s"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: simple heuristic based on indentation depth and branches
    if target.is_file():
        content = target.read_text()
        branches = sum(
            1
            for line in content.splitlines()
            if any(kw in line for kw in ["if ", "elif ", "for ", "while ", "except "])
        )
        functions = sum(1 for line in content.splitlines() if line.strip().startswith("def "))
        avg_cc = branches / max(functions, 1) + 1
        return (
            f"{target.name}: {functions} functions, ~{branches} branches\n"
            f"Estimated avg CC={avg_cc:.1f}, max CC≈{avg_cc * 2:.1f}"
        )

    return "Unable to analyze (not a file)"


async def _dependency_scanner(path: str, check_circular: bool = True, **kwargs) -> str:
    """Scan imports. Real impl would build a full graph."""
    path = Path(path)

    if not path.exists():
        return f"Error: {path} does not exist"

    imports: dict[str, list[str]] = {}
    for py_file in path.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        module = str(py_file.relative_to(path)).replace("/", ".").removesuffix(".py")
        deps = []
        try:
            for line in py_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("import ") or line.startswith("from "):
                    deps.append(line.split()[1].split(".")[0])
        except (OSError, UnicodeDecodeError):
            continue
        imports[module] = deps

    # Simple circular dependency detection
    circular = []
    if check_circular:
        for mod_a, deps_a in imports.items():
            for dep in deps_a:
                if dep in imports and mod_a.split(".")[0] in imports.get(dep, []):
                    pair = tuple(sorted([mod_a.split(".")[0], dep]))
                    if pair not in [(c[0], c[1]) for c in circular]:
                        circular.append(pair)

    total = sum(len(d) for d in imports.values())
    lines = [f"Dependency scan: {path}"]
    lines.append(f"{len(imports)} modules, {total} import statements")
    if circular:
        lines.append(f"\n⚠️ {len(circular)} circular dependencies:")
        for a, b in circular:
            lines.append(f"  • {a} ↔ {b}")
    else:
        lines.append("✅ No circular dependencies detected")

    return "\n".join(lines)
