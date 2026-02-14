"""
Omnigent — Knowledge Base Loader

Intelligent, section-level knowledge retrieval for the agent.

Architecture:
1. Markdown files are split into section-level chunks on ``## `` headers
   (cached after first load — zero repeated I/O).
2. KNOWLEDGE_MAP maps context keys to file references with optional section hints.
3. Priority ordering + phase-adaptive token budgets.

Domain customisation:
  - Set KNOWLEDGE_DIR to point to your domain knowledge files
  - Populate KNOWLEDGE_MAP with your domain entries
  - Override get_relevant_knowledge() or PHASE_BUDGETS as needed

Example (security domain):
  KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
  KNOWLEDGE_MAP["sqli"] = [
      ("web_cheatsheet.md", "SQL Injection"),  # Only load SQL section
      "attack_chains/sqli_to_rce.md",          # Load full file
  ]
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from omnigent.context import estimate_tokens as _estimate_tokens
from omnigent.domain_profile import DomainProfile

# Override in domain implementation to point to your knowledge files
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"


# ═══════════════════════════════════════════════════════════════════════════
# Section-Level Chunking
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class _Chunk:
    """An individual section extracted from a knowledge markdown file."""
    heading: str   # Text after "## " (empty for intro before first heading)
    content: str   # Full section text including the heading line
    tokens: int    # Estimated token count via context.estimate_tokens()


def _flush(out: list[_Chunk], heading: str, buf: list[str]) -> None:
    """Flush accumulated lines into a Chunk if the section is non-trivial."""
    content = "\n".join(buf).strip()
    if content and len(content) > 80:
        out.append(_Chunk(heading=heading, content=content, tokens=_estimate_tokens(content)))


def _split_sections(text: str) -> list[_Chunk]:
    """Split markdown into sections on ``## `` headers, respecting code fences."""
    lines = text.split("\n")
    sections: list[_Chunk] = []
    heading = ""
    buf: list[str] = []
    in_fence = False

    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence

        if not in_fence and line.startswith("## "):
            _flush(sections, heading, buf)
            heading = line.lstrip("#").strip()
            buf = [line]
        else:
            buf.append(line)

    _flush(sections, heading, buf)
    return sections


@lru_cache(maxsize=128)
def _load_chunks(rel_path: str) -> tuple[_Chunk, ...]:
    """Load a markdown file and split into section chunks. Result is cached."""
    full_path = KNOWLEDGE_DIR / rel_path
    if not full_path.is_file():
        return ()
    try:
        text = full_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ()
    if not text.strip():
        return ()
    return tuple(_split_sections(text))


# ═══════════════════════════════════════════════════════════════════════════
# Knowledge Map — populate in your domain implementation
# ═══════════════════════════════════════════════════════════════════════════

# Each key maps to a list of entries. An entry is either:
#   "file.md"             → load ALL sections from the file
#   ("file.md", "hint")   → load only sections whose heading contains hint

_Entry = str | tuple[str, str]

KNOWLEDGE_MAP: dict[str, list[_Entry]] = {}


# ═══════════════════════════════════════════════════════════════════════════
# Phase-Adaptive Token Budgets
# ═══════════════════════════════════════════════════════════════════════════

PHASE_BUDGETS: dict[str, int] = {
    # Override in domain implementation, e.g.:
    # "recon": 1500,
    # "analysis": 2500,
    # "deep_investigation": 5000,
    # "reporting": 500,
}
DEFAULT_BUDGET = 3000


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def load_knowledge(keys: list[str], max_total_tokens: int = 3000) -> str:
    """Load knowledge sections for a list of context keys within a token budget.

    Uses section-level granularity: broad files are filtered to include only
    the sections matching the requested key.
    """
    seen_sections: set[str] = set()
    collected: list[str] = []
    budget_used = 0

    for key in keys:
        if budget_used >= max_total_tokens:
            break
        normalized = key.lower().strip().replace(" ", "_").replace("-", "_")
        entries = KNOWLEDGE_MAP.get(normalized, [])

        for entry in entries:
            if budget_used >= max_total_tokens:
                break

            if isinstance(entry, tuple):
                rel_path, section_hint = entry
            else:
                rel_path, section_hint = entry, None

            chunks = _load_chunks(rel_path)
            if not chunks:
                continue

            if section_hint:
                hint_lower = section_hint.lower()
                selected = [c for c in chunks if hint_lower in c.heading.lower()]
                if not selected:
                    continue
            else:
                selected = list(chunks)

            for chunk in selected:
                dedup_key = f"{rel_path}::{chunk.heading}"
                if dedup_key in seen_sections:
                    continue
                seen_sections.add(dedup_key)

                if budget_used + chunk.tokens > max_total_tokens:
                    remaining_chars = (max_total_tokens - budget_used) * 3
                    if remaining_chars > 500:
                        text = chunk.content[:remaining_chars] + "\n\n[... truncated for context budget ...]"
                        label = f"### {rel_path}" + (f" — {chunk.heading}" if chunk.heading else "")
                        collected.append(f"{label}\n{text}")
                        budget_used += _estimate_tokens(text)
                    break

                label = f"### {rel_path}" + (f" — {chunk.heading}" if chunk.heading else "")
                collected.append(f"{label}\n{chunk.content}")
                budget_used += chunk.tokens

    return "\n\n---\n\n".join(collected) if collected else ""


def get_relevant_knowledge(profile: DomainProfile, current_phase: str = "") -> str:
    """Auto-select and load relevant knowledge based on the DomainProfile.

    Override this in domain implementations for domain-specific priority.

    Default behaviour:
    1. Confirmed findings (need escalation guidance)
    2. Untested hypotheses (need investigation techniques)
    3. Current phase (methodology guidance)
    """
    keys: list[str] = []

    # Priority 1: Confirmed findings
    for hyp in profile.get_confirmed():
        keys.append(hyp.hypothesis_type)

    # Priority 2: Untested hypotheses
    for hyp in profile.get_untested_hypotheses():
        keys.append(hyp.hypothesis_type)

    # Priority 3: Current phase
    if current_phase:
        keys.append(current_phase)

    if not keys:
        return ""

    # Dedup preserving priority order
    seen: set[str] = set()
    unique: list[str] = []
    for k in keys:
        n = k.lower().strip().replace(" ", "_").replace("-", "_")
        if n not in seen:
            seen.add(n)
            unique.append(n)

    # Phase-adaptive budget
    phase_key = current_phase.lower().strip().replace(" ", "_")
    budget = PHASE_BUDGETS.get(phase_key, DEFAULT_BUDGET)

    return load_knowledge(unique, max_total_tokens=budget)
