# CodeLens — Example Domain Implementation

This is a complete example showing how to build a domain-specific agent on Omnigent.

## Files

| File | Purpose |
|------|---------|
| `profile.py` | `CodeProfile` subclass of `DomainProfile` — structured domain memory |
| `graph.py` | `CodeQualityGraph` subclass of `ReasoningGraph` — 13 nodes, 12 edges, 4 paths |
| `registries.py` | Populates ALL empty registries: plans, chains, extractors, reflectors, errors, knowledge, examples |
| `tools.py` | 3 domain tools: `file_scanner`, `complexity_analyzer`, `dependency_scanner` |
| `main.py` | Entry point — shows the complete wiring sequence |
| `knowledge/` | Domain knowledge files referenced by `KNOWLEDGE_MAP` |

## Usage

```bash
cd /path/to/omnigent
python -m examples.codelens.main /path/to/repo
```

## Extension Points Demonstrated

1. **DomainProfile** — `CodeProfile` with repo metadata, metrics, and architecture fields
2. **ReasoningGraph** — 4-layer graph (Discovery → Metrics → Patterns → Impact)
3. **PLAN_TEMPLATES** — Python and JavaScript project templates
4. **CHAINS** — 4 escalation chains (god_class, circular_dep, high_complexity, low_coverage)
5. **EXTRACTORS** — 3 parsers (file_scanner, complexity, dependency)
6. **REFLECTORS** — 2 strategic reflectors (complexity, circular deps)
7. **ERROR_PATTERNS** — 5 recovery strategies (permission, binary, parse, timeout, large repo)
8. **KNOWLEDGE_MAP** — 4 knowledge topics with phase budgets
9. **EXAMPLES** — 3 few-shot examples
10. **enrich_fn** — Finding enrichment with ISO 25010 quality dimensions
