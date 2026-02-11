<div align="center">

# Omnigent

**The universal scaffold for building autonomous AI agents.**

Build any AI agent — security, code analysis, DevOps, compliance, research — on a production-proven foundation. Extracted from a real-world agent with 17k+ LOC and 320 tests.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-109%20passing-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

[Architecture](ARCHITECTURE.md) · [Examples](examples/) · [Contributing](CONTRIBUTING.md)

</div>

---

## What is Omnigent?

Most AI agent frameworks give you wrappers around LLM APIs. Omnigent gives you **the entire brain**.

It's the domain-agnostic architecture of a production autonomous agent — the ReAct loop, multi-provider LLM routing, structured memory, hierarchical planning, reasoning graphs, error recovery, reflection, and plugin system. Everything you need to build a **real** agent, not a chatbot with tools.

**You bring the domain. Omnigent brings the intelligence.**

```
┌──────────────────────────────────────────────────────┐
│                   Agent Loop (ReAct)                 │
│         Reason → Act → Observe → Reflect             │
├──────────────┬────────────┬────────────┬─────────────┤
│   Router     │  Planner   │  Context   │   Graph     │
│  4 Providers │  Phases    │  Smart Trim│  Reasoning  │
├──────────────┴────────────┴────────────┴─────────────┤
│            Post-Processing Pipeline                  │
│    Extractors → Reflection → Error Recovery          │
├──────────────────────────────────────────────────────┤
│            Tool Registry + Plugin System             │
├──────────────────────────────────────────────────────┤
│  State │ DomainProfile │ Session │ Cost │ Knowledge  │
├──────────────────────────────────────────────────────┤
│          Config │ Logging │ MCP Integration          │
└──────────────────────────────────────────────────────┘
```

## Why Omnigent?

| Problem | Omnigent Solution |
|---------|-------------------|
| Agents that loop forever | **Circuit breaker** + **loop detection** (hash-based, blocks on first repeat) |
| Context window overflow | **3-level smart trimming** preserving atomic message groups |
| "Just a tool caller" | **Reasoning Graph** chains findings into multi-step escalation paths |
| No methodology | **Hierarchical Planner** with phase-based execution and LLM refinement |
| Blind tool execution | **Extractors** auto-parse results → **structured memory** → **reflection** |
| Failures crash the agent | **Error recovery patterns** with retry strategies and graceful degradation |
| Vendor lock-in | **4 LLM providers** with task-based routing 和 automatic fallback |
| Starting from zero | **Production-proven** — extracted from a real agent, not built in a weekend |

## Quick Start

### Install

```bash
pip install -e .
```

### Set up an API key

```bash
export DEEPSEEK_API_KEY="sk-..."  # Cheapest option (~$0.001 per analysis)
# or
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."
```

### Run the example agent

```bash
# CodeLens — code quality analyzer (included example)
python -m examples.codelens.main /path/to/any/project
```

### Build your own agent in 4 steps

**Step 1: Define your domain memory**

```python
from dataclasses import dataclass, field
from omnigent.domain_profile import DomainProfile

@dataclass
class MyProfile(DomainProfile):
    items_analyzed: list[str] = field(default_factory=list)
    risk_score: float = 0.0
```

**Step 2: Register your tools**

```python
from omnigent.tools import ToolRegistry

registry = ToolRegistry()
registry.register(
    name="my_scanner",
    schema={"description": "Scan a target", "parameters": {
        "type": "object",
        "properties": {"target": {"type": "string"}},
        "required": ["target"],
    }},
    handler=my_scanner_function,
)
```

**Step 3: Populate registries** (plan templates, chains, extractors, reflectors, error patterns)

```python
from omnigent.planner import PLAN_TEMPLATES
from omnigent.chains import CHAINS, ChainStep
from omnigent.extractors import EXTRACTORS

PLAN_TEMPLATES["my_domain"] = [
    {"name": "Discovery", "objective": "Map the target", "steps": [
        ("Initial scan", "my_scanner"),
    ]},
]

CHAINS["high_risk"] = [
    ChainStep("Deep dive on flagged items", "deep_scanner"),
    ChainStep("Generate remediation plan", ""),
]

EXTRACTORS["my_scanner"] = lambda profile, result, args: setattr(
    profile, 'risk_score', 0.8
)
```

**Step 4: Wire it up and run**

```python
import asyncio
from omnigent.agent import Agent
from omnigent.router import LLMRouter, Provider

async def main():
    agent = Agent(
        router=LLMRouter(primary=Provider.DEEPSEEK),
        tools=registry,
    )
    async for event in agent.run("Analyze this target"):
        if event.type == "text":
            print(event.content, end="")
        elif event.type == "finding":
            print(f"\n[{event.finding.severity}] {event.finding.title}")

asyncio.run(main())
```

## What Can You Build?

Omnigent is **domain-agnostic** — it provides the intelligence architecture, you provide the domain knowledge:

| Domain | What You Add | What Omnigent Provides |
|--------|-------------|------------------------|
| **Security** | Nmap, SQLMap, Burp tools + vuln knowledge | ReAct loop, attack chain reasoning, session persistence |
| **Code Quality** | AST parsers, complexity tools + refactoring patterns | Planning, structured findings, escalation chains |
| **DevOps** | K8s, Terraform, monitoring tools + runbooks | Error recovery, multi-step incident chains, cost tracking |
| **Compliance** | Document scanners, policy tools + regulation KB | Hypothesis tracking, evidence collection, reporting |
| **Research** | Search, scraping, DB tools + domain ontology | Context management, iterative refinement, reflection |

See [examples/codelens/](examples/codelens/) for a complete working implementation.

## Components

| Module | Purpose | How to Customize |
|--------|---------|-----------------|
| [agent.py](src/omnigent/agent.py) | ReAct loop, circuit breaker, loop detection | Subclass `Agent`, override hooks |
| [router.py](src/omnigent/router.py) | Multi-provider LLM routing (DeepSeek, Claude, OpenAI, Ollama) | Works as-is |
| [reasoning_graph.py](src/omnigent/reasoning_graph.py) | Directed graph for multi-step reasoning chains | Subclass `ReasoningGraph` |
| [planner.py](src/omnigent/planner.py) | Hierarchical task planning with phases and steps | Populate `PLAN_TEMPLATES` |
| [context.py](src/omnigent/context.py) | Smart context trimming with atomic grouping | Works as-is |
| [domain_profile.py](src/omnigent/domain_profile.py) | Structured memory with hypothesis tracking | Subclass `DomainProfile` |
| [state.py](src/omnigent/state.py) | Agent state with Pydantic-validated findings | Set `enrich_fn` hook |
| [extractors.py](src/omnigent/extractors.py) | Auto-parse tool results into DomainProfile | Populate `EXTRACTORS` |
| [reflection.py](src/omnigent/reflection.py) | Strategic insight after each tool call | Populate `REFLECTORS` |
| [error_recovery.py](src/omnigent/error_recovery.py) | Pattern-matched recovery guidance | Populate `ERROR_PATTERNS` |
| [chains.py](src/omnigent/chains.py) | Escalation chains for confirmed findings | Populate `CHAINS` |
| [knowledge_loader.py](src/omnigent/knowledge_loader.py) | Section-level knowledge retrieval with budgets | Populate `KNOWLEDGE_MAP` |
| [few_shot_examples.py](src/omnigent/few_shot_examples.py) | Tool usage examples for improved accuracy | Populate `EXAMPLES` |
| [plugins.py](src/omnigent/plugins.py) | Filesystem plugin discovery and loading | Drop into `~/.omnigent/plugins/` |
| [session.py](src/omnigent/session.py) | Session persistence, resume, export | Works as-is |
| [cost_tracker.py](src/omnigent/cost_tracker.py) | Per-provider, per-task cost tracking | Works as-is |
| [config.py](src/omnigent/config.py) | YAML + .env + ENV config loading | Works as-is |
| [tools/](src/omnigent/tools/__init__.py) | Tool registry with scope checking | Register domain tools |

## Key Design Patterns

### Data-Driven Registries (Zero Domain Code in Core)
All domain-specific behavior lives in **dictionaries that start empty**. Your agent populates them at startup:

```python
PLAN_TEMPLATES  # Task plan templates
CHAINS          # Escalation chains
EXTRACTORS      # Tool result parsers
REFLECTORS      # Post-tool strategic analysis
ERROR_PATTERNS  # Failure recovery patterns
KNOWLEDGE_MAP   # Knowledge file routing
EXAMPLES        # Few-shot tool examples
TOOL_TIMEOUTS   # Per-tool timeouts
```

### Subclass for Complex Domains
For behavior that can't be expressed as data, override methods:

```python
class MyAgent(Agent):
    def _is_failure(self, tool_name, result): ...
    def _extract_finding(self, text): ...
    def _build_dynamic_system_prompt(self): ...

class MyGraph(ReasoningGraph):
    def _build_default_graph(self): ...

class MyProfile(DomainProfile):
    def to_prompt_summary(self): ...
```

### The Reasoning Graph (The Differentiator)
What separates Omnigent from simple tool-callers: when a finding is confirmed, the reasoning graph activates downstream escalation paths. The agent doesn't just find issues — it **chains them into multi-step reasoning**.

```
Security:    SQLi → DB Dump → Credential Extraction → Admin Access → RCE
Code:        God Object → High Coupling → Low Testability → Regression Risk
Incident:    Alert → Log Correlation → Root Cause → Blast Radius
Compliance:  Gap Found → Control Missing → Risk Assessment → Remediation
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=omnigent --cov-report=term-missing

# Run specific module tests
pytest tests/test_reasoning_graph.py -v

# Run only unit tests
pytest -m unit
```

109 tests covering all core components. Every test runs without LLM calls or network access.

## Project Structure

```
omnigent/
├── src/omnigent/         # Core framework
│   ├── agent.py           # The ReAct loop (688 lines)
│   ├── router.py          # Multi-provider LLM routing (571 lines)
│   ├── reasoning_graph.py # Chain reasoning engine (377 lines)
│   ├── planner.py         # Hierarchical task planner (402 lines)
│   ├── context.py         # Smart context management (229 lines)
│   ├── state.py           # State + Pydantic findings
│   ├── domain_profile.py  # Structured agent memory
│   ├── extractors.py      # Result parsing pipeline
│   ├── reflection.py      # Post-tool strategic analysis
│   ├── error_recovery.py  # Failure recovery engine
│   ├── chains.py          # Escalation chain registry
│   ├── knowledge_loader.py # Knowledge base retrieval
│   ├── few_shot_examples.py # Tool usage examples
│   ├── plugins.py         # Plugin system (487 lines)
│   ├── session.py         # Session persistence
│   ├── cost_tracker.py    # Cost tracking
│   ├── config.py          # Configuration management
│   ├── logging_config.py  # Structured JSON logging
│   ├── prompts/system.md  # Base system prompt
│   └── tools/             # Tool registry
├── examples/codelens/     # Complete working example agent
├── tests/                 # 109 tests
├── ARCHITECTURE.md        # Deep technical architecture doc
├── CONTRIBUTING.md        # Contribution guide
└── CHANGELOG.md           # Version history
```

## Origin

Omnigent was extracted from [NumaSec](https://github.com/francescostabile/numasec), a production autonomous security agent (17,878 LOC, 320 tests).

**Built by [Francesco Stabile](https://www.linkedin.com/in/francesco-stabile-dev)**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/francesco-stabile-dev)
[![X](https://img.shields.io/badge/X-000000?style=flat-square&logo=x&logoColor=white)](https://x.com/Francesco_Sta)


## License

[MIT](LICENSE) — use it for anything.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). We welcome domain implementations, bug fixes, and documentation improvements.
