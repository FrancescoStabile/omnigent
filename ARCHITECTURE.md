# Omnigent — Architecture

> Deep technical documentation of the Omnigent framework architecture.
> For usage and quick start, see [README.md](README.md).

## Table of Contents

- [Design Philosophy](#design-philosophy)
- [System Overview](#system-overview)
- [The Agent Loop (ReAct)](#the-agent-loop-react)
- [LLM Router](#llm-router)
- [Reasoning Graph](#reasoning-graph)
- [Hierarchical Planner](#hierarchical-planner)
- [Context Management](#context-management)
- [Domain Profile & State](#domain-profile--state)
- [Post-Processing Pipeline](#post-processing-pipeline)
- [Tool System](#tool-system)
- [Plugin Architecture](#plugin-architecture)
- [Session Persistence](#session-persistence)
- [Extension Model](#extension-model)
- [Data Flow](#data-flow)
- [Design Decisions](#design-decisions)

---

## Design Philosophy

Omnigent is built on four principles:

1. **Domain-agnostic core, domain-specific extensions.** No security, code quality, or DevOps logic exists in the core. All domain behavior flows through registries (empty dicts) and subclass hooks.

2. **Never crash.** Every extension point (extractors, reflectors, plugins, enrichment hooks) wraps execution in try/except. A bad plugin or malformed tool result must never stop the agent loop.

3. **Structured intelligence over raw prompting.** The agent maintains a DomainProfile (structured memory), TaskPlan (methodology), and ReasoningGraph (chain logic). These are injected into every LLM call as structured context — the LLM sees *what it knows*, *what to do next*, and *what chains to pursue*.

4. **Production-first.** Every component was extracted from a production agent (NumaSec, 17k+ LOC). Nothing is theoretical — circuit breakers, loop detection, context trimming, session resume all exist because production required them.

---

## System Overview

```
                          ┌─────────────┐
                          │  User Input │
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │   Planner   │ ← PLAN_TEMPLATES + LLM refinement
                          │ (TaskPlan)  │
                          └──────┬──────┘
                                 │
                    ┌────────────▼────────────┐
                    │      Agent Loop         │
                    │   (ReAct Iteration)     │
                    │                         │
                    │  ┌───────────────────┐  │
                    │  │ Build System      │  │  ← DomainProfile + TaskPlan
                    │  │ Prompt (Dynamic)  │  │    + ReasoningGraph + Knowledge
                    │  └────────┬──────────┘  │
                    │           │             │
                    │  ┌────────▼──────────┐  │
                    │  │  LLM Router       │  │  ← Task-based provider selection
                    │  │  (Stream Response)│  │    + fallback + streaming
                    │  └────────┬──────────┘  │
                    │           │             │
                    │     ┌─────┴─────┐       │
                    │     │           │       │
                    │  [Text]    [Tool Calls] │
                    │     │           │       │
                    │     │    ┌──────▼─────┐ │
                    │     │    │ Loop Check │ │  ← Hash-based repeat detection
                    │     │    └──────┬─────┘ │
                    │     │           │       │
                    │     │    ┌──────▼─────┐ │
                    │     │    │  Execute   │ │  ← ToolRegistry.call() + timeout
                    │     │    │  Tool      │ │
                    │     │    └──────┬─────┘ │
                    │     │           │       │
                    │     │    ┌──────▼─────┐ │
                    │     │    │ Extractor  │ │  ← EXTRACTORS[tool_name]
                    │     │    │ Pipeline   │ │    → Updates DomainProfile
                    │     │    └──────┬─────┘ │
                    │     │           │       │
                    │     │    ┌──────▼─────┐ │
                    │     │    │ Reflection │ │  ← REFLECTORS[tool_name]
                    │     │    │ Engine     │ │    → Strategic insight
                    │     │    └──────┬─────┘ │
                    │     │           │       │
                    │     │    ┌──────▼─────┐ │
                    │     │    │ Error      │ │  ← ERROR_PATTERNS[tool_name]
                    │     │    │ Recovery   │ │    → Recovery if failure
                    │     │    └──────┬─────┘ │
                    │     │           │       │
                    │     │    [Plan Step ✓] │  ← Mark step complete
                    │     │           │       │
                    │  ┌──▼───────────▼────┐  │
                    │  │  Context Trim     │  │  ← 3-level smart trimming
                    │  │  (if needed)      │  │
                    │  └──────────┬────────┘  │
                    │             │           │
                    │      [Next Iteration]   │
                    │             or          │
                    │      [Termination]      │  ← Plan complete / done indicators
                    └─────────────────────────┘
                                 │
                          ┌──────▼──────┐
                          │   Session   │  ← Auto-save, export, resume
                          │   Persist   │
                          └─────────────┘
```

---

## The Agent Loop (ReAct)

**File:** `agent.py` (688 lines)

The core of Omnigent is a ReAct (Reason-Act-Observe) loop with several production hardening layers.

### Iteration Cycle

Each iteration:
1. **Check context budget** — trim if messages exceed token threshold
2. **Build dynamic system prompt** — base prompt + DomainProfile + TaskPlan + ReasoningGraph + Knowledge
3. **Stream LLM response** — via Router with task-based provider selection
4. **Handle text tokens** — emit to UI, check for finding patterns (`[FINDING: SEVERITY]`)
5. **Handle tool calls** — loop check → execute → extract → reflect → recover
6. **Update plan** — mark completed steps, advance phases
7. **Check termination** — plan complete, done indicators, or consecutive text-only responses

### Safety Mechanisms

| Mechanism | How It Works |
|-----------|-------------|
| **Loop Detection** | MD5 hash of `{tool_name, args}`. Blocks on first repeated identical call. Recent history: 10 calls. |
| **Circuit Breaker** | Tracks repeated LLM errors. After 3 identical errors, stops the loop entirely. |
| **Adaptive Timeouts** | Per-tool timeout via `TOOL_TIMEOUTS` dict. Default: 300s. |
| **Max Iterations** | Hard cap (default: 50). Prevents infinite loops on edge cases. |
| **Termination Detection** | Pattern matching on text for "task complete", "summary", etc. Also: 3+ consecutive text-only responses = done. |

### Event System

The agent yields `AgentEvent` objects for the UI layer:

```python
AgentEvent("text", content="...")           # LLM text token
AgentEvent("tool_start", tool_name="...")   # Tool execution starting
AgentEvent("tool_end", tool_result="...")   # Tool execution finished
AgentEvent("finding", finding=Finding(...)) # New finding registered
AgentEvent("plan_generated", plan="...")    # Plan created
AgentEvent("phase_complete", ...)           # Phase advanced
AgentEvent("reflection", guidance="...")    # Error recovery guidance
AgentEvent("usage", input_tokens=N, ...)   # Token usage stats
AgentEvent("error", message="...")          # Error occurred
AgentEvent("done")                          # Loop finished
```

---

## LLM Router

**File:** `router.py` (571 lines)

Multi-provider LLM routing with streaming, automatic fallback, and task-based selection.

### Supported Providers

| Provider | Model | Cost (1K tokens) | Best For |
|----------|-------|-------------------|----------|
| DeepSeek | deepseek-chat | $0.00014 in / $0.00028 out | Tool use, general (cheapest) |
| Claude | claude-sonnet-4 | $0.003 in / $0.015 out | Planning, analysis, reports |
| OpenAI | gpt-4o-mini | $0.00015 in / $0.0006 out | Tool use, general |
| Ollama | qwen2.5-coder:3b | Free (local) | Reflection, simple tasks |

### Task-Based Routing

The router selects the best available provider for each task type:

```
PLANNING   → Claude > DeepSeek
TOOL_USE   → DeepSeek > OpenAI
ANALYSIS   → Claude > DeepSeek
REFLECTION → DeepSeek > Local
REPORT     → Claude > OpenAI
```

### Fallback Chain

If the primary provider fails, the router automatically falls back:
1. Task-appropriate provider (if configured)
2. Primary provider
3. Fallback provider
4. Error event (circuit breaker may trigger)

### Streaming Architecture

All LLM calls use streaming via `httpx`. Each provider has its own streaming parser:
- **DeepSeek/OpenAI**: OpenAI-compatible SSE stream
- **Claude**: Anthropic SSE stream with content blocks
- **Ollama**: OpenAI-compatible endpoint

The router yields `StreamChunk` objects with content, tool calls, and usage stats.

---

## Reasoning Graph

**File:** `reasoning_graph.py` (377 lines)

**This is the key differentiator.** The reasoning graph transforms the agent from a simple tool-caller into a system that chains findings into multi-step reasoning paths.

### Concept

```
Nodes = Capabilities/States     (e.g., "sqli_found", "db_access", "credentials")
Edges = Reasoning Steps          (e.g., "dump_database", "extract_creds")
Paths = Named Multi-Step Chains  (e.g., "SQLi → DB → Creds → Admin")
```

When a finding is confirmed via tools, the graph marks the corresponding node as `CONFIRMED` and activates downstream edges. The agent sees these activated paths in its system prompt and knows what to pursue next.

### Node States

```
UNKNOWN    → Not yet investigated
SUSPECTED  → Hypothesis, not confirmed
CONFIRMED  → Confirmed via investigation
EXPLOITED  → Successfully acted upon
FAILED     → Investigated and ruled out
```

### Context Injection

The graph generates prompt text that shows the LLM:
1. What has been confirmed
2. What edges are now available (with tool hints)
3. What named paths can be pursued
4. Priority ordering by impact

### Domain Example (Code Quality)

```
project_structure  ──complexity_scan──▶  high_complexity
                                              │
                                        class_analysis
                                              │
                                              ▼
                                          god_class  ──testability_check──▶  low_testability
```

---

## Hierarchical Planner

**File:** `planner.py` (402 lines)

The planner decomposes objectives into structured execution plans.

### Structure

```
TaskPlan
├── objective: "Analyze repository X"
├── TaskPhase: "Discovery"
│   ├── objective: "Map project structure"
│   ├── TaskStep: "Scan project tree" (tool_hint: file_scanner)
│   ├── TaskStep: "Parse dependencies" (tool_hint: dependency_scanner)
│   └── status: ACTIVE
├── TaskPhase: "Analysis"
│   ├── objective: "Deep quality metrics"
│   ├── TaskStep: "Complexity analysis" (tool_hint: complexity_analyzer)
│   └── status: PENDING
└── TaskPhase: "Impact Assessment"
    └── status: PENDING
```

### Plan Generation

1. **Template matching**: `detect_subject_type(profile)` → template key → `PLAN_TEMPLATES[key]`
2. **LLM refinement** (optional): Send objective + template to LLM for customization
3. **Fallback**: If no template matches, generate a generic 3-phase plan

### Plan Injection

The plan is serialized as markdown and injected into the system prompt:
```
## Task Plan: Analyze repository X

[>] **Phase 1: Discovery** [1/3]
  Objective: Map project structure
  [x] 1. Scan project tree (hint: file_scanner)
  [ ] 2. Parse dependencies (hint: dependency_scanner)
  [ ] 3. Detect framework (hint: framework_detector)

[ ] **Phase 2: Analysis** [0/3]
```

The LLM sees the active phase, completed steps, and pending work — guiding its next tool selection.

---

## Context Management

**File:** `context.py` (229 lines)

### The Problem
LLM context windows are finite. Long agent sessions can generate hundreds of messages with large tool outputs.

### 3-Level Strategy

1. **DomainProfile + TaskPlan** — Always in system prompt (never trimmed). This is the agent's persistent memory.
2. **Middle messages** — Tool results are compressed (heuristic summarization: first 300 chars + truncation notice)
3. **Recent window** — Last N messages kept intact for LLM coherence

### Atomic Groups
Critical: assistant messages with tool calls and their corresponding tool result messages form **atomic groups**. The trimmer never splits these — breaking them would cause LLM API validation errors.

```python
# This is ONE atomic group — never split:
{"role": "assistant", "content": [{"type": "tool_use", ...}]}
{"role": "tool", "content": "result..."}
{"role": "tool", "content": "result..."}
```

---

## Domain Profile & State

### DomainProfile (`domain_profile.py`)

Everything the agent knows about its subject. Auto-populated by extractors. Injected into every LLM call.

**Base fields:**
- `subject` — What's being analyzed
- `scope` — Allowed targets
- `hypotheses` — List of Hypothesis objects with confidence tracking
- `metadata` — Flexible dict for domain-specific data

**Key behaviors:**
- Duplicate hypothesis detection (same type + location)
- Confidence upgrading (new evidence can increase confidence)
- Serialization for session persistence

### State (`state.py`)

Session state container:
- `messages` — LLM conversation history
- `findings` — Pydantic-validated discoveries
- `profile` — DomainProfile instance
- `plan` — TaskPlan instance
- `enrich_fn` — Optional enrichment hook for findings

### Finding Validation (Pydantic)

```python
class Finding(BaseModel):
    title: str        # Validated: rejects generic titles like "issue"
    severity: str     # Normalized: "HIGH" → "high", "crit" → "critical"
    description: str
    evidence: str
    enrichment: dict  # Populated by enrich_fn hook
```

---

## Post-Processing Pipeline

After every tool execution, three systems run in sequence:

### 1. Extractors (`extractors.py`)

Parse tool results into structured DomainProfile data.

```
Tool Result (raw string) → EXTRACTORS[tool_name](profile, result, args) → Profile mutation
```

Example: nmap output → `profile.ports`, `profile.technologies`, `profile.hypotheses`

### 2. Reflection (`reflection.py`)

Generate strategic insight that guides the LLM's next decision.

```
REFLECTORS[tool_name](result, args, profile, lines) → Insight text injected as context
```

Also checks for untested hypotheses and confirmed findings to prompt escalation.

### 3. Error Recovery (`error_recovery.py`)

When `_is_failure()` returns True, pattern-match against known failure modes:

```
ERROR_PATTERNS[tool_name]["pattern_name"]["indicators"] → RecoveryStrategy
  → guidance: "Try with different scan type"
  → retry_tool: "alternative_tool"
  → give_up: False
```

---

## Tool System

**File:** `tools/__init__.py` (222 lines)

### ToolRegistry

```python
registry = ToolRegistry(allowed_targets=["target.com"])
registry.register("my_tool", handler=my_func, schema={...})
result = await registry.call("my_tool", {"arg": "value"})
schemas = registry.get_schemas()  # For LLM tool definitions
```

### Scope Checking
Tools can be restricted to allowed targets:
- Exact hostname match
- Subdomain matching (`sub.target.com` matches `target.com`)
- CIDR range matching (`10.0.0.5` matches `10.0.0.0/24`)

### Built-in Tool: `create_finding`
Always available. Allows the LLM to directly register findings:
```json
{"name": "create_finding", "arguments": {"title": "...", "severity": "high", ...}}
```

### Few-Shot Examples
Tool schemas include examples from `EXAMPLES` dict for improved LLM accuracy.

---

## Plugin Architecture

**File:** `plugins.py` (487 lines)

### Discovery
Plugins are Python packages in `~/.omnigent/plugins/`:

```
~/.omnigent/plugins/
  my_plugin/
    __init__.py      # PLUGIN_META dict or plugin.json
    tool.py          # Tool implementations
    extractor.py     # Custom extractors
    knowledge/       # Knowledge files
```

### Plugin Types
- **tool** — Adds functions + schemas to ToolRegistry
- **extractor** — Adds extractors to the pipeline
- **knowledge** — Adds markdown knowledge files

### Loading Sequence
1. Scan plugin dir for directories with `__init__.py`
2. Read metadata from `plugin.json` or `PLUGIN_META` dict
3. Import module, extract tools/extractors/knowledge
4. Register into core registries

---

## Session Persistence

**File:** `session.py` (277 lines)

Sessions auto-save to `~/.omnigent/sessions/` as JSON:
- Full conversation history
- All findings
- DomainProfile snapshot
- TaskPlan state
- Cost/token data

Supports: resume, list, delete, export (Markdown, JSON, HTML).

---

## Extension Model

Omnigent has two extension mechanisms:

### 1. Registry Population (Data-Driven)
For behavior expressible as data. Import → mutate the dict → done:

| Registry | Module | Key → Value |
|----------|--------|------------|
| `PLAN_TEMPLATES` | planner.py | template_key → phase list |
| `CHAINS` | chains.py | category → ChainStep list |
| `EXTRACTORS` | extractors.py | tool_name → callable |
| `REFLECTORS` | reflection.py | tool_name → callable |
| `ERROR_PATTERNS` | error_recovery.py | tool_name → pattern dict |
| `KNOWLEDGE_MAP` | knowledge_loader.py | key → file references |
| `EXAMPLES` | few_shot_examples.py | tool_name → ToolExample list |
| `TOOL_TIMEOUTS` | agent.py | tool_name → seconds |

### 2. Subclass Hooks (Object-Oriented)
For behavior requiring complex logic:

| Class | Override | Purpose |
|-------|----------|---------|
| `Agent` | `_is_failure()` | Domain-specific failure detection |
| `Agent` | `_extract_finding()` | Custom finding patterns |
| `Agent` | `_build_dynamic_system_prompt()` | Extra context sections |
| `Agent` | `_on_finding()` | Finding post-processing |
| `ReasoningGraph` | `_build_default_graph()` | Domain reasoning chains |
| `DomainProfile` | `to_prompt_summary()` | Custom LLM context format |

---

## Data Flow

### Single Tool Call Flow

```
1. LLM requests tool_use("nmap", {"target": "10.0.0.1"})
2. Agent checks loop detection (hash comparison)
3. Agent calls ToolRegistry.call("nmap", args) with adaptive timeout
4. ToolRegistry checks scope (is 10.0.0.1 in allowed targets?)
5. Tool function executes, returns raw result string
6. EXTRACTORS["nmap"](profile, result, args) → updates DomainProfile
7. Plan marks "nmap" step as complete (if matching tool_hint)
8. _is_failure() checks result for error patterns
9. If failure: inject_recovery_guidance() → append to result
10. REFLECTORS["nmap"](result, args, profile, lines) → strategic insight
11. Reflection + result added to conversation as tool message
12. ReasoningGraph.mark_discovered() → activate downstream edges
13. Next iteration: LLM sees updated profile + plan + graph + reflection
```

### Finding Flow

```
1. LLM text contains "[FINDING: HIGH] SQL Injection in login form"
2. _extract_finding() parses with regex
3. Finding created with Pydantic validation (title, severity)
4. state.enrich_fn(finding) → adds CWE, CVSS, etc. (domain hook)
5. state.add_finding(finding) → stored in session
6. reasoning_graph.mark_discovered("SQL Injection") → node CONFIRMED
7. format_chain_for_prompt("sqli") → escalation path injected
8. AgentEvent("finding") → UI renders the finding
```

---

## Design Decisions

### Why empty dicts instead of abstract classes?
Dicts are simpler to populate from any codebase. No inheritance boilerplate. Just import and assign. This matters because domain implementations may have very different codebases — a security agent, a code analysis tool, and a compliance checker shouldn't need to fit the same class structure.

### Why MD5 for loop detection?
Speed. Loop detection runs on every tool call. MD5 is fast and collision-resistant enough for 10-element deques. We only need "same or different", not cryptographic security.

### Why streaming everywhere?
Production agents run long tasks. Users need real-time feedback (text tokens, tool status, findings). Buffering entire responses creates terrible UX and wastes memory.

### Why Pydantic for Findings but dataclasses for everything else?
Findings come from LLM output — messy, unpredictable, needs validation. Pydantic normalizes severity ("HIGH" → "high"), rejects generic titles, auto-generates timestamps. Everything else is created by trusted code, so dataclasses are lighter.

### Why no LangChain / CrewAI / etc.?
Omnigent is lower-level than these frameworks. It's the engine, not the car. You could build a LangChain-compatible agent on top of Omnigent, but you can also build something completely different. No opinions on chains, no mandatory abstractions.

### Why httpx instead of provider SDKs?
Direct HTTP control. Provider SDKs add dependency weight, version conflicts, and abstraction layers that hide streaming behavior. With httpx, we control every byte of the stream.
