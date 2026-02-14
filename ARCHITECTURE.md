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

**File:** `agent.py` (1024 lines)

The core of Omnigent is a ReAct (Reason-Act-Observe) loop with several production hardening layers. The main loop is decomposed into **overridable step methods** — subclass any step to customize behavior without rewriting the loop.

### Iteration Cycle

Each iteration calls these overridable methods:
1. **`_do_context_management()`** — trim if messages exceed token threshold
2. **Build dynamic system prompt** — base prompt + DomainProfile + TaskPlan + ReasoningGraph + Knowledge
3. **`_do_llm_call()`** — stream LLM response via Router with task-based provider selection
4. **Handle text tokens** — emit to UI, parse findings with `re.finditer()` (multi-finding per response)
5. **Rate limiting** — per-iteration cap (`max_tool_calls_per_iteration`) and total cap (`max_total_tool_calls`)
6. **Human-in-the-loop** — tools with `requires_approval` flag trigger `approval_callback` before execution
7. **`_do_tool_execution()`** — execute approved tools in parallel with adaptive timeouts
8. **`_do_post_tool_processing()`** — extract → async reflect → error recover
9. **`_check_phase_advancement()`** — advance phases, generate **macro-reflection** via LLM at phase end
10. **`_check_termination()`** — plan complete, done indicators, or consecutive text-only responses
11. **Checkpoint** — if `checkpoint_interval > 0`, periodically save state for replay

### Safety Mechanisms

| Mechanism | How It Works |
|-----------|-------------|
| **Loop Detection** | MD5 hash of `{tool_name, args}`. Blocks on second repeated identical call. Recent history: 10 calls. |
| **Circuit Breaker** | Tracks repeated LLM errors. After 3 identical errors, stops the loop entirely. |
| **Rate Limiting** | Two-tier: per-iteration cap (default 20) truncates excessive calls; total cap (default 500) halts execution. |
| **Human-in-the-Loop** | Tools with `requires_approval` in schema trigger async `approval_callback` before execution. Denied calls return an error message to the LLM. |
| **Adaptive Timeouts** | Per-tool timeout via `DomainRegistry.tool_timeouts`. Default: 300s. |
| **Max Iterations** | Hard cap (default: 50). Prevents infinite loops on edge cases. |
| **Termination Detection** | Pattern matching on text for "task complete", "summary", etc. Also: 3+ consecutive text-only responses = done. |
| **Checkpoint/Replay** | If `checkpoint_interval > 0`, saves full state (messages, findings, profile, plan) every N iterations for mid-execution recovery. |

### Event System

The agent yields typed event subclasses of `AgentEvent` for the UI layer. Use `isinstance()` for type-safe handling or match on `event.type`:

```python
TextEvent(content="...")               # LLM text token
ToolStartEvent(tool_name, arguments)   # Tool execution starting
ToolEndEvent(tool_name, tool_result)   # Tool execution finished
FindingEvent(finding=Finding(...))     # New finding registered
PlanEvent(plan="...")                  # Plan created
PhaseCompleteEvent(phase_name, next)   # Phase advanced
UsageEvent(input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens)
ErrorEvent(message="...")              # Error occurred
DoneEvent()                            # Loop finished
AgentEvent("reflection", guidance=...) # Error recovery guidance
AgentEvent("paused")                   # Agent paused by user
```

All events inherit from `AgentEvent` and support `.type`, `.content`, `.tool_name`, `.finding` properties for backward compatibility.

---

## LLM Router

**File:** `router.py` (700 lines)

Multi-provider LLM routing with streaming, automatic fallback, task-based selection, and **extended thinking** support for Claude.

### Provider Architecture

New providers can be added by subclassing the `LLMProvider` abstract base class:

```python
class LLMProvider:
    """Abstract base for LLM providers."""
    async def stream(self, client, config, messages, tools, system) -> AsyncGenerator[StreamChunk, None]:
        raise NotImplementedError
```

Register custom providers via `LLMRouter.register_provider()` — no need to modify the router itself.

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

System prompts are always sent as **native system messages** (not injected via `[SYSTEM]:` prefix) — proper `system` parameter for Anthropic, `{"role": "system"}` for OpenAI-compatible APIs.

### Extended Thinking (Claude)

The `stream()` method accepts an optional `thinking_budget` parameter. When set, it enables Claude's extended thinking mode with the specified token budget, useful for complex planning and analysis tasks.

---

## Reasoning Graph

**File:** `reasoning_graph.py` (389 lines)

**This is the key differentiator.** The reasoning graph transforms the agent from a simple tool-caller into a system that chains findings into multi-step reasoning paths.

### Concept

```
Nodes = Capabilities/States     (e.g., "sqli_found", "db_access", "credentials")
Edges = Reasoning Steps          (e.g., "dump_database", "extract_creds")
Paths = Named Multi-Step Chains  (e.g., "SQLi → DB → Creds → Admin")
```

When a finding is confirmed via tools, the graph marks the corresponding node as `CONFIRMED` and activates downstream edges. Edges support a `requires_all` field — a list of prerequisite node names that must all be confirmed before the edge activates, enabling complex multi-prerequisite reasoning chains. The agent sees these activated paths in its system prompt and knows what to pursue next.

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

**File:** `planner.py` (544 lines)

The planner decomposes objectives into structured execution plans with skip conditions and macro-reflection.

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

### Skip Conditions

Phases can declare a `skip_condition` (e.g., `"Discovery complete"`). The planner evaluates these against completed phases and skips phases whose conditions are already satisfied.

### Macro-Reflection

When a phase completes, the planner generates a **macro-reflection** via LLM (`generate_phase_reflection()`). This summarizes what was accomplished, key findings, and strategic adjustments. The reflection is stored and injected into the next phase's context, ensuring continuity across phases.

---

## Context Management

**File:** `context.py` (358 lines)

### The Problem
LLM context windows are finite. Long agent sessions can generate hundreds of messages with large tool outputs.

### 3-Level Strategy

1. **DomainProfile + TaskPlan** — Always in system prompt (never trimmed). This is the agent's persistent memory.
2. **Middle messages** — Tool results are compressed (heuristic summarization: first 300 chars + truncation notice)
3. **Recent window** — Last N messages kept intact for LLM coherence

### Semantic Compression (LLM-based)

Beyond heuristic trimming, `semantic_compress_messages()` uses the LLM via the router to intelligently summarize old tool results and conversations while preserving key facts, findings, and decision points. This provides much better context retention than simple truncation.

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
- **Bounded `to_prompt_summary()`** — limits confirmed hypotheses (default 15) and untested hypotheses (default 10) with "... and N more" truncation to prevent token bloat
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

Generate strategic insight that guides the LLM's next decision. Supports both **sync and async reflectors** — async reflectors are awaited via `reflect_on_result_async()`, enabling reflectors that make LLM calls or I/O.

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

**File:** `tools/__init__.py` (309 lines)

### ToolRegistry

```python
registry = ToolRegistry(allowed_targets=["target.com"])
registry.register("my_tool", handler=my_func, schema={...})
result = await registry.call("my_tool", {"arg": "value"})
schemas = registry.get_schemas()  # For LLM tool definitions (cached, invalidated on register/unregister)
```

Schema output is **cached** after first call to `get_schemas()` and invalidated on `register()` / `unregister()`, avoiding repeated schema construction on every LLM call.

### Scope Checking
Tools can be restricted to allowed targets:
- Exact hostname match
- Subdomain matching (`sub.target.com` matches `target.com`)
- CIDR range matching (`10.0.0.5` matches `10.0.0.0/24`)

### Built-in Tools

**`create_finding`** — Always available. Allows the LLM to directly register individual findings:
```json
{"name": "create_finding", "arguments": {"title": "...", "severity": "high", ...}}
```

**`submit_analysis`** — Structured output tool for formal analysis submission. Accepts step summaries, findings array, conclusions, and recommendations. Provides structured outputs via tool use rather than free-form text.

### Few-Shot Examples
Tool schemas include examples from `EXAMPLES` dict for improved LLM accuracy.

---

## Plugin Architecture

**File:** `plugins.py` (556 lines)

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

### Strict Checksum Mode

`PluginManager(strict_checksums=True)` enables SHA-256 checksum verification. Plugins must include a `checksums.json` mapping filenames to expected hashes. In strict mode, plugins without checksums are rejected entirely.

### Loading Sequence
1. Scan plugin dir for directories with `__init__.py`
2. Read metadata from `plugin.json` or `PLUGIN_META` dict
3. **Verify checksums** (if strict mode enabled)
4. Import module, extract tools/extractors/knowledge
5. Register into core registries

---

## Session Persistence

**File:** `session.py` (546 lines)

Sessions auto-save to `~/.omnigent/sessions/` as JSON:
- Full conversation history
- All findings
- DomainProfile snapshot
- TaskPlan state
- Cost/token data

Supports: resume, list, delete, export (Markdown, JSON, HTML).

### Checkpoint/Replay

Mid-execution checkpointing enables recovery from crashes or long-running sessions:
- `save_checkpoint(state, iteration)` — saves full agent state at a specific iteration
- `restore_checkpoint(session_id)` — restores state to resume from a checkpoint
- `list_checkpoints()` / `delete_checkpoints()` — checkpoint management

Enable via `Agent(checkpoint_interval=5, session_manager=sm)` to auto-save every 5 iterations.

### Encryption

Session encryption uses `derive_key()` with a **random 16-byte salt** (generated per session) and PBKDF2 key derivation.

---

## Extension Model

Omnigent has two extension mechanisms:

### 1. DomainRegistry (Data-Driven)

All domain-specific behavior is centralised in a single injectable `DomainRegistry` dataclass (`registry.py`). Pass it to `Agent` — no global state mutations needed:

```python
from omnigent.registry import DomainRegistry

registry = DomainRegistry(
    plan_templates={...},
    chains={...},
    extractors={...},
    reflectors={...},
    error_patterns={...},
    examples={...},
    knowledge_map={...},
    tool_timeouts={...},
)
agent = Agent(registry=registry)
```

| Field | Key → Value |
|-------|------------|
| `plan_templates` | template_key → phase list |
| `chains` | category → ChainStep list |
| `extractors` | tool_name → callable |
| `reflectors` | tool_name → callable (sync or async) |
| `error_patterns` | tool_name → pattern dict |
| `knowledge_map` | key → file references |
| `examples` | tool_name → ToolExample list |
| `tool_timeouts` | tool_name → seconds |

Multiple agents can run with independent registries. For backward compatibility, `DomainRegistry.default()` reads the module-level dicts. Use `registry.merge(other)` to combine registries.

### 2. Subclass Hooks (Object-Oriented)
For behavior requiring complex logic:

| Class | Override | Purpose |
|-------|----------|---------|
| `Agent` | `_is_failure()` | Domain-specific failure detection |
| `Agent` | `_extract_finding()` | Custom finding patterns |
| `Agent` | `_build_dynamic_system_prompt()` | Extra context sections |
| `Agent` | `_on_finding()` | Finding post-processing |
| `Agent` | `_do_context_management()` | Custom context budget strategy |
| `Agent` | `_do_llm_call()` | Custom LLM interaction |
| `Agent` | `_do_tool_execution()` | Custom tool execution (e.g., sequential) |
| `Agent` | `_do_post_tool_processing()` | Custom post-processing pipeline |
| `Agent` | `_check_termination()` | Custom termination logic |
| `ReasoningGraph` | `_build_default_graph()` | Domain reasoning chains |
| `DomainProfile` | `to_prompt_summary()` | Custom LLM context format |
| `LLMProvider` | `stream()` | Add new LLM providers |

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

### Why a DomainRegistry dataclass?
Originally the framework used module-level empty dicts. This worked for single-agent scenarios, but leaked state across multiple Agent instances and test cases. The `DomainRegistry` dataclass centralises all registries into a single injectable object, enabling multi-agent scenarios with independent state. The dicts-as-data philosophy remains — fields are still plain dicts, just scoped per agent. `DomainRegistry.default()` preserves backward compatibility with module-level dicts.

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
