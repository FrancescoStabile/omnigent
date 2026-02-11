# Changelog

All notable changes to Omnigent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-02-11

### Added

- **Agent Loop** — Full ReAct architecture with streaming events, circuit breaker, loop detection, and adaptive timeouts
- **LLM Router** — Multi-provider routing (DeepSeek, Claude, OpenAI, Ollama) with streaming, fallback, and task-based selection
- **Reasoning Graph** — Directed graph for multi-step chain reasoning with node/edge/path registration and prompt injection
- **Hierarchical Planner** — TaskPlan → TaskPhase → TaskStep with LLM refinement and template system
- **Smart Context Management** — 3-level context trimming preserving atomic message groups
- **Domain Profile** — Structured agent memory with hypothesis tracking, serialization, and subclass support
- **State Management** — Pydantic-validated findings with severity normalization and enrichment hooks
- **Tool Registry** — Registration, scope checking, few-shot examples, and async execution
- **Extractor Pipeline** — Auto-parse tool results into DomainProfile via registry
- **Reflection Engine** — Strategic insight generation after each tool call
- **Error Recovery** — Pattern-matched failure detection with recovery strategies
- **Escalation Chains** — Finding-triggered multi-step escalation paths
- **Knowledge Loader** — Section-level markdown retrieval with token budgets
- **Few-Shot Examples** — Tool usage examples for improved LLM accuracy
- **Plugin System** — Filesystem-based plugin discovery and loading
- **Session Persistence** — Auto-save, resume, and multi-format export
- **Cost Tracker** — Per-provider, per-task-type token and cost tracking
- **Config System** — YAML + .env + ENV config loading with interactive setup
- **Structured Logging** — JSON logging with rotation
- **CodeLens Example** — Complete code quality analysis agent demonstrating all extension points
- **109 passing tests** covering all core components
