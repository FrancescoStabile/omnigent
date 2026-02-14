"""Omnigent — Universal Autonomous Agent Framework.

Extracted from NumaSec v3.2.1 architecture. Domain-agnostic scaffold
for building specialized AI agents with:

- ReAct loop with streaming events
- Multi-provider LLM routing (DeepSeek, Claude, OpenAI, Ollama)
- Reasoning graph for multi-step chain reasoning
- Hierarchical task planner with LLM refinement
- Smart context window management
- Pattern-matched error recovery
- Plugin system for extensibility
- MCP protocol integration
- Session persistence
- Cost tracking
"""

__version__ = "0.1.0"
__author__ = "Francesco Stabile"
__description__ = "Universal autonomous agent framework with ReAct loop, reasoning graph, and MCP integration."

from omnigent.config import Config, ensure_config, load_config

__all__ = ["load_config", "ensure_config", "Config"]
