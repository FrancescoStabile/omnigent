"""
Omnigent — The Agent Loop (ReAct Architecture)

THE agent loop — ReAct architecture with structured intelligence.

Architecture:
1. Build DYNAMIC system prompt (base + profile + plan + knowledge + chains)
2. Call LLM (streaming, task-aware routing)
3. Execute tools if requested
4. Run extractors → update DomainProfile
5. Reflect on results → generate strategic insight
6. Inject escalation chains if finding confirmed
7. Self-correct on failures with error recovery guidance
8. Loop detection prevents infinite cycles
9. Adaptive timeouts per tool type
10. Repeat until plan complete or max iterations

Domain customisation points:
  - TOOL_TIMEOUTS dict: override with domain-specific tool timeouts
  - _is_failure(): override for domain-specific failure detection
  - _extract_finding(): override for domain-specific finding patterns
  - _build_dynamic_system_prompt(): override for extra context sections
  - The agent delegates to pluggable extractors, reflectors, chains, etc.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import deque
from pathlib import Path
from typing import AsyncGenerator, Any, Union

from omnigent.router import LLMRouter, Provider, TaskType
from omnigent.state import State, Finding
from omnigent.tools import ToolRegistry
from omnigent.error_recovery import inject_recovery_guidance
from omnigent.context import smart_trim_context, should_trim_context, estimate_tokens
from omnigent.extractors import run_extractor
from omnigent.planner import generate_plan, generate_plan_with_llm
from omnigent.reflection import reflect_on_result
from omnigent.chains import format_chain_for_prompt
from omnigent.knowledge_loader import get_relevant_knowledge
from omnigent.reasoning_graph import ReasoningGraph

logger = logging.getLogger("omnigent.agent")


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive Timeouts per tool type — override in domain implementation
# ═══════════════════════════════════════════════════════════════════════════

TOOL_TIMEOUTS: dict[str, int] = {
    # Populate in your domain implementation, e.g.:
    # "http": 30,
    # "read_file": 10,
    # "run_command": 120,
}


# ═══════════════════════════════════════════════════════════════════════════
# Agent Events (for CLI/UI streaming)
# ═══════════════════════════════════════════════════════════════════════════


class AgentEvent:
    """Event emitted by agent during execution.

    Event types:
      text          — LLM text token (content=str)
      tool_start    — Tool execution starting (tool_name, arguments)
      tool_end      — Tool execution finished (tool_name, tool_result)
      finding       — New finding registered (finding=Finding)
      plan_generated — Plan created (plan=str)
      phase_complete — Phase completed (phase_name, next_phase)
      reflection    — Recovery guidance after failure (tool_name, guidance)
      usage         — Token usage stats (input_tokens, output_tokens, ...)
      error         — Error occurred (message=str)
      paused        — Agent paused by user
      done          — Agent loop finished
    """

    def __init__(self, type: str, **data):
        self.type = type
        self.data = data

    @property
    def content(self) -> str:
        return self.data.get("content", "")

    @property
    def tool_name(self) -> str:
        return self.data.get("tool_name", "")

    @property
    def tool_result(self) -> str:
        return self.data.get("tool_result", "")

    @property
    def finding(self) -> Finding | None:
        return self.data.get("finding")


# ═══════════════════════════════════════════════════════════════════════════
# The Agent — ReAct loop with extractors, planner, reflection
# ═══════════════════════════════════════════════════════════════════════════


class Agent:
    """The Omnigent agent — ReAct architecture.

    Key architecture features:
    - DomainProfile: Structured memory of everything discovered
    - Extractors: Auto-parse tool results into DomainProfile
    - Planner: Hierarchical task plan guides LLM decisions
    - Reflection: Strategic analysis after each tool call
    - Chains: Escalation paths for confirmed findings
    - Knowledge: Context-aware reference injection
    - Loop detection: Prevents infinite tool call cycles
    - Smart _is_failure: Context-aware failure detection
    - Adaptive timeouts: Per-tool-type timeout values
    - Task-based routing: Best provider for each task type

    Customisation:
        Subclass and override:
        - _build_dynamic_system_prompt() for extra context
        - _is_failure() for domain-specific failure detection
        - _extract_finding() for domain-specific finding patterns
        - _on_finding() for domain-specific finding processing
    """

    def __init__(
        self,
        router: LLMRouter | None = None,
        tools: ToolRegistry | None = None,
        reasoning_graph: ReasoningGraph | None = None,
        max_iterations: int = 50,
        tool_timeout: int = 300,
        session_timeout: int = 3600,
    ):
        self.router = router or LLMRouter(primary=Provider.DEEPSEEK)
        self.tools = tools or ToolRegistry()
        self.max_iterations = max_iterations
        self.tool_timeout = tool_timeout
        self.session_timeout = session_timeout

        # State
        self.state = State()
        self.is_running = False

        # Reasoning graph for multi-stage chain reasoning
        self.reasoning_graph = reasoning_graph or ReasoningGraph()

        # Circuit breaker: Track repeated errors
        self.error_counter: dict[str, int] = {}
        self.max_repeated_errors = 3

        # Loop detection: Track recent tool calls
        self._recent_tool_hashes: deque[str] = deque(maxlen=10)
        self._loop_threshold = 1  # Block on first repeat

        # Load base system prompt
        self._base_system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load base system prompt from file."""
        prompt_file = Path(__file__).parent / "prompts" / "system.md"
        if prompt_file.exists():
            return prompt_file.read_text()
        return "You are an expert AI agent. Analyze the task and use tools methodically."

    def _build_dynamic_system_prompt(self) -> str:
        """Build dynamic system prompt with all context layers.

        Includes:
        1. Base system prompt
        2. DomainProfile summary (structured memory)
        3. TaskPlan status (what to do next)
        4. Reasoning graph — multi-stage escalation paths
        5. Escalation chains (for confirmed findings)
        6. Relevant knowledge (context-aware)

        Override in subclass to add domain-specific sections.
        """
        parts = [self._base_system_prompt]

        # DomainProfile summary
        profile_summary = self.state.profile.to_prompt_summary()
        if profile_summary and "No data" not in profile_summary:
            parts.append("\n\n---\n\n" + profile_summary)

        # TaskPlan status
        if self.state.plan and self.state.plan.objective:
            plan_summary = self.state.plan.to_prompt_summary()
            parts.append("\n\n---\n\n" + plan_summary)

        # Reasoning graph — multi-stage escalation guidance
        graph_ctx = self.reasoning_graph.to_prompt_context()
        if graph_ctx:
            parts.append("\n\n---\n\n" + graph_ctx)

        # Escalation chains for confirmed findings
        confirmed = self.state.profile.get_confirmed()
        for hyp in confirmed[:3]:  # Limit to avoid token bloat
            chain_text = format_chain_for_prompt(hyp.hypothesis_type)
            if chain_text:
                parts.append("\n" + chain_text)

        # Context-aware knowledge injection
        current_phase = ""
        if self.state.plan:
            cp = self.state.plan.current_phase()
            if cp:
                current_phase = cp.name.lower()

        knowledge = get_relevant_knowledge(self.state.profile, current_phase)
        if knowledge:
            parts.append("\n\n---\n\n## Relevant Knowledge\n\n" + knowledge)

        return "\n".join(parts)

    def _compute_tool_hash(self, name: str, args: dict) -> str:
        """Compute hash for a tool call to detect loops."""
        key = json.dumps({"name": name, "args": args}, sort_keys=True)
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _detect_loop(self, name: str, args: dict) -> bool:
        """Check if this tool call is a repeated loop."""
        tool_hash = self._compute_tool_hash(name, args)
        count = sum(1 for h in self._recent_tool_hashes if h == tool_hash)
        self._recent_tool_hashes.append(tool_hash)
        return count >= self._loop_threshold

    def _get_tool_timeout(self, tool_name: str) -> int:
        """Get adaptive timeout for a tool."""
        return TOOL_TIMEOUTS.get(tool_name, self.tool_timeout)

    async def run(self, user_input: str) -> AsyncGenerator[AgentEvent, None]:
        """Run agent loop with user input.

        Yields AgentEvent objects for the CLI/UI to render.
        """
        self.is_running = True

        # Add user message
        self.state.add_message("user", user_input)

        # Generate task plan if this is a new objective
        if not self.state.plan.objective or self.state.plan.is_complete():
            try:
                self.state.plan = await generate_plan_with_llm(
                    user_input, self.state.profile, self.router
                )
            except Exception:
                self.state.plan = generate_plan(user_input, self.state.profile)
            yield AgentEvent("plan_generated", plan=self.state.plan.to_prompt_summary())

        try:
            for iteration in range(self.max_iterations):
                if not self.is_running:
                    yield AgentEvent("paused")
                    break

                # Smart context trimming
                needs_trim, tokens_before = should_trim_context(
                    self.state.messages, threshold=25, max_tokens=80000
                )
                if needs_trim:
                    msgs_before = len(self.state.messages)
                    self.state.messages = smart_trim_context(
                        self.state.messages,
                        max_tokens=80000,
                        recent_window=12,
                    )
                    tokens_after = sum(
                        estimate_tokens(msg.get("content", ""))
                        for msg in self.state.messages
                    )
                    logger.info(
                        f"Context trimmed: {msgs_before}→{len(self.state.messages)} msgs, "
                        f"~{tokens_before:,}→{tokens_after:,} tokens"
                    )

                # Build dynamic system prompt
                system_prompt = self._build_dynamic_system_prompt()

                # Call LLM
                text_buffer = ""
                tool_calls = []
                usage_stats = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_creation_tokens": 0,
                }

                try:
                    async for chunk in self.router.stream(
                        messages=self.state.messages,
                        tools=self.tools.get_schemas(),
                        system=system_prompt,
                        task_type=TaskType.TOOL_USE,
                    ):
                        if chunk.content:
                            text_buffer += chunk.content
                            yield AgentEvent("text", content=chunk.content)

                        if chunk.tool_call:
                            tool_calls.append(chunk.tool_call)

                        if chunk.input_tokens or chunk.cache_read_tokens:
                            usage_stats["input_tokens"] += chunk.input_tokens
                            usage_stats["output_tokens"] += chunk.output_tokens
                            usage_stats["cache_read_tokens"] += chunk.cache_read_tokens
                            usage_stats["cache_creation_tokens"] += chunk.cache_creation_tokens

                        if chunk.done:
                            break

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"LLM streaming error: {error_msg}", exc_info=True)

                    self.error_counter[error_msg] = self.error_counter.get(error_msg, 0) + 1

                    if self.error_counter[error_msg] >= self.max_repeated_errors:
                        logger.error(f"Circuit breaker triggered: Same error {self.max_repeated_errors} times")
                        yield AgentEvent(
                            "error",
                            message=(
                                f"STOPPING: Repeated LLM error detected.\n\n"
                                f"Error: {error_msg}\n\n"
                                f"Check your API keys and configuration."
                            ),
                        )
                        break

                    yield AgentEvent("error", message=f"LLM API error: {error_msg}")
                    continue

                # Emit usage stats
                if usage_stats["input_tokens"] or usage_stats["cache_read_tokens"]:
                    yield AgentEvent("usage", **usage_stats)

                # Check for findings in text
                if text_buffer:
                    finding = self._extract_finding(text_buffer)
                    if finding:
                        self.state.add_finding(finding)
                        self.reasoning_graph.mark_discovered(finding.title)
                        yield AgentEvent("finding", finding=finding)

                # Process tool calls
                if tool_calls:
                    # Add assistant message with tool calls
                    if text_buffer:
                        content = [{"type": "text", "text": text_buffer}]
                        content.extend([
                            {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]}
                            for tc in tool_calls
                        ])
                        self.state.add_message("assistant", content)
                    else:
                        self.state.add_message("assistant", [
                            {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]}
                            for tc in tool_calls
                        ])

                    # Execute tools
                    for tc in tool_calls:
                        # Loop detection
                        if self._detect_loop(tc["name"], tc["arguments"]):
                            loop_msg = (
                                f"LOOP DETECTED: '{tc['name']}' with same arguments was called recently. "
                                "Try a DIFFERENT approach, tool, or parameter."
                            )
                            yield AgentEvent("tool_start", tool_name=tc["name"], arguments=tc["arguments"])
                            yield AgentEvent("tool_end", tool_name=tc["name"], tool_result=loop_msg)
                            self.state.add_message("tool", {
                                "tool_call_id": tc["id"],
                                "content": loop_msg,
                            })
                            continue

                        yield AgentEvent("tool_start", tool_name=tc["name"], arguments=tc["arguments"])

                        # ── CREATE_FINDING: intercept and register ──
                        if tc["name"] == "create_finding":
                            finding = self._handle_create_finding(tc)
                            result = json.dumps({
                                "registered": True,
                                "title": finding.title,
                                "severity": finding.severity,
                            })
                            yield AgentEvent("finding", finding=finding)
                            yield AgentEvent("tool_end", tool_name=tc["name"], tool_result=result)
                            self.state.add_message("tool", {
                                "tool_call_id": tc["id"],
                                "content": result,
                            })
                            continue

                        # Execute with adaptive timeout
                        timeout = self._get_tool_timeout(tc["name"])
                        try:
                            result = await asyncio.wait_for(
                                self.tools.call(tc["name"], tc["arguments"]),
                                timeout=timeout,
                            )
                        except asyncio.TimeoutError:
                            logger.error(f"Tool timeout: {tc['name']} exceeded {timeout}s")
                            result = f"Error: Tool '{tc['name']}' timed out after {timeout} seconds. Try with smaller scope or different approach."
                        except Exception as e:
                            logger.error(f"Tool execution error: {tc['name']}: {e}", exc_info=True)
                            result = f"Error executing tool: {str(e)}"

                        yield AgentEvent("tool_end", tool_name=tc["name"], tool_result=result)

                        # ── EXTRACTOR: Update DomainProfile ──
                        try:
                            run_extractor(tc["name"], self.state.profile, result, tc.get("arguments", {}))
                        except Exception as e:
                            logger.warning(f"Extractor failed for {tc['name']}: {e}")

                        # ── PLANNER: Mark step complete ──
                        if self.state.plan and self.state.plan.objective:
                            summary = result[:200] if isinstance(result, str) else str(result)[:200]
                            failed = self._is_failure(tc["name"], result)
                            self.state.plan.mark_step_complete(tc["name"], summary, is_failure=failed)

                        # Build tool result content
                        tool_result_content = self._truncate_tool_result(tc["name"], result)

                        if isinstance(tool_result_content, dict):
                            tool_result_content = json.dumps(tool_result_content, indent=2)

                        # ── REFLECTION: Strategic insight ──
                        reflection_text = ""
                        try:
                            reflection_text = reflect_on_result(
                                tc["name"], tc["arguments"], result, self.state.profile
                            )
                        except Exception as e:
                            logger.warning(f"Reflection failed: {e}")

                        # Check for failure and inject recovery guidance
                        if self._is_failure(tc["name"], result):
                            guidance = inject_recovery_guidance(tc["name"], result)
                            tool_result_content = f"{tool_result_content}\n\n{guidance}"
                            yield AgentEvent("reflection", tool_name=tc["name"], guidance=guidance)

                        # Append reflection
                        if reflection_text:
                            tool_result_content = f"{tool_result_content}\n\n---\n**Reflection:**\n{reflection_text}"

                        # Add tool message
                        self.state.add_message("tool", {
                            "tool_call_id": tc["id"],
                            "content": tool_result_content,
                        })

                    # Check if plan phase should advance
                    if self.state.plan and self.state.plan.objective:
                        current = self.state.plan.current_phase()
                        if current:
                            all_done = all(
                                s.status.value in ("complete", "skipped")
                                for s in current.steps
                            )
                            if all_done:
                                completed_name = current.name
                                self.state.plan.advance_phase()
                                next_phase = self.state.plan.current_phase()
                                next_name = next_phase.name if next_phase else ""
                                yield AgentEvent(
                                    "phase_complete",
                                    phase_name=completed_name,
                                    next_phase=next_name,
                                )

                    continue

                # No tool calls — text-only response
                if text_buffer and not tool_calls:
                    self.state.add_message("assistant", text_buffer)

                    # ── TERMINATION DETECTION ──
                    done_indicators = self._get_done_indicators()
                    text_lower = text_buffer.lower()
                    is_done = any(indicator in text_lower for indicator in done_indicators)

                    plan_done = self.state.plan and self.state.plan.is_complete()

                    # Terminate after 3+ consecutive text-only responses
                    consecutive_text = 0
                    for msg in reversed(self.state.messages):
                        if isinstance(msg.get("content"), str) and msg.get("role") == "assistant":
                            consecutive_text += 1
                        else:
                            break

                    if is_done or plan_done or consecutive_text >= 3:
                        break
                    else:
                        continue

                # Safety: empty response
                if not text_buffer and not tool_calls:
                    yield AgentEvent("error", message="LLM returned empty response")
                    break

            yield AgentEvent("done")

        finally:
            self.is_running = False

    # ═══════════════════════════════════════════════════════════════════
    # Domain-overrideable hooks
    # ═══════════════════════════════════════════════════════════════════

    def _get_done_indicators(self) -> list[str]:
        """Return text patterns that signal task completion.

        Override in domain subclass for domain-specific indicators.
        """
        return [
            "i have completed",
            "task complete",
            "no further",
            "that's all",
            "finished the",
            "no more",
            "all phases",
            "is done",
            "is complete",
            "completed the",
            "completed my",
            "here is a summary",
            "here's a summary",
            "in summary",
            "to summarize",
            "wrapping up",
        ]

    def _is_failure(self, tool_name: str, result: str) -> bool:
        """Context-aware failure detection.

        Override in domain subclass for tool-specific failure patterns.
        Base implementation catches generic errors.
        """
        if not isinstance(result, str):
            result = str(result)
        result_lower = result.lower()

        failure_indicators = [
            "error executing",
            "command failed",
            "timed out",
            "command not found",
            "permission denied",
            "connection refused",
        ]
        return any(ind in result_lower for ind in failure_indicators)

    def _extract_finding(self, text: str) -> Finding | None:
        """Extract finding from agent text.

        Looks for pattern: [FINDING: SEVERITY] Title
        Override for custom finding patterns.
        """
        match = re.search(r'\[FINDING:\s*(\w+)\]\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if not match:
            return None

        severity = match.group(1).lower()
        title = match.group(2).strip()

        description = ""
        evidence = ""

        desc_match = re.search(r'\*\*Description\*\*:\s*(.+?)(?:\n\*\*|\n\n|$)', text, re.DOTALL)
        if desc_match:
            description = desc_match.group(1).strip()

        ev_match = re.search(r'\*\*Evidence\*\*:\s*(.+?)(?:\n\*\*|\n\n|$)', text, re.DOTALL)
        if ev_match:
            evidence = ev_match.group(1).strip()

        title = self._clean_markup(title)
        description = self._clean_markup(description)
        evidence = self._clean_markup(evidence)

        # Hook for domain-specific finding processing
        self._on_finding(title, severity, description, evidence)

        return Finding(
            title=title,
            severity=severity,
            description=description or title,
            evidence=evidence,
        )

    def _on_finding(self, title: str, severity: str, description: str, evidence: str):
        """Hook called when a finding is extracted. Override for domain logic.

        Example (security domain): update TargetProfile with VulnHypothesis.
        """
        pass

    def _handle_create_finding(self, tc: dict) -> Finding:
        """Handle the create_finding tool call."""
        args = tc.get("arguments", {})
        finding = Finding(
            title=args.get("title", "Untitled"),
            severity=args.get("severity", "info"),
            description=args.get("description", ""),
            evidence=args.get("evidence", ""),
        )
        self.state.add_finding(finding)
        self.reasoning_graph.mark_discovered(finding.title)
        return finding

    def _clean_markup(self, text: str) -> str:
        """Remove Rich markup tags from text."""
        if not text:
            return text
        cleaned = re.sub(r'\[/?(?:[a-z_]+)?(?:\s*#?[0-9a-fA-F]{6})?\s*[^\]]*?\]', '', text)
        cleaned = re.sub(r'\*\*|\__', '', cleaned)
        return cleaned.strip()

    def _truncate_tool_result(self, tool_name: str, result: Any, max_chars: int = 10000) -> str:
        """Intelligently truncate tool results to prevent context overflow."""
        if isinstance(result, dict):
            result_str = json.dumps(result)
        elif isinstance(result, str):
            result_str = result
        else:
            result_str = str(result)

        if len(result_str) <= max_chars:
            return result_str

        logger.info(f"Truncating {tool_name} result: {len(result_str)} chars → {max_chars} chars")

        # Generic truncation: preserve head and tail
        head_size = max_chars // 2
        tail_size = max_chars - head_size - 100
        head = result_str[:head_size]
        tail = result_str[-tail_size:]
        return f"{head}\n\n... [TRUNCATED {len(result_str) - max_chars} characters] ...\n\n{tail}"

    def pause(self):
        """Pause agent execution."""
        self.is_running = False

    def reset(self):
        """Reset agent state."""
        self.state.clear()
        self._recent_tool_hashes.clear()
        self.error_counter.clear()

    async def close(self):
        """Cleanup all resources — router, tools, HTTP client."""
        errors = []
        try:
            await self.tools.close()
        except Exception as e:
            errors.append(f"tools: {e}")
        try:
            await self.router.close()
        except Exception as e:
            errors.append(f"router: {e}")
        if errors:
            logger.debug(f"Cleanup warnings: {'; '.join(errors)}")


# ═══════════════════════════════════════════════════════════════════════════
# Convenience
# ═══════════════════════════════════════════════════════════════════════════


async def create_agent(**kwargs) -> Agent:
    """Create and initialize agent."""
    return Agent(**kwargs)
