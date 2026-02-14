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
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from omnigent.chains import format_chain_for_prompt
from omnigent.context import estimate_tokens, should_trim_context, smart_trim_context
from omnigent.error_recovery import inject_recovery_guidance
from omnigent.extractors import run_extractor
from omnigent.knowledge_loader import get_relevant_knowledge
from omnigent.planner import generate_phase_reflection, generate_plan, generate_plan_with_llm
from omnigent.reasoning_graph import ReasoningGraph
from omnigent.reflection import reflect_on_result, reflect_on_result_async
from omnigent.registry import DomainRegistry
from omnigent.router import LLMRouter, Provider, TaskType
from omnigent.state import Finding, State
from omnigent.tools import CREATE_FINDING_SCHEMA, SUBMIT_ANALYSIS_SCHEMA, ToolRegistry

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

    For type-safe event handling, use isinstance() with the typed subclasses
    (TextEvent, ToolStartEvent, etc.) or match on event.type string.
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

    def __repr__(self) -> str:
        return f"AgentEvent({self.type!r}, {self.data})"


class TextEvent(AgentEvent):
    """LLM text token."""
    def __init__(self, content: str):
        super().__init__("text", content=content)


class ToolStartEvent(AgentEvent):
    """Tool execution starting."""
    def __init__(self, tool_name: str, arguments: dict):
        super().__init__("tool_start", tool_name=tool_name, arguments=arguments)


class ToolEndEvent(AgentEvent):
    """Tool execution finished."""
    def __init__(self, tool_name: str, tool_result: str):
        super().__init__("tool_end", tool_name=tool_name, tool_result=tool_result)


class FindingEvent(AgentEvent):
    """New finding registered."""
    def __init__(self, finding: Finding):
        super().__init__("finding", finding=finding)


class PlanEvent(AgentEvent):
    """Plan created or updated."""
    def __init__(self, plan: str):
        super().__init__("plan_generated", plan=plan)


class PhaseCompleteEvent(AgentEvent):
    """Phase completed."""
    def __init__(self, phase_name: str, next_phase: str):
        super().__init__("phase_complete", phase_name=phase_name, next_phase=next_phase)


class UsageEvent(AgentEvent):
    """Token usage stats."""
    def __init__(self, input_tokens: int = 0, output_tokens: int = 0,
                 cache_read_tokens: int = 0, cache_creation_tokens: int = 0):
        super().__init__("usage", input_tokens=input_tokens, output_tokens=output_tokens,
                         cache_read_tokens=cache_read_tokens, cache_creation_tokens=cache_creation_tokens)


class ErrorEvent(AgentEvent):
    """Error occurred."""
    def __init__(self, message: str):
        super().__init__("error", message=message)


class DoneEvent(AgentEvent):
    """Agent loop finished."""
    def __init__(self):
        super().__init__("done")


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
        registry: DomainRegistry | None = None,
        max_tool_calls_per_iteration: int = 20,
        max_total_tool_calls: int = 500,
        approval_callback: Any | None = None,
        checkpoint_interval: int = 0,
        session_manager: Any | None = None,
    ):
        self.router = router or LLMRouter(primary=Provider.DEEPSEEK)
        self.tools = tools or ToolRegistry()
        self.registry = registry or DomainRegistry.default()

        # Rate limiting
        self.max_tool_calls_per_iteration = max_tool_calls_per_iteration
        self.max_total_tool_calls = max_total_tool_calls
        self._total_tool_calls = 0

        # Human-in-the-loop: async callback that returns True to approve
        # Signature: async def callback(tool_name: str, args: dict) -> bool
        self._approval_callback = approval_callback

        # Checkpoint/replay: save state every N iterations (0 = disabled)
        self._checkpoint_interval = checkpoint_interval
        self._session_manager = session_manager

        # Always register create_finding and submit_analysis so LLM sees them
        if "create_finding" not in self.tools.tools:
            async def _noop_create_finding(**kwargs: str) -> str:
                return '{"registered": true}'  # Never actually called; agent intercepts
            self.tools.register("create_finding", _noop_create_finding, CREATE_FINDING_SCHEMA)

        if "submit_analysis" not in self.tools.tools:
            async def _noop_submit_analysis(**kwargs: str) -> str:
                return '{"registered": true}'  # Never actually called; agent intercepts
            self.tools.register("submit_analysis", _noop_submit_analysis, SUBMIT_ANALYSIS_SCHEMA)

        self.max_iterations = max_iterations
        self.tool_timeout = tool_timeout
        self.session_timeout = session_timeout

        # State
        self.state = State()
        self.is_running = False

        # Reasoning graph for multi-stage chain reasoning
        self.reasoning_graph = reasoning_graph or ReasoningGraph()
        self._pending_graph_paths: list = []  # Newly activated paths for prompt injection

        # Circuit breaker: Track repeated errors
        self.error_counter: dict[str, int] = {}
        self.max_repeated_errors = 3

        # Loop detection: Track recent tool calls
        self._recent_tool_hashes: deque[str] = deque(maxlen=10)
        self._loop_threshold = 2  # Allow one retry, block on second repeat

        # Phase reflection storage (macro-reflection at end of phase)
        self._last_phase_reflection: str = ""

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

        # Graph-activated path suggestions for planner coordination
        if self._pending_graph_paths:
            path_lines = ["\n\n---\n\n## Newly Activated Reasoning Paths"]
            path_lines.append("The following escalation paths are now available based on confirmed findings:")
            for path in self._pending_graph_paths[:3]:
                path_lines.append(f"- **{path.name}** ({path.impact} impact): {path.description}")
            path_lines.append("\nConsider adjusting your approach to pursue these paths.")
            parts.append("\n".join(path_lines))

        # Escalation chains for confirmed findings
        confirmed = self.state.profile.get_confirmed()
        for hyp in confirmed[:3]:  # Limit to avoid token bloat
            chain_text = format_chain_for_prompt(hyp.hypothesis_type)
            if chain_text:
                parts.append("\n" + chain_text)

        # Phase reflection (macro-reflection from previous phase)
        if self._last_phase_reflection:
            parts.append(
                "\n\n---\n\n## Previous Phase Reflection\n\n"
                + self._last_phase_reflection
            )

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
        return self.registry.tool_timeouts.get(tool_name, self.tool_timeout)

    # ═══════════════════════════════════════════════════════════════════
    # Overridable Step Methods
    # ═══════════════════════════════════════════════════════════════════

    def _do_context_management(self) -> None:
        """Step 1: Trim context if needed. Override for custom budget strategy."""
        needs_trim, tokens_before = should_trim_context(
            self.state.messages, threshold=25, max_tokens=80000
        )
        if needs_trim:
            msgs_before = len(self.state.messages)
            self.state.messages = smart_trim_context(
                self.state.messages, max_tokens=80000, recent_window=12,
            )
            tokens_after = sum(
                estimate_tokens(msg.get("content", ""))
                for msg in self.state.messages
            )
            logger.info(
                f"Context trimmed: {msgs_before}→{len(self.state.messages)} msgs, "
                f"~{tokens_before:,}→{tokens_after:,} tokens"
            )

    async def _do_llm_call(
        self, system_prompt: str
    ) -> AsyncGenerator[AgentEvent, None]:
        """Step 2: Stream LLM response. Override for custom LLM interaction.

        Yields AgentEvent("text", ...) for streaming tokens.
        Sets self._last_text_buffer, self._last_tool_calls, self._last_usage.
        """
        self._last_text_buffer = ""
        self._last_tool_calls = []
        self._last_usage = {
            "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "cache_creation_tokens": 0,
        }

        async for chunk in self.router.stream(
            messages=self.state.messages,
            tools=self.tools.get_schemas(),
            system=system_prompt,
            task_type=TaskType.TOOL_USE,
        ):
            if chunk.content:
                self._last_text_buffer += chunk.content
                yield AgentEvent("text", content=chunk.content)
            if chunk.tool_call:
                self._last_tool_calls.append(chunk.tool_call)
            if chunk.input_tokens or chunk.cache_read_tokens:
                self._last_usage["input_tokens"] += chunk.input_tokens
                self._last_usage["output_tokens"] += chunk.output_tokens
                self._last_usage["cache_read_tokens"] += chunk.cache_read_tokens
                self._last_usage["cache_creation_tokens"] += chunk.cache_creation_tokens
            if chunk.done:
                break

    async def _do_tool_execution(self, tool_calls: list[dict]) -> list[tuple[dict, str]]:
        """Step 3: Execute tool calls in parallel. Override for custom execution.

        Returns list of (tool_call_dict, result_string) tuples.
        """
        async def _exec_one(tc: dict) -> tuple[dict, str]:
            timeout = self._get_tool_timeout(tc["name"])
            try:
                r = await asyncio.wait_for(
                    self.tools.call(tc["name"], tc["arguments"]),
                    timeout=timeout,
                )
            except TimeoutError:
                logger.error(f"Tool timeout: {tc['name']} exceeded {timeout}s")
                r = (
                    f"Error: Tool '{tc['name']}' timed out after {timeout} seconds. "
                    "Try with smaller scope or different approach."
                )
            except Exception as e:
                logger.error(f"Tool execution error: {tc['name']}: {e}", exc_info=True)
                r = f"Error executing tool: {str(e)}"
            return tc, r

        raw_results = await asyncio.gather(
            *[_exec_one(tc) for tc in tool_calls],
            return_exceptions=True,
        )

        results: list[tuple[dict, str]] = []
        for item in raw_results:
            if isinstance(item, BaseException):
                logger.error(f"Unexpected error in parallel tool execution: {item}")
                continue
            results.append(item)
        return results

    async def _do_post_tool_processing(
        self, tc: dict, result: str
    ) -> tuple[str, str | None, str | None]:
        """Step 4: Process a single tool result. Override for custom post-processing.

        Args:
            tc: Tool call dict {id, name, arguments}
            result: Raw tool result string

        Returns:
            (processed_content, reflection_text, recovery_guidance)
        """
        # Extractor
        try:
            run_extractor(tc["name"], self.state.profile, result, tc.get("arguments", {}))
        except Exception as e:
            logger.warning(f"Extractor failed for {tc['name']}: {e}")

        # Planner update
        if self.state.plan and self.state.plan.objective:
            summary = result[:200] if isinstance(result, str) else str(result)[:200]
            failed = self._is_failure(tc["name"], result)
            self.state.plan.mark_step_complete(tc["name"], summary, is_failure=failed)
            if failed:
                self.state.plan.record_failure(tc["name"])
            else:
                self.state.plan.reset_failure_count()

        # Truncate + sanitize
        tool_result_content = self._truncate_tool_result(tc["name"], result)
        tool_result_content = self._sanitize_tool_output(tc["name"], tool_result_content)
        if isinstance(tool_result_content, dict):
            tool_result_content = json.dumps(tool_result_content, indent=2)

        # Async reflection (supports both sync and async reflectors)
        reflection_text = None
        try:
            r = await reflect_on_result_async(tc["name"], tc["arguments"], result, self.state.profile)
            if r:
                reflection_text = r
        except Exception as e:
            logger.warning(f"Reflection failed: {e}")

        # Recovery guidance
        guidance = None
        if self._is_failure(tc["name"], result):
            guidance = inject_recovery_guidance(tc["name"], result)

        return tool_result_content, reflection_text, guidance

    def _check_termination(self, text_buffer: str) -> bool:
        """Step 5: Check if agent should stop. Override for custom termination logic.

        Returns True if agent should terminate.
        """
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

        return is_done or plan_done or consecutive_text >= 3

    async def _check_phase_advancement(self) -> tuple[str, str] | None:
        """Check if plan phase should advance. Returns (completed, next) or None.

        When a phase completes, generates a macro-reflection via LLM
        and stores it for injection into the next phase's context.
        """
        if not (self.state.plan and self.state.plan.objective):
            return None
        current = self.state.plan.current_phase()
        if not current:
            return None
        all_done = all(
            s.status.value in ("complete", "skipped") for s in current.steps
        )
        if all_done:
            completed_name = current.name

            # Generate macro-reflection for the completed phase
            try:
                reflection = await generate_phase_reflection(
                    current, self.state.profile, self.router,
                )
                if reflection:
                    self._last_phase_reflection = reflection
            except Exception as e:
                logger.debug(f"Phase reflection generation failed: {e}")

            self.state.plan.advance_phase()
            next_phase = self.state.plan.current_phase()
            return completed_name, next_phase.name if next_phase else ""
        return None

    # ═══════════════════════════════════════════════════════════════════
    # Main Loop — orchestrates the overridable steps
    # ═══════════════════════════════════════════════════════════════════

    async def run(self, user_input: str) -> AsyncGenerator[AgentEvent, None]:
        """Run agent loop with user input.

        Yields AgentEvent objects for the CLI/UI to render.
        The loop calls overridable step methods — subclass those for
        custom behaviour without rewriting the entire loop.
        """
        self.is_running = True
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

                # Step 1: Context management
                self._do_context_management()

                # Step 2: Build prompt + LLM call
                system_prompt = self._build_dynamic_system_prompt()
                self._pending_graph_paths.clear()

                try:
                    async for event in self._do_llm_call(system_prompt):
                        yield event
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"LLM streaming error: {error_msg}", exc_info=True)
                    self.error_counter[error_msg] = self.error_counter.get(error_msg, 0) + 1
                    if self.error_counter[error_msg] >= self.max_repeated_errors:
                        logger.error(f"Circuit breaker triggered: Same error {self.max_repeated_errors} times")
                        yield AgentEvent("error", message=(
                            f"STOPPING: Repeated LLM error detected.\n\n"
                            f"Error: {error_msg}\n\n"
                            f"Check your API keys and configuration."
                        ))
                        break
                    yield AgentEvent("error", message=f"LLM API error: {error_msg}")
                    continue

                text_buffer = self._last_text_buffer
                tool_calls = self._last_tool_calls
                usage_stats = self._last_usage

                # Emit usage stats
                if usage_stats["input_tokens"] or usage_stats["cache_read_tokens"]:
                    yield AgentEvent("usage", **usage_stats)

                # Check for findings in text (supports multiple per response)
                if text_buffer:
                    for finding in self._extract_all_findings(text_buffer):
                        self.state.add_finding(finding)
                        activated_paths = self.reasoning_graph.mark_discovered(finding.title)
                        if activated_paths:
                            self._pending_graph_paths.extend(activated_paths[:3])
                        yield AgentEvent("finding", finding=finding)

                # Step 3-4: Process tool calls
                if tool_calls:
                    # Rate limiting: per-iteration cap
                    if len(tool_calls) > self.max_tool_calls_per_iteration:
                        logger.warning(
                            f"Rate limit: {len(tool_calls)} tool calls in one iteration "
                            f"(max {self.max_tool_calls_per_iteration}). Truncating."
                        )
                        tool_calls = tool_calls[:self.max_tool_calls_per_iteration]

                    # Rate limiting: total cap
                    if self._total_tool_calls >= self.max_total_tool_calls:
                        logger.error(f"Rate limit: {self._total_tool_calls} total tool calls reached max ({self.max_total_tool_calls})")
                        yield AgentEvent("error", message=f"Tool call limit reached ({self.max_total_tool_calls} total calls)")
                        break

                    # Add assistant message with tool calls
                    content_items = []
                    if text_buffer:
                        content_items.append({"type": "text", "text": text_buffer})
                    content_items.extend([
                        {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]}
                        for tc in tool_calls
                    ])
                    self.state.add_message("assistant", content_items)

                    # Separate intercepted vs executable
                    intercepted: list[tuple[str, dict]] = []
                    executable: list[dict] = []
                    for tc in tool_calls:
                        if self._detect_loop(tc["name"], tc["arguments"]):
                            intercepted.append(("loop", tc))
                        elif tc["name"] == "create_finding":
                            intercepted.append(("finding", tc))
                        elif tc["name"] == "submit_analysis":
                            intercepted.append(("analysis", tc))
                        else:
                            executable.append(tc)

                    # Handle intercepted calls
                    for reason, tc in intercepted:
                        if reason == "loop":
                            loop_msg = (
                                f"LOOP DETECTED: '{tc['name']}' with same arguments was called recently. "
                                "Try a DIFFERENT approach, tool, or parameter."
                            )
                            yield AgentEvent("tool_start", tool_name=tc["name"], arguments=tc["arguments"])
                            yield AgentEvent("tool_end", tool_name=tc["name"], tool_result=loop_msg)
                            self.state.add_message("tool", {"tool_call_id": tc["id"], "content": loop_msg})
                        elif reason == "finding":
                            yield AgentEvent("tool_start", tool_name=tc["name"], arguments=tc["arguments"])
                            finding = self._handle_create_finding(tc)
                            result = json.dumps({"registered": True, "title": finding.title, "severity": finding.severity})
                            yield AgentEvent("finding", finding=finding)
                            yield AgentEvent("tool_end", tool_name=tc["name"], tool_result=result)
                            self.state.add_message("tool", {"tool_call_id": tc["id"], "content": result})
                        elif reason == "analysis":
                            yield AgentEvent("tool_start", tool_name=tc["name"], arguments=tc["arguments"])
                            findings_list, result = self._handle_submit_analysis(tc)
                            for f in findings_list:
                                yield AgentEvent("finding", finding=f)
                            yield AgentEvent("tool_end", tool_name=tc["name"], tool_result=result)
                            self.state.add_message("tool", {"tool_call_id": tc["id"], "content": result})

                    # Human-in-the-loop: check if any tool requires approval
                    if executable and self._approval_callback:
                        approved: list[dict] = []
                        for tc in executable:
                            tool_schema = self.tools.tools.get(tc["name"], {}).get("schema", {})
                            if tool_schema.get("requires_approval", False):
                                try:
                                    ok = await self._approval_callback(tc["name"], tc["arguments"])
                                except Exception:
                                    ok = False
                                if not ok:
                                    msg = f"Tool '{tc['name']}' requires approval — denied by user."
                                    yield AgentEvent("tool_start", tool_name=tc["name"], arguments=tc["arguments"])
                                    yield AgentEvent("tool_end", tool_name=tc["name"], tool_result=msg)
                                    self.state.add_message("tool", {"tool_call_id": tc["id"], "content": msg})
                                    continue
                            approved.append(tc)
                        executable = approved

                    # Execute normal tools in parallel
                    if executable:
                        for tc in executable:
                            yield AgentEvent("tool_start", tool_name=tc["name"], arguments=tc["arguments"])

                        exec_results = await self._do_tool_execution(executable)
                        self._total_tool_calls += len(exec_results)

                        for tc, result in exec_results:
                            yield AgentEvent("tool_end", tool_name=tc["name"], tool_result=result)

                            # Post-processing (Step 4)
                            content, reflection, guidance = await self._do_post_tool_processing(tc, result)

                            # Replanning check
                            if self.state.plan and self.state.plan.needs_replan():
                                try:
                                    self.state.plan = await generate_plan_with_llm(
                                        self.state.plan.objective, self.state.profile, self.router
                                    )
                                    yield AgentEvent("plan_generated", plan=self.state.plan.to_prompt_summary())
                                except Exception:
                                    pass

                            if guidance:
                                content = f"{content}\n\n{guidance}"
                                yield AgentEvent("reflection", tool_name=tc["name"], guidance=guidance)
                            if reflection:
                                content = f"{content}\n\n---\n**Reflection:**\n{reflection}"

                            self.state.add_message("tool", {"tool_call_id": tc["id"], "content": content})

                    # Check phase advancement (async for macro-reflection)
                    advancement = await self._check_phase_advancement()
                    if advancement:
                        yield AgentEvent("phase_complete", phase_name=advancement[0], next_phase=advancement[1])

                    continue

                # No tool calls — text-only response
                if text_buffer and not tool_calls:
                    self.state.add_message("assistant", text_buffer)
                    if self._check_termination(text_buffer):
                        break
                    continue

                # Periodic checkpoint
                if (
                    self._checkpoint_interval > 0
                    and self._session_manager
                    and (iteration + 1) % self._checkpoint_interval == 0
                ):
                    try:
                        self._session_manager.save_checkpoint(
                            self.state, iteration + 1
                        )
                    except Exception as e:
                        logger.debug(f"Checkpoint save failed: {e}")

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
        """Extract first finding from agent text (returns first for backward compat).

        Looks for pattern: [FINDING: SEVERITY] Title
        Override for custom finding patterns.
        """
        findings = self._extract_all_findings(text)
        return findings[0] if findings else None

    def _extract_all_findings(self, text: str) -> list[Finding]:
        """Extract ALL findings from agent text.

        Looks for pattern: [FINDING: SEVERITY] Title
        Supports multiple findings in a single response.
        """
        findings: list[Finding] = []
        for match in re.finditer(r'\[FINDING:\s*(\w+)\]\s*(.+?)(?:\n|$)', text, re.IGNORECASE):
            severity = match.group(1).lower()
            title = match.group(2).strip()

            description = ""
            evidence = ""

            # Search for description/evidence after this finding's position
            remaining = text[match.start():]
            desc_match = re.search(r'\*\*Description\*\*:\s*(.+?)(?:\n\*\*|\n\n|$)', remaining, re.DOTALL)
            if desc_match:
                description = desc_match.group(1).strip()

            ev_match = re.search(r'\*\*Evidence\*\*:\s*(.+?)(?:\n\*\*|\n\n|$)', remaining, re.DOTALL)
            if ev_match:
                evidence = ev_match.group(1).strip()

            title = self._clean_markup(title)
            description = self._clean_markup(description)
            evidence = self._clean_markup(evidence)

            self._on_finding(title, severity, description, evidence)

            findings.append(Finding(
                title=title,
                severity=severity,
                description=description or title,
                evidence=evidence,
            ))
        return findings

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
        activated_paths = self.reasoning_graph.mark_discovered(finding.title)
        if activated_paths:
            self._pending_graph_paths.extend(activated_paths[:3])
        return finding

    def _handle_submit_analysis(self, tc: dict) -> tuple[list[Finding], str]:
        """Handle the submit_analysis tool call (structured output).

        Processes the structured analysis: registers findings, adds hypotheses
        to the domain profile, and returns a summary.
        """
        from omnigent.domain_profile import Hypothesis

        args = tc.get("arguments", {})
        registered_findings: list[Finding] = []

        # Process findings from structured output
        for f_data in args.get("findings", []):
            finding = Finding(
                title=f_data.get("title", "Untitled"),
                severity=f_data.get("severity", "info"),
                description=f_data.get("description", ""),
                evidence=f_data.get("evidence", ""),
            )
            self.state.add_finding(finding)
            activated_paths = self.reasoning_graph.mark_discovered(finding.title)
            if activated_paths:
                self._pending_graph_paths.extend(activated_paths[:3])
            registered_findings.append(finding)

        # Process hypotheses
        for h_data in args.get("hypotheses", []):
            hyp = Hypothesis(
                hypothesis_type=h_data.get("hypothesis_type", ""),
                location=h_data.get("location", ""),
                confidence=h_data.get("confidence", 0.5),
                evidence=h_data.get("evidence", ""),
            )
            self.state.profile.add_hypothesis(hyp)

        result = json.dumps({
            "registered": True,
            "findings_count": len(registered_findings),
            "hypotheses_count": len(args.get("hypotheses", [])),
            "step_summary": args.get("step_summary", ""),
            "next_action": args.get("next_action", ""),
        })
        return registered_findings, result

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

    def _sanitize_tool_output(self, tool_name: str, output: str) -> str:
        """Sanitize tool output to prevent prompt injection.

        Strips patterns that could be mistaken for agent findings or
        system instructions embedded in tool results.
        """
        if not isinstance(output, str):
            return output

        # Neutralize finding-like patterns from tool output
        sanitized = re.sub(
            r'\[FINDING:\s*\w+\]',
            '[TOOL_OUTPUT_FINDING_PATTERN]',
            output,
            flags=re.IGNORECASE,
        )
        # Neutralize system-instruction-like patterns
        sanitized = re.sub(
            r'\[SYSTEM\]:\s*',
            '[TOOL_OUTPUT_SYSTEM_PATTERN]: ',
            sanitized,
            flags=re.IGNORECASE,
        )
        return sanitized

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
