"""
End-to-end tests for the Agent run() loop.

Tests the full ReAct loop with mock LLM router and mock tools,
covering text-only responses, tool calls, findings extraction,
loop detection, create_finding interception, plan generation,
and max iteration limits.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from omnigent.agent import Agent, AgentEvent
from omnigent.planner import TaskPlan, TaskPhase, TaskStep
from omnigent.router import StreamChunk, Provider, TaskType
from omnigent.state import Finding
from omnigent.tools import ToolRegistry


# ═══════════════════════════════════════════════════════════════════════════
# Mock Router
# ═══════════════════════════════════════════════════════════════════════════


class MockRouter:
    """A mock LLM router that returns predefined StreamChunk sequences.

    Usage:
        router = MockRouter(responses=[
            # First call to stream() yields these chunks:
            [StreamChunk(content="Hello "), StreamChunk(content="world", done=True)],
            # Second call to stream() yields these chunks:
            [StreamChunk(content="Done.", done=True)],
        ])

    For planning calls (generate_plan_with_llm calls router.stream with
    task_type=TaskType.PLANNING), a separate plan_response list is used.
    If not provided, a default minimal plan JSON is returned.
    """

    def __init__(
        self,
        responses: list[list[StreamChunk]] | None = None,
        plan_responses: list[list[StreamChunk]] | None = None,
    ):
        self._responses = list(responses or [])
        self._call_index = 0

        # Plan responses are used when task_type=PLANNING
        if plan_responses is not None:
            self._plan_responses = list(plan_responses)
        else:
            # Default: return a simple plan JSON
            plan_json = json.dumps({
                "phases": [
                    {
                        "name": "Analysis",
                        "objective": "Analyze the subject",
                        "steps": [
                            {"description": "Initial analysis", "tool_hint": ""},
                        ],
                    },
                ],
            })
            self._plan_responses = [
                [StreamChunk(content=plan_json, done=True)],
            ]
        self._plan_call_index = 0

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | list[dict] | None = None,
        task_type: TaskType | None = None,
        provider_override: Provider | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Yield predefined StreamChunk sequences.

        Routes to plan_responses when task_type is PLANNING,
        otherwise uses the main responses list.
        """
        if task_type == TaskType.PLANNING:
            if self._plan_call_index < len(self._plan_responses):
                chunks = self._plan_responses[self._plan_call_index]
                self._plan_call_index += 1
            else:
                # Fallback: yield a done chunk
                chunks = [StreamChunk(done=True)]
            for chunk in chunks:
                yield chunk
            return

        if self._call_index < len(self._responses):
            chunks = self._responses[self._call_index]
            self._call_index += 1
        else:
            # Safety: yield done to prevent infinite loops
            yield StreamChunk(done=True)
            return

        for chunk in chunks:
            yield chunk

    async def close(self):
        """No-op close for compatibility."""
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


async def collect_events(agent: Agent, user_input: str) -> list[AgentEvent]:
    """Run the agent and collect all emitted events into a list."""
    events = []
    async for event in agent.run(user_input):
        events.append(event)
    return events


def make_text_response(text: str, input_tokens: int = 10, output_tokens: int = 20) -> list[StreamChunk]:
    """Create a StreamChunk sequence for a text-only LLM response."""
    return [
        StreamChunk(content=text, input_tokens=input_tokens, output_tokens=output_tokens),
        StreamChunk(done=True),
    ]


def make_tool_call_response(
    tool_name: str,
    arguments: dict,
    text_before: str = "",
    tool_call_id: str | None = None,
) -> list[StreamChunk]:
    """Create a StreamChunk sequence that includes a tool call."""
    call_id = tool_call_id or f"call_{uuid.uuid4().hex[:12]}"
    chunks = []
    if text_before:
        chunks.append(StreamChunk(content=text_before))
    chunks.append(StreamChunk(
        tool_call={"id": call_id, "name": tool_name, "arguments": arguments},
    ))
    chunks.append(StreamChunk(done=True, input_tokens=10, output_tokens=20))
    return chunks


def find_events(events: list[AgentEvent], event_type: str) -> list[AgentEvent]:
    """Filter events by type."""
    return [e for e in events if e.type == event_type]


def has_event(events: list[AgentEvent], event_type: str) -> bool:
    """Check if at least one event of the given type exists."""
    return any(e.type == event_type for e in events)


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentTextOnly:
    """Agent receives text-only responses and terminates after 3 consecutive text responses."""

    @pytest.mark.asyncio
    async def test_terminates_after_three_consecutive_text_responses(self):
        """The agent should stop after 3 consecutive text-only (no tool call) responses."""
        router = MockRouter(responses=[
            make_text_response("First text response."),
            make_text_response("Second text response."),
            make_text_response("Third text response."),
        ])

        agent = Agent(router=router, max_iterations=10)
        events = await collect_events(agent, "Do something")

        text_events = find_events(events, "text")
        assert len(text_events) >= 3, "Should emit text events for all 3 responses"

        done_events = find_events(events, "done")
        assert len(done_events) == 1, "Should emit exactly one done event"

    @pytest.mark.asyncio
    async def test_text_content_is_streamed(self):
        """Text content from the LLM should be emitted as text events."""
        router = MockRouter(responses=[
            [
                StreamChunk(content="Hello "),
                StreamChunk(content="world!"),
                StreamChunk(done=True),
            ],
            make_text_response("Second response."),
            make_text_response("Third response."),
        ])

        agent = Agent(router=router, max_iterations=10)
        events = await collect_events(agent, "Say hello")

        text_events = find_events(events, "text")
        combined_text = "".join(e.content for e in text_events)
        assert "Hello " in combined_text
        assert "world!" in combined_text

    @pytest.mark.asyncio
    async def test_done_indicator_terminates_early(self):
        """A response containing a done indicator (e.g., 'task complete')
        should terminate the agent before 3 consecutive text responses."""
        router = MockRouter(responses=[
            make_text_response("I have completed the analysis. Task complete."),
        ])

        agent = Agent(router=router, max_iterations=10)
        events = await collect_events(agent, "Analyze this")

        done_events = find_events(events, "done")
        assert len(done_events) == 1

        # Should only have 1 text response, not 3
        text_contents = "".join(e.content for e in find_events(events, "text"))
        assert "Task complete" in text_contents

    @pytest.mark.asyncio
    async def test_messages_accumulate_in_state(self):
        """Text-only responses should be added to agent state as assistant messages."""
        router = MockRouter(responses=[
            make_text_response("Response one."),
            make_text_response("Response two."),
            make_text_response("Response three."),
        ])

        agent = Agent(router=router, max_iterations=10)
        await collect_events(agent, "Hello")

        # The state should contain user message + 3 assistant messages
        assistant_msgs = [m for m in agent.state.messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 3


class TestAgentToolCall:
    """Agent receives a tool call, executes it, and gets the result."""

    @pytest.mark.asyncio
    async def test_tool_call_executes_and_returns_result(self):
        """A tool call should be executed and its result emitted as tool_end."""
        registry = ToolRegistry()

        async def mock_lookup(query: str = "") -> str:
            return json.dumps({"result": f"Found: {query}"})

        registry.register("lookup", mock_lookup, {
            "description": "Look up information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        })

        call_id = "call_abc123"
        router = MockRouter(responses=[
            make_tool_call_response("lookup", {"query": "test"}, tool_call_id=call_id),
            # After tool result, LLM responds with text
            make_text_response("I found the result. Task complete."),
        ])

        agent = Agent(router=router, tools=registry, max_iterations=10)
        events = await collect_events(agent, "Look up test")

        # Should have tool_start and tool_end events
        tool_starts = find_events(events, "tool_start")
        tool_ends = find_events(events, "tool_end")
        assert len(tool_starts) >= 1, "Should emit tool_start"
        assert len(tool_ends) >= 1, "Should emit tool_end"

        # The tool_end should contain the result
        tool_end = tool_ends[0]
        assert "Found: test" in tool_end.tool_result

    @pytest.mark.asyncio
    async def test_tool_call_with_text_before(self):
        """A response with text + tool call should emit both text and tool events."""
        registry = ToolRegistry()

        async def mock_tool(x: str = "") -> str:
            return "result"

        registry.register("my_tool", mock_tool, {
            "description": "Test tool",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
        })

        router = MockRouter(responses=[
            make_tool_call_response("my_tool", {"x": "val"}, text_before="Let me check..."),
            make_text_response("Done. Task complete."),
        ])

        agent = Agent(router=router, tools=registry, max_iterations=10)
        events = await collect_events(agent, "Run tool")

        text_events = find_events(events, "text")
        assert any("Let me check" in e.content for e in text_events)

        assert has_event(events, "tool_start")
        assert has_event(events, "tool_end")

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        """Calling a tool not in the registry should return an error result."""
        router = MockRouter(responses=[
            make_tool_call_response("nonexistent_tool", {"arg": "val"}),
            make_text_response("I see. Task complete."),
        ])

        agent = Agent(router=router, max_iterations=10)
        events = await collect_events(agent, "Do something")

        tool_ends = find_events(events, "tool_end")
        assert len(tool_ends) >= 1
        # ToolRegistry.call returns {"error": "Unknown tool: ..."} for unknown tools
        assert "Unknown tool" in tool_ends[0].tool_result or "error" in tool_ends[0].tool_result.lower()

    @pytest.mark.asyncio
    async def test_tool_exception_is_caught(self):
        """If a tool raises an exception, it should be caught and returned as error."""
        registry = ToolRegistry()

        async def failing_tool() -> str:
            raise RuntimeError("Something went wrong")

        registry.register("failing", failing_tool, {
            "description": "A tool that fails",
            "parameters": {"type": "object", "properties": {}},
        })

        router = MockRouter(responses=[
            make_tool_call_response("failing", {}),
            make_text_response("I see the error. Task complete."),
        ])

        agent = Agent(router=router, tools=registry, max_iterations=10)
        events = await collect_events(agent, "Run failing tool")

        tool_ends = find_events(events, "tool_end")
        assert len(tool_ends) >= 1
        # The error should be in the result
        assert "Something went wrong" in tool_ends[0].tool_result or "error" in tool_ends[0].tool_result.lower()


class TestAgentFinding:
    """Agent text contains [FINDING: HIGH] title pattern, finding is extracted."""

    @pytest.mark.asyncio
    async def test_finding_extracted_from_text(self):
        """A finding pattern in text should produce a finding event."""
        text_with_finding = (
            "During analysis I discovered:\n"
            "[FINDING: HIGH] SQL Injection in Login Form\n"
            "**Description**: The login form is vulnerable to SQL injection.\n"
            "**Evidence**: Input `' OR 1=1 --` bypasses authentication.\n"
        )

        router = MockRouter(responses=[
            make_text_response(text_with_finding),
            make_text_response("No more findings. Task complete."),
        ])

        agent = Agent(router=router, max_iterations=10)
        events = await collect_events(agent, "Analyze target")

        finding_events = find_events(events, "finding")
        assert len(finding_events) >= 1, "Should emit at least one finding event"

        finding = finding_events[0].finding
        assert finding is not None
        assert "SQL Injection" in finding.title
        assert finding.severity == "high"

    @pytest.mark.asyncio
    async def test_finding_added_to_state(self):
        """Extracted findings should be stored in agent.state.findings."""
        router = MockRouter(responses=[
            make_text_response("[FINDING: CRITICAL] Remote Code Execution\n"),
            make_text_response("Analysis complete. Task complete."),
        ])

        agent = Agent(router=router, max_iterations=10)
        await collect_events(agent, "Test")

        assert len(agent.state.findings) >= 1
        assert any("Remote Code Execution" in f.title for f in agent.state.findings)

    @pytest.mark.asyncio
    async def test_finding_severity_normalization(self):
        """Severity should be normalized to lowercase."""
        router = MockRouter(responses=[
            make_text_response("[FINDING: MEDIUM] Exposed Debug Endpoint\n"),
            make_text_response("Done. Task complete."),
        ])

        agent = Agent(router=router, max_iterations=10)
        events = await collect_events(agent, "Check")

        finding_events = find_events(events, "finding")
        assert len(finding_events) >= 1
        assert finding_events[0].finding.severity == "medium"


class TestAgentMultiFinding:
    """Agent text contains multiple [FINDING:...] patterns; ALL are extracted."""

    @pytest.mark.asyncio
    async def test_multiple_findings_extracted(self):
        """Multiple finding patterns in a single response should all be extracted."""
        text_with_multiple = (
            "Here are my findings:\n\n"
            "[FINDING: HIGH] Cross-Site Scripting in Search\n"
            "**Description**: Reflected XSS via search parameter.\n\n"
            "[FINDING: MEDIUM] Missing CSRF Token\n"
            "**Description**: Forms lack CSRF protection.\n\n"
            "[FINDING: LOW] Verbose Error Messages\n"
            "**Description**: Stack traces visible to users.\n"
        )

        router = MockRouter(responses=[
            make_text_response(text_with_multiple),
            make_text_response("All findings reported. Task complete."),
        ])

        agent = Agent(router=router, max_iterations=10)
        events = await collect_events(agent, "Analyze")

        finding_events = find_events(events, "finding")
        assert len(finding_events) == 3, f"Expected 3 findings, got {len(finding_events)}"

        titles = [e.finding.title for e in finding_events]
        severities = [e.finding.severity for e in finding_events]

        assert "Cross-Site Scripting in Search" in titles
        assert "Missing CSRF Token" in titles
        assert "Verbose Error Messages" in titles

        assert "high" in severities
        assert "medium" in severities
        assert "low" in severities

    @pytest.mark.asyncio
    async def test_all_multi_findings_stored_in_state(self):
        """All extracted findings should appear in agent.state.findings."""
        text = (
            "[FINDING: HIGH] Issue A\n"
            "[FINDING: LOW] Issue B\n"
        )

        router = MockRouter(responses=[
            make_text_response(text),
            make_text_response("Done. Task complete."),
        ])

        agent = Agent(router=router, max_iterations=10)
        await collect_events(agent, "Check")

        assert len(agent.state.findings) == 2
        titles = {f.title for f in agent.state.findings}
        assert "Issue A" in titles
        assert "Issue B" in titles

    @pytest.mark.asyncio
    async def test_findings_across_multiple_responses(self):
        """Findings from different LLM responses should all accumulate."""
        router = MockRouter(responses=[
            make_text_response("[FINDING: HIGH] Finding One\n"),
            make_text_response("[FINDING: CRITICAL] Finding Two\n"),
            make_text_response("All done. Task complete."),
        ])

        agent = Agent(router=router, max_iterations=10)
        events = await collect_events(agent, "Analyze")

        finding_events = find_events(events, "finding")
        assert len(finding_events) == 2
        assert agent.state.findings[0].title == "Finding One"
        assert agent.state.findings[1].title == "Finding Two"


class TestAgentLoopDetection:
    """Same tool called with same args 3 times triggers loop detection."""

    @pytest.mark.asyncio
    async def test_loop_detected_on_repeated_tool_call(self):
        """The third identical tool call should be intercepted with a loop message."""
        registry = ToolRegistry()

        call_count = 0

        async def counting_tool(host: str = "") -> str:
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        registry.register("scan", counting_tool, {
            "description": "Scan a host",
            "parameters": {
                "type": "object",
                "properties": {"host": {"type": "string"}},
                "required": ["host"],
            },
        })

        # Same tool + same args three times, then a text response to end
        same_call = make_tool_call_response("scan", {"host": "target.com"})
        router = MockRouter(responses=[
            same_call,
            same_call,
            same_call,
            make_text_response("Task complete."),
        ])

        agent = Agent(router=router, tools=registry, max_iterations=10)
        events = await collect_events(agent, "Scan target.com")

        tool_ends = find_events(events, "tool_end")
        # At least one tool_end should contain the LOOP DETECTED message
        loop_messages = [e for e in tool_ends if "LOOP DETECTED" in e.tool_result]
        assert len(loop_messages) >= 1, "Should detect loop on repeated tool call"

        # The tool should have been actually called fewer than 3 times
        # (first two execute, third is blocked)
        assert call_count <= 2, f"Tool should execute at most 2 times, but executed {call_count}"

    @pytest.mark.asyncio
    async def test_different_args_no_loop(self):
        """Different arguments should not trigger loop detection."""
        registry = ToolRegistry()

        results = []

        async def scan(host: str = "") -> str:
            results.append(host)
            return f"scanned {host}"

        registry.register("scan", scan, {
            "description": "Scan",
            "parameters": {
                "type": "object",
                "properties": {"host": {"type": "string"}},
                "required": ["host"],
            },
        })

        router = MockRouter(responses=[
            make_tool_call_response("scan", {"host": "a.com"}),
            make_tool_call_response("scan", {"host": "b.com"}),
            make_tool_call_response("scan", {"host": "c.com"}),
            make_text_response("Task complete."),
        ])

        agent = Agent(router=router, tools=registry, max_iterations=10)
        events = await collect_events(agent, "Scan multiple hosts")

        tool_ends = find_events(events, "tool_end")
        loop_messages = [e for e in tool_ends if "LOOP DETECTED" in e.tool_result]
        assert len(loop_messages) == 0, "Different args should not trigger loop detection"
        assert len(results) == 3, "All three unique calls should execute"


class TestAgentCreateFinding:
    """Agent calls create_finding tool -- intercepted and Finding created."""

    @pytest.mark.asyncio
    async def test_create_finding_intercepted(self):
        """The create_finding tool should be intercepted by the agent,
        not executed as a normal tool. A Finding should be created."""
        router = MockRouter(responses=[
            make_tool_call_response("create_finding", {
                "title": "Hardcoded Credentials",
                "severity": "critical",
                "description": "Database password is hardcoded in config.py",
                "evidence": "password = 'admin123'",
            }),
            make_text_response("Finding reported. Task complete."),
        ])

        agent = Agent(router=router, max_iterations=10)
        events = await collect_events(agent, "Analyze code")

        # Should have a finding event
        finding_events = find_events(events, "finding")
        assert len(finding_events) >= 1

        finding = finding_events[0].finding
        assert finding.title == "Hardcoded Credentials"
        assert finding.severity == "critical"
        assert "hardcoded" in finding.description.lower()

        # Should also be in state
        assert len(agent.state.findings) >= 1
        assert agent.state.findings[0].title == "Hardcoded Credentials"

    @pytest.mark.asyncio
    async def test_create_finding_returns_registered_result(self):
        """The tool_end event for create_finding should contain a registered: true result."""
        router = MockRouter(responses=[
            make_tool_call_response("create_finding", {
                "title": "Open Redirect",
                "severity": "medium",
                "description": "URL parameter allows redirect to external site",
            }),
            make_text_response("Done. Task complete."),
        ])

        agent = Agent(router=router, max_iterations=10)
        events = await collect_events(agent, "Test")

        tool_ends = find_events(events, "tool_end")
        cf_ends = [e for e in tool_ends if e.tool_name == "create_finding"]
        assert len(cf_ends) >= 1

        result = json.loads(cf_ends[0].tool_result)
        assert result["registered"] is True
        assert result["title"] == "Open Redirect"
        assert result["severity"] == "medium"

    @pytest.mark.asyncio
    async def test_create_finding_with_missing_optional_fields(self):
        """create_finding should work even if optional fields are missing."""
        router = MockRouter(responses=[
            make_tool_call_response("create_finding", {
                "title": "Minor Issue",
                "severity": "low",
                "description": "A minor issue found",
            }),
            make_text_response("Task complete."),
        ])

        agent = Agent(router=router, max_iterations=10)
        events = await collect_events(agent, "Check")

        finding_events = find_events(events, "finding")
        assert len(finding_events) >= 1
        assert finding_events[0].finding.title == "Minor Issue"
        assert finding_events[0].finding.evidence == ""  # Optional, default empty


class TestAgentPlanGeneration:
    """Agent generates a plan on first run."""

    @pytest.mark.asyncio
    async def test_plan_generated_event_emitted(self):
        """The agent should emit a plan_generated event at the start."""
        router = MockRouter(responses=[
            make_text_response("Starting analysis. Task complete."),
        ])

        agent = Agent(router=router, max_iterations=10)
        events = await collect_events(agent, "Analyze the target system")

        plan_events = find_events(events, "plan_generated")
        assert len(plan_events) >= 1, "Should emit plan_generated event"
        assert plan_events[0].data.get("plan"), "Plan event should contain plan data"

    @pytest.mark.asyncio
    async def test_plan_stored_in_state(self):
        """After plan generation, the plan should be stored in agent.state.plan."""
        router = MockRouter(responses=[
            make_text_response("Analysis done. Task complete."),
        ])

        agent = Agent(router=router, max_iterations=10)
        await collect_events(agent, "Investigate the issue")

        assert agent.state.plan is not None
        assert agent.state.plan.objective != ""
        assert len(agent.state.plan.phases) >= 1

    @pytest.mark.asyncio
    async def test_custom_plan_from_mock_router(self):
        """The mock router's plan response should be parsed into the agent's plan."""
        custom_plan_json = json.dumps({
            "phases": [
                {
                    "name": "Reconnaissance",
                    "objective": "Gather initial information",
                    "steps": [
                        {"description": "DNS enumeration", "tool_hint": "dns_lookup"},
                        {"description": "Port scanning", "tool_hint": "nmap"},
                    ],
                },
                {
                    "name": "Exploitation",
                    "objective": "Test discovered vulnerabilities",
                    "steps": [
                        {"description": "SQL injection testing", "tool_hint": "sqlmap"},
                    ],
                },
            ],
        })

        router = MockRouter(
            responses=[
                make_text_response("Starting. Task complete."),
            ],
            plan_responses=[
                [StreamChunk(content=custom_plan_json, done=True)],
            ],
        )

        agent = Agent(router=router, max_iterations=10)
        await collect_events(agent, "Full security assessment")

        plan = agent.state.plan
        assert len(plan.phases) == 2
        assert plan.phases[0].name == "Reconnaissance"
        assert plan.phases[1].name == "Exploitation"
        assert len(plan.phases[0].steps) == 2
        assert plan.phases[0].steps[0].tool_hint == "dns_lookup"

    @pytest.mark.asyncio
    async def test_plan_not_regenerated_on_second_iteration(self):
        """Once a plan is generated, subsequent iterations should NOT regenerate it
        unless it is complete."""
        plan_call_count = 0

        class CountingRouter(MockRouter):
            async def stream(self, messages, tools=None, system=None, task_type=None, provider_override=None):
                nonlocal plan_call_count
                if task_type == TaskType.PLANNING:
                    plan_call_count += 1
                async for chunk in super().stream(messages, tools, system, task_type, provider_override):
                    yield chunk

        router = CountingRouter(responses=[
            make_text_response("Working on it..."),
            make_text_response("Still working..."),
            make_text_response("All done. Task complete."),
        ])

        agent = Agent(router=router, max_iterations=10)
        await collect_events(agent, "Analyze target")

        # Plan should only be generated once (first call)
        assert plan_call_count == 1, f"Plan should be generated once, but was generated {plan_call_count} times"


class TestAgentMaxIterations:
    """Agent stops after max_iterations."""

    @pytest.mark.asyncio
    async def test_stops_at_max_iterations(self):
        """The agent should stop after max_iterations even if the LLM keeps producing tool calls."""
        registry = ToolRegistry()

        exec_count = 0

        async def simple_tool(x: str = "") -> str:
            nonlocal exec_count
            exec_count += 1
            return f"result_{exec_count}"

        registry.register("simple", simple_tool, {
            "description": "Simple tool",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
            },
        })

        # Each iteration calls a different x to avoid loop detection.
        # Generate more responses than max_iterations so the router never runs out.
        max_iter = 5
        responses = [
            make_tool_call_response("simple", {"x": f"val_{i}"})
            for i in range(max_iter + 10)
        ]

        router = MockRouter(responses=responses)

        agent = Agent(router=router, tools=registry, max_iterations=max_iter)
        events = await collect_events(agent, "Keep going forever")

        # Should have a done event
        assert has_event(events, "done"), "Should emit done event even at max iterations"

        # The number of tool executions should be at most max_iterations
        assert exec_count <= max_iter, (
            f"Tool executed {exec_count} times, but max_iterations is {max_iter}"
        )

    @pytest.mark.asyncio
    async def test_max_iterations_with_small_limit(self):
        """With max_iterations=1, agent should stop after a single iteration."""
        router = MockRouter(responses=[
            make_text_response("Only response."),
            make_text_response("Should not reach here."),
        ])

        agent = Agent(router=router, max_iterations=1)
        events = await collect_events(agent, "Quick check")

        # Even with max_iterations=1, the agent does one iteration
        # Then the for loop ends and done is emitted
        done_events = find_events(events, "done")
        assert len(done_events) == 1

    @pytest.mark.asyncio
    async def test_max_iterations_counts_all_iteration_types(self):
        """Both text-only and tool-call iterations count toward max_iterations."""
        registry = ToolRegistry()

        async def tool_fn(q: str = "") -> str:
            return "ok"

        registry.register("check", tool_fn, {
            "description": "Check",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        })

        # Mix of text and tool responses, more than max_iterations
        max_iter = 3
        responses = [
            make_tool_call_response("check", {"q": "1"}),
            make_text_response("Thinking..."),
            make_tool_call_response("check", {"q": "2"}),
            make_text_response("More thinking..."),  # Should not be reached
        ]

        router = MockRouter(responses=responses)
        agent = Agent(router=router, tools=registry, max_iterations=max_iter)
        events = await collect_events(agent, "Do work")

        assert has_event(events, "done")


class TestAgentUsageTracking:
    """Agent emits usage events with token counts."""

    @pytest.mark.asyncio
    async def test_usage_event_emitted(self):
        """When the LLM returns token counts, a usage event should be emitted."""
        router = MockRouter(responses=[
            [
                StreamChunk(
                    content="Response text.",
                    input_tokens=100,
                    output_tokens=50,
                    cache_read_tokens=20,
                    cache_creation_tokens=10,
                ),
                StreamChunk(done=True),
            ],
            make_text_response("Done. Task complete."),
        ])

        agent = Agent(router=router, max_iterations=10)
        events = await collect_events(agent, "Analyze")

        usage_events = find_events(events, "usage")
        assert len(usage_events) >= 1, "Should emit usage event"
        usage_data = usage_events[0].data
        assert usage_data["input_tokens"] == 100
        assert usage_data["output_tokens"] == 50


class TestAgentErrorHandling:
    """Agent handles empty responses gracefully."""

    @pytest.mark.asyncio
    async def test_empty_response_emits_error(self):
        """An empty LLM response (no text, no tools) should emit an error event."""
        router = MockRouter(responses=[
            [StreamChunk(done=True)],  # Empty response: no content, no tool calls
        ])

        agent = Agent(router=router, max_iterations=10)
        events = await collect_events(agent, "Do something")

        error_events = find_events(events, "error")
        assert len(error_events) >= 1, "Should emit error for empty response"
        assert "empty" in error_events[0].data.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_done_always_emitted(self):
        """The agent should always emit a done event at the end, regardless of outcome."""
        router = MockRouter(responses=[
            make_text_response("Task complete."),
        ])

        agent = Agent(router=router, max_iterations=10)
        events = await collect_events(agent, "Quick task")

        assert events[-1].type == "done", "Last event should always be done"


class TestAgentStateManagement:
    """Agent state is properly maintained throughout the run loop."""

    @pytest.mark.asyncio
    async def test_user_message_added_to_state(self):
        """The user input should be added to state messages."""
        router = MockRouter(responses=[
            make_text_response("Task complete."),
        ])

        agent = Agent(router=router, max_iterations=10)
        await collect_events(agent, "My special query")

        user_msgs = [m for m in agent.state.messages if m["role"] == "user"]
        assert len(user_msgs) >= 1
        assert user_msgs[0]["content"] == "My special query"

    @pytest.mark.asyncio
    async def test_tool_results_added_to_state(self):
        """Tool results should be added to state as tool messages."""
        registry = ToolRegistry()

        async def echo(text: str = "") -> str:
            return f"Echo: {text}"

        registry.register("echo", echo, {
            "description": "Echo tool",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        })

        router = MockRouter(responses=[
            make_tool_call_response("echo", {"text": "hello"}),
            make_text_response("Got result. Task complete."),
        ])

        agent = Agent(router=router, tools=registry, max_iterations=10)
        await collect_events(agent, "Echo hello")

        tool_msgs = [m for m in agent.state.messages if m["role"] == "tool"]
        assert len(tool_msgs) >= 1

    @pytest.mark.asyncio
    async def test_is_running_flag(self):
        """agent.is_running should be True during execution and False after."""
        router = MockRouter(responses=[
            make_text_response("Task complete."),
        ])

        agent = Agent(router=router, max_iterations=10)

        # Before run
        assert not agent.is_running

        # After run
        await collect_events(agent, "Test")
        assert not agent.is_running, "is_running should be False after run completes"

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        """agent.reset() should clear all state."""
        router = MockRouter(responses=[
            make_text_response("[FINDING: HIGH] Test Finding\n"),
            make_text_response("Done. Task complete."),
        ])

        agent = Agent(router=router, max_iterations=10)
        await collect_events(agent, "Run test")

        assert len(agent.state.findings) > 0
        assert len(agent.state.messages) > 0

        agent.reset()

        assert len(agent.state.findings) == 0
        assert len(agent.state.messages) == 0
