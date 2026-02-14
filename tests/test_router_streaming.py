"""Tests for LLMRouter streaming functionality with mock httpx responses.

Covers OpenAI-compatible streaming (DeepSeek, OpenAI, Ollama), Anthropic streaming,
provider fallback, and retry on transient errors. All HTTP interactions are mocked
using httpx.MockTransport.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from omnigent.router import LLMRouter, Provider

# ═══════════════════════════════════════════════════════════════════════════
# Helpers — SSE mock response builders
# ═══════════════════════════════════════════════════════════════════════════


def _sse_lines(events: list[str]) -> bytes:
    """Build raw SSE byte payload from a list of 'data: ...' strings.

    Each event is separated by a double newline as per the SSE specification.
    A final ``data: [DONE]`` sentinel is appended for OpenAI-compatible streams.
    """
    lines: list[str] = []
    for event in events:
        lines.append(f"data: {event}")
        lines.append("")  # blank line after each event
    lines.append("data: [DONE]")
    lines.append("")
    return "\n".join(lines).encode()


def _sse_lines_anthropic(events: list[str]) -> bytes:
    """Build raw SSE byte payload for Anthropic-style streams (no [DONE])."""
    lines: list[str] = []
    for event in events:
        lines.append(f"data: {event}")
        lines.append("")
    return "\n".join(lines).encode()


def _openai_text_chunk(content: str, index: int = 0) -> str:
    """Return a single OpenAI-format SSE text-delta payload."""
    return json.dumps({
        "choices": [{"delta": {"content": content}, "index": index}],
    })


def _openai_tool_call_chunk(
    *,
    index: int = 0,
    tool_call_index: int = 0,
    tc_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> str:
    """Return a single OpenAI-format SSE tool-call delta payload."""
    tc: dict[str, Any] = {"index": tool_call_index}
    if tc_id is not None:
        tc["id"] = tc_id
    func: dict[str, str] = {}
    if name is not None:
        func["name"] = name
    if arguments is not None:
        func["arguments"] = arguments
    if func:
        tc["function"] = func
    return json.dumps({"choices": [{"delta": {"tool_calls": [tc]}, "index": index}]})


def _openai_finish_chunk(reason: str = "stop") -> str:
    """Return an OpenAI-format finish-reason SSE payload."""
    return json.dumps({"choices": [{"delta": {}, "index": 0, "finish_reason": reason}]})


def _openai_usage_chunk(prompt_tokens: int = 0, completion_tokens: int = 0) -> str:
    """Return an OpenAI-format usage SSE payload.

    With ``stream_options: {include_usage: true}``, OpenAI sends usage in a
    dedicated SSE event whose ``choices`` array is empty.  The router code
    accesses ``choices[0]`` after processing usage, so we include a single
    empty-delta choice to avoid an IndexError.
    """
    return json.dumps({
        "choices": [{"delta": {}, "index": 0}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    })


def _anthropic_message_start(
    input_tokens: int = 0,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> str:
    return json.dumps({
        "type": "message_start",
        "message": {
            "usage": {
                "input_tokens": input_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_creation,
            }
        },
    })


def _anthropic_text_delta(text: str) -> str:
    return json.dumps({
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": text},
    })


def _anthropic_tool_start(tool_id: str, name: str) -> str:
    return json.dumps({
        "type": "content_block_start",
        "content_block": {"type": "tool_use", "id": tool_id, "name": name},
    })


def _anthropic_tool_delta(partial_json: str) -> str:
    return json.dumps({
        "type": "content_block_delta",
        "delta": {"type": "input_json_delta", "partial_json": partial_json},
    })


def _anthropic_content_block_stop() -> str:
    return json.dumps({"type": "content_block_stop"})


def _anthropic_message_delta(
    output_tokens: int = 0,
    input_tokens: int = 0,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> str:
    return json.dumps({
        "type": "message_delta",
        "usage": {
            "output_tokens": output_tokens,
            "input_tokens": input_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
        },
    })


def _anthropic_message_stop() -> str:
    return json.dumps({"type": "message_stop"})


# ═══════════════════════════════════════════════════════════════════════════
# Transport helpers — create httpx.MockTransport for async streaming
# ═══════════════════════════════════════════════════════════════════════════


def _make_mock_transport(
    body: bytes,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.MockTransport:
    """Create an httpx.MockTransport that returns a streaming response.

    The transport works for ``client.stream(...)`` context-manager usage
    because ``httpx.Response`` with ``stream=...`` supports async iteration
    when used within MockTransport's async handler.
    """
    resp_headers = {"content-type": "text/event-stream"}
    if headers:
        resp_headers.update(headers)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, content=body, headers=resp_headers)

    return httpx.MockTransport(handler)


def _make_sequential_transport(responses: list[tuple[int, bytes]]) -> httpx.MockTransport:
    """Transport that yields different responses on successive calls.

    ``responses`` is a list of (status_code, body) tuples. Each call consumes
    the next entry. If calls exceed the list length, the last entry repeats.
    """
    call_index = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        idx = min(call_index["n"], len(responses) - 1)
        call_index["n"] += 1
        status, body = responses[idx]
        return httpx.Response(status, content=body, headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


def _make_capture_transport(
    body: bytes,
    status_code: int = 200,
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """Transport that records all requests for later inspection."""
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(status_code, content=body, headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler), captured


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _set_api_keys(monkeypatch):
    """Ensure fake API keys are present so the router doesn't skip providers."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek-key-000000000")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-anthropic-key-0000000000000000")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-key-000000000")


def _simple_messages() -> list[dict]:
    return [{"role": "user", "content": "Hello"}]


# ═══════════════════════════════════════════════════════════════════════════
# 1. TestOpenAIStreamText
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenAIStreamText:
    """Mock SSE response with text deltas -> collect text content."""

    async def test_single_text_chunk(self):
        """A single text delta is yielded as StreamChunk.content."""
        body = _sse_lines([
            _openai_text_chunk("Hello world"),
            _openai_finish_chunk("stop"),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        chunks = []
        async for chunk in router.stream(_simple_messages()):
            chunks.append(chunk)
        await router.close()

        text_chunks = [c for c in chunks if c.content]
        assert len(text_chunks) == 1
        assert text_chunks[0].content == "Hello world"

    async def test_multiple_text_deltas_concatenate(self):
        """Multiple text deltas should produce multiple chunks that concatenate."""
        body = _sse_lines([
            _openai_text_chunk("Hello "),
            _openai_text_chunk("beautiful "),
            _openai_text_chunk("world!"),
            _openai_finish_chunk("stop"),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        collected_text = ""
        async for chunk in router.stream(_simple_messages()):
            if chunk.content:
                collected_text += chunk.content
        await router.close()

        assert collected_text == "Hello beautiful world!"

    async def test_done_chunk_is_emitted(self):
        """The stream should end with a chunk where done=True."""
        body = _sse_lines([
            _openai_text_chunk("Hi"),
            _openai_finish_chunk("stop"),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        chunks = []
        async for chunk in router.stream(_simple_messages()):
            chunks.append(chunk)
        await router.close()

        assert any(c.done for c in chunks), "Expected a done=True chunk"

    async def test_model_name_in_chunks(self):
        """Each chunk should carry the model name from the provider config."""
        body = _sse_lines([
            _openai_text_chunk("test"),
            _openai_finish_chunk("stop"),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        models = set()
        async for chunk in router.stream(_simple_messages()):
            if chunk.model:
                models.add(chunk.model)
        await router.close()

        assert "deepseek-chat" in models


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestOpenAIStreamToolCall
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenAIStreamToolCall:
    """Mock SSE with tool_calls in delta -> collect tool call."""

    async def test_single_tool_call(self):
        """A complete tool call streamed in pieces is assembled correctly."""
        body = _sse_lines([
            _openai_tool_call_chunk(tc_id="call_abc123", name="get_weather"),
            _openai_tool_call_chunk(arguments='{"city":'),
            _openai_tool_call_chunk(arguments='"Paris"}'),
            _openai_finish_chunk("tool_calls"),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        tool_chunks = []
        async for chunk in router.stream(_simple_messages()):
            if chunk.tool_call:
                tool_chunks.append(chunk)
        await router.close()

        assert len(tool_chunks) == 1
        tc = tool_chunks[0].tool_call
        assert tc["id"] == "call_abc123"
        assert tc["name"] == "get_weather"
        assert tc["arguments"] == {"city": "Paris"}

    async def test_multiple_tool_calls(self):
        """Multiple tool calls with different indices are collected independently."""
        body = _sse_lines([
            # Tool call 0
            _openai_tool_call_chunk(tool_call_index=0, tc_id="call_001", name="search"),
            _openai_tool_call_chunk(tool_call_index=0, arguments='{"q": "test"}'),
            # Tool call 1
            _openai_tool_call_chunk(tool_call_index=1, tc_id="call_002", name="fetch"),
            _openai_tool_call_chunk(tool_call_index=1, arguments='{"url": "https://example.com"}'),
            _openai_finish_chunk("tool_calls"),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        tool_chunks = []
        async for chunk in router.stream(_simple_messages()):
            if chunk.tool_call:
                tool_chunks.append(chunk)
        await router.close()

        assert len(tool_chunks) == 2
        names = {tc.tool_call["name"] for tc in tool_chunks}
        assert names == {"search", "fetch"}

    async def test_tool_call_with_empty_arguments(self):
        """A tool call with no arguments results in an empty dict."""
        body = _sse_lines([
            _openai_tool_call_chunk(tc_id="call_empty", name="list_files"),
            _openai_finish_chunk("tool_calls"),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        tool_chunks = []
        async for chunk in router.stream(_simple_messages()):
            if chunk.tool_call:
                tool_chunks.append(chunk)
        await router.close()

        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_call["arguments"] == {}


# ═══════════════════════════════════════════════════════════════════════════
# 3. TestOpenAIStreamUsage
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenAIStreamUsage:
    """Mock SSE with usage info -> verify token counts."""

    async def test_usage_tokens_captured(self):
        """Usage chunk carries prompt_tokens and completion_tokens."""
        body = _sse_lines([
            _openai_text_chunk("Hi"),
            _openai_usage_chunk(prompt_tokens=150, completion_tokens=42),
            _openai_finish_chunk("stop"),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        usage_chunks = []
        async for chunk in router.stream(_simple_messages()):
            if chunk.input_tokens or chunk.output_tokens:
                usage_chunks.append(chunk)
        await router.close()

        assert len(usage_chunks) >= 1
        u = usage_chunks[0]
        assert u.input_tokens == 150
        assert u.output_tokens == 42

    async def test_usage_zero_by_default(self):
        """Text chunks without usage have input_tokens and output_tokens == 0."""
        body = _sse_lines([
            _openai_text_chunk("Hello"),
            _openai_finish_chunk("stop"),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        text_chunks = []
        async for chunk in router.stream(_simple_messages()):
            if chunk.content:
                text_chunks.append(chunk)
        await router.close()

        for c in text_chunks:
            assert c.input_tokens == 0
            assert c.output_tokens == 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. TestOpenAISystemPrompt
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenAISystemPrompt:
    """Verify system prompt is sent as native {"role":"system"} message."""

    async def test_system_prompt_as_system_role_message(self):
        """System prompt must appear as a {"role": "system"} message, not injected into user content."""
        body = _sse_lines([
            _openai_text_chunk("ok"),
            _openai_finish_chunk("stop"),
        ])
        transport, captured = _make_capture_transport(body)

        router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        async for _ in router.stream(
            _simple_messages(),
            system="You are a helpful assistant.",
        ):
            pass
        await router.close()

        assert len(captured) == 1
        payload = json.loads(captured[0].content.decode())
        messages = payload["messages"]

        # First message should be the system message
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        # Second message should be the original user message (unmodified)
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    async def test_system_prompt_not_in_user_content(self):
        """The user message must not contain the system prompt text."""
        body = _sse_lines([
            _openai_text_chunk("ok"),
            _openai_finish_chunk("stop"),
        ])
        transport, captured = _make_capture_transport(body)

        router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        async for _ in router.stream(
            _simple_messages(),
            system="SECRET_SYSTEM_INSTRUCTIONS",
        ):
            pass
        await router.close()

        payload = json.loads(captured[0].content.decode())
        user_messages = [m for m in payload["messages"] if m["role"] == "user"]
        for msg in user_messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                assert "SECRET_SYSTEM_INSTRUCTIONS" not in content
            elif isinstance(content, list):
                for block in content:
                    assert "SECRET_SYSTEM_INSTRUCTIONS" not in str(block)

    async def test_system_prompt_list_format(self):
        """System prompt passed as Anthropic cacheable list format is extracted correctly."""
        body = _sse_lines([
            _openai_text_chunk("ok"),
            _openai_finish_chunk("stop"),
        ])
        transport, captured = _make_capture_transport(body)

        router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        system_list = [{"type": "text", "text": "Be concise.", "cache_control": {"type": "ephemeral"}}]
        async for _ in router.stream(_simple_messages(), system=system_list):
            pass
        await router.close()

        payload = json.loads(captured[0].content.decode())
        messages = payload["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise."

    async def test_no_system_prompt_no_system_message(self):
        """When no system prompt is given, no system message should appear."""
        body = _sse_lines([
            _openai_text_chunk("ok"),
            _openai_finish_chunk("stop"),
        ])
        transport, captured = _make_capture_transport(body)

        router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        async for _ in router.stream(_simple_messages(), system=None):
            pass
        await router.close()

        payload = json.loads(captured[0].content.decode())
        messages = payload["messages"]
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 5. TestAnthropicStreamText
# ═══════════════════════════════════════════════════════════════════════════


class TestAnthropicStreamText:
    """Mock Anthropic SSE with content_block_delta text_delta -> collect text."""

    async def test_single_text_delta(self):
        """A single Anthropic text delta yields a content chunk."""
        body = _sse_lines_anthropic([
            _anthropic_message_start(input_tokens=10),
            _anthropic_text_delta("Hello from Claude"),
            _anthropic_message_stop(),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.CLAUDE, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        chunks = []
        async for chunk in router.stream(
            _simple_messages(),
            provider_override=Provider.CLAUDE,
        ):
            chunks.append(chunk)
        await router.close()

        text_chunks = [c for c in chunks if c.content]
        assert len(text_chunks) == 1
        assert text_chunks[0].content == "Hello from Claude"

    async def test_multiple_text_deltas(self):
        """Multiple Anthropic text deltas concatenate properly."""
        body = _sse_lines_anthropic([
            _anthropic_message_start(input_tokens=5),
            _anthropic_text_delta("Part 1 "),
            _anthropic_text_delta("Part 2 "),
            _anthropic_text_delta("Part 3"),
            _anthropic_message_stop(),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.CLAUDE, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        collected = ""
        async for chunk in router.stream(
            _simple_messages(),
            provider_override=Provider.CLAUDE,
        ):
            if chunk.content:
                collected += chunk.content
        await router.close()

        assert collected == "Part 1 Part 2 Part 3"

    async def test_done_chunk_on_message_stop(self):
        """message_stop event yields a done=True chunk."""
        body = _sse_lines_anthropic([
            _anthropic_message_start(input_tokens=1),
            _anthropic_text_delta("x"),
            _anthropic_message_stop(),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.CLAUDE, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        chunks = []
        async for chunk in router.stream(
            _simple_messages(),
            provider_override=Provider.CLAUDE,
        ):
            chunks.append(chunk)
        await router.close()

        assert any(c.done for c in chunks)

    async def test_model_name_in_anthropic_chunks(self):
        """Anthropic chunks carry the configured model name."""
        body = _sse_lines_anthropic([
            _anthropic_message_start(input_tokens=1),
            _anthropic_text_delta("hi"),
            _anthropic_message_stop(),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.CLAUDE, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        models = set()
        async for chunk in router.stream(
            _simple_messages(),
            provider_override=Provider.CLAUDE,
        ):
            if chunk.model:
                models.add(chunk.model)
        await router.close()

        assert "claude-sonnet-4-20250514" in models


# ═══════════════════════════════════════════════════════════════════════════
# 6. TestAnthropicStreamToolCall
# ═══════════════════════════════════════════════════════════════════════════


class TestAnthropicStreamToolCall:
    """Mock Anthropic SSE with tool_use content_block -> collect tool call."""

    async def test_single_tool_call(self):
        """Tool use block start + input_json_delta + block stop yields complete tool call."""
        body = _sse_lines_anthropic([
            _anthropic_message_start(input_tokens=20),
            _anthropic_tool_start("toolu_01XYZ", "run_command"),
            _anthropic_tool_delta('{"comma'),
            _anthropic_tool_delta('nd": "ls"}'),
            _anthropic_content_block_stop(),
            _anthropic_message_stop(),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.CLAUDE, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        tool_chunks = []
        async for chunk in router.stream(
            _simple_messages(),
            provider_override=Provider.CLAUDE,
        ):
            if chunk.tool_call:
                tool_chunks.append(chunk)
        await router.close()

        assert len(tool_chunks) == 1
        tc = tool_chunks[0].tool_call
        assert tc["id"] == "toolu_01XYZ"
        assert tc["name"] == "run_command"
        assert tc["arguments"] == {"command": "ls"}

    async def test_tool_call_with_empty_arguments(self):
        """Tool use block with no JSON deltas yields empty dict arguments."""
        body = _sse_lines_anthropic([
            _anthropic_message_start(input_tokens=5),
            _anthropic_tool_start("toolu_empty", "list_items"),
            _anthropic_content_block_stop(),
            _anthropic_message_stop(),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.CLAUDE, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        tool_chunks = []
        async for chunk in router.stream(
            _simple_messages(),
            provider_override=Provider.CLAUDE,
        ):
            if chunk.tool_call:
                tool_chunks.append(chunk)
        await router.close()

        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_call["arguments"] == {}

    async def test_text_then_tool_call(self):
        """Text content followed by a tool call are both captured in order."""
        body = _sse_lines_anthropic([
            _anthropic_message_start(input_tokens=10),
            _anthropic_text_delta("Let me check that."),
            _anthropic_content_block_stop(),
            _anthropic_tool_start("toolu_mixed", "search"),
            _anthropic_tool_delta('{"query": "test"}'),
            _anthropic_content_block_stop(),
            _anthropic_message_stop(),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.CLAUDE, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        text_chunks = []
        tool_chunks = []
        async for chunk in router.stream(
            _simple_messages(),
            provider_override=Provider.CLAUDE,
        ):
            if chunk.content:
                text_chunks.append(chunk)
            if chunk.tool_call:
                tool_chunks.append(chunk)
        await router.close()

        assert len(text_chunks) >= 1
        assert text_chunks[0].content == "Let me check that."
        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_call["name"] == "search"


# ═══════════════════════════════════════════════════════════════════════════
# 7. TestAnthropicMessageStart
# ═══════════════════════════════════════════════════════════════════════════


class TestAnthropicMessageStart:
    """Mock Anthropic SSE with message_start containing input_tokens -> verify captured."""

    async def test_input_tokens_from_message_start(self):
        """message_start event carries input_tokens to StreamChunk."""
        body = _sse_lines_anthropic([
            _anthropic_message_start(input_tokens=512),
            _anthropic_text_delta("ok"),
            _anthropic_message_stop(),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.CLAUDE, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        input_token_chunks = []
        async for chunk in router.stream(
            _simple_messages(),
            provider_override=Provider.CLAUDE,
        ):
            if chunk.input_tokens > 0:
                input_token_chunks.append(chunk)
        await router.close()

        assert len(input_token_chunks) >= 1
        assert input_token_chunks[0].input_tokens == 512

    async def test_cache_read_tokens_from_message_start(self):
        """message_start carries cache_read_input_tokens."""
        body = _sse_lines_anthropic([
            _anthropic_message_start(input_tokens=100, cache_read=300),
            _anthropic_text_delta("cached"),
            _anthropic_message_stop(),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.CLAUDE, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        chunks = []
        async for chunk in router.stream(
            _simple_messages(),
            provider_override=Provider.CLAUDE,
        ):
            if chunk.cache_read_tokens > 0:
                chunks.append(chunk)
        await router.close()

        assert len(chunks) >= 1
        assert chunks[0].cache_read_tokens == 300

    async def test_cache_creation_tokens_from_message_start(self):
        """message_start carries cache_creation_input_tokens."""
        body = _sse_lines_anthropic([
            _anthropic_message_start(input_tokens=50, cache_creation=200),
            _anthropic_text_delta("new cache"),
            _anthropic_message_stop(),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.CLAUDE, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        chunks = []
        async for chunk in router.stream(
            _simple_messages(),
            provider_override=Provider.CLAUDE,
        ):
            if chunk.cache_creation_tokens > 0:
                chunks.append(chunk)
        await router.close()

        assert len(chunks) >= 1
        assert chunks[0].cache_creation_tokens == 200

    async def test_all_usage_fields_together(self):
        """All three usage fields from message_start appear on the same chunk."""
        body = _sse_lines_anthropic([
            _anthropic_message_start(input_tokens=100, cache_read=50, cache_creation=25),
            _anthropic_text_delta("x"),
            _anthropic_message_stop(),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.CLAUDE, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        chunks = []
        async for chunk in router.stream(
            _simple_messages(),
            provider_override=Provider.CLAUDE,
        ):
            if chunk.input_tokens > 0:
                chunks.append(chunk)
        await router.close()

        assert len(chunks) >= 1
        c = chunks[0]
        assert c.input_tokens == 100
        assert c.cache_read_tokens == 50
        assert c.cache_creation_tokens == 25


# ═══════════════════════════════════════════════════════════════════════════
# 8. TestAnthropicMessageDelta
# ═══════════════════════════════════════════════════════════════════════════


class TestAnthropicMessageDelta:
    """Mock message_delta with output_tokens -> verify."""

    async def test_output_tokens_from_message_delta(self):
        """message_delta event carries output_tokens to StreamChunk."""
        body = _sse_lines_anthropic([
            _anthropic_message_start(input_tokens=10),
            _anthropic_text_delta("response"),
            _anthropic_message_delta(output_tokens=75),
            _anthropic_message_stop(),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.CLAUDE, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        output_chunks = []
        async for chunk in router.stream(
            _simple_messages(),
            provider_override=Provider.CLAUDE,
        ):
            if chunk.output_tokens > 0:
                output_chunks.append(chunk)
        await router.close()

        assert len(output_chunks) >= 1
        assert output_chunks[0].output_tokens == 75

    async def test_message_delta_cache_tokens(self):
        """message_delta may also carry cache token counts."""
        body = _sse_lines_anthropic([
            _anthropic_message_start(input_tokens=10),
            _anthropic_text_delta("x"),
            _anthropic_message_delta(output_tokens=30, cache_read=15, cache_creation=5),
            _anthropic_message_stop(),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.CLAUDE, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        delta_chunks = []
        async for chunk in router.stream(
            _simple_messages(),
            provider_override=Provider.CLAUDE,
        ):
            if chunk.output_tokens > 0:
                delta_chunks.append(chunk)
        await router.close()

        assert len(delta_chunks) >= 1
        c = delta_chunks[0]
        assert c.output_tokens == 30
        assert c.cache_read_tokens == 15
        assert c.cache_creation_tokens == 5

    async def test_both_message_start_and_delta_usage(self):
        """Both message_start (input) and message_delta (output) usage are captured."""
        body = _sse_lines_anthropic([
            _anthropic_message_start(input_tokens=200),
            _anthropic_text_delta("analysis"),
            _anthropic_message_delta(output_tokens=150),
            _anthropic_message_stop(),
        ])
        transport = _make_mock_transport(body)

        router = LLMRouter(primary=Provider.CLAUDE, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        all_chunks = []
        async for chunk in router.stream(
            _simple_messages(),
            provider_override=Provider.CLAUDE,
        ):
            all_chunks.append(chunk)
        await router.close()

        input_chunks = [c for c in all_chunks if c.input_tokens > 0]
        output_chunks = [c for c in all_chunks if c.output_tokens > 0]
        assert len(input_chunks) >= 1
        assert input_chunks[0].input_tokens == 200
        assert len(output_chunks) >= 1
        assert output_chunks[0].output_tokens == 150


# ═══════════════════════════════════════════════════════════════════════════
# 9. TestProviderFallback
# ═══════════════════════════════════════════════════════════════════════════


class TestProviderFallback:
    """Primary fails -> fallback is tried."""

    async def test_fallback_on_primary_failure(self):
        """When primary provider returns 401, fallback provider is used."""
        # The primary (DeepSeek) will fail with 401, the fallback (Claude/Anthropic)
        # should succeed. We mock at the _stream_provider level to control per-provider behavior.
        success_body = _sse_lines_anthropic([
            _anthropic_message_start(input_tokens=5),
            _anthropic_text_delta("fallback response"),
            _anthropic_message_stop(),
        ])

        call_log: list[Provider] = []
        original_stream_provider = LLMRouter._stream_provider

        async def mock_stream_provider(self, provider, messages, tools, system):
            call_log.append(provider)
            if provider == Provider.DEEPSEEK:
                # Simulate a non-retryable error (401) from primary
                request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
                response = httpx.Response(401, request=request)
                raise httpx.HTTPStatusError("Unauthorized", request=request, response=response)
            else:
                # Fallback provider (Claude) succeeds
                transport = _make_mock_transport(success_body)
                client = httpx.AsyncClient(transport=transport)
                self._client = client
                async for chunk in original_stream_provider(self, provider, messages, tools, system):
                    yield chunk

        with patch.object(LLMRouter, "_stream_provider", mock_stream_provider):
            router = LLMRouter(primary=Provider.DEEPSEEK, fallback=Provider.CLAUDE)
            router._client = httpx.AsyncClient()  # will be replaced in mock

            chunks = []
            async for chunk in router.stream(_simple_messages()):
                chunks.append(chunk)
            await router.close()

        # Primary was tried first, then fallback
        assert Provider.DEEPSEEK in call_log
        assert Provider.CLAUDE in call_log
        assert call_log.index(Provider.DEEPSEEK) < call_log.index(Provider.CLAUDE)

        # Fallback actually produced content
        text_chunks = [c for c in chunks if c.content]
        assert len(text_chunks) >= 1
        assert text_chunks[0].content == "fallback response"

    async def test_all_providers_fail_raises(self):
        """When all providers fail, an exception is raised."""

        async def mock_stream_provider(self, provider, messages, tools, system):
            request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
            response = httpx.Response(401, request=request)
            raise httpx.HTTPStatusError("Unauthorized", request=request, response=response)
            # Make this an async generator
            yield  # pragma: no cover

        with patch.object(LLMRouter, "_stream_provider", mock_stream_provider):
            router = LLMRouter(primary=Provider.DEEPSEEK, fallback=Provider.CLAUDE)

            with pytest.raises(Exception, match="All providers failed"):
                async for _ in router.stream(_simple_messages()):
                    pass
            await router.close()

    async def test_no_fallback_raises_immediately(self):
        """When there is no fallback and primary fails, exception is raised."""

        async def mock_stream_provider(self, provider, messages, tools, system):
            request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
            response = httpx.Response(403, request=request)
            raise httpx.HTTPStatusError("Forbidden", request=request, response=response)
            yield  # pragma: no cover

        with patch.object(LLMRouter, "_stream_provider", mock_stream_provider):
            router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)

            with pytest.raises(Exception, match="All providers failed"):
                async for _ in router.stream(_simple_messages()):
                    pass
            await router.close()


# ═══════════════════════════════════════════════════════════════════════════
# 10. TestRetryOnTransientError
# ═══════════════════════════════════════════════════════════════════════════


class TestRetryOnTransientError:
    """429/500 -> retry with backoff."""

    async def test_retry_on_429(self):
        """A 429 response triggers a retry that eventually succeeds."""
        success_body = _sse_lines([
            _openai_text_chunk("success after retry"),
            _openai_finish_chunk("stop"),
        ])

        attempt_count = {"n": 0}

        async def mock_stream_provider(self, provider, messages, tools, system):
            attempt_count["n"] += 1
            if attempt_count["n"] <= 2:
                request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
                response = httpx.Response(429, request=request)
                raise httpx.HTTPStatusError("Rate limited", request=request, response=response)
            # Third attempt succeeds
            transport = _make_mock_transport(success_body)
            client = httpx.AsyncClient(transport=transport)
            self._client = client
            async for chunk in LLMRouter._stream_openai(self, provider, messages, tools, system):
                yield chunk

        with (
            patch.object(LLMRouter, "_stream_provider", mock_stream_provider),
            patch("omnigent.router.asyncio.sleep", new_callable=AsyncMock),
        ):
            router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)
            router._client = httpx.AsyncClient()

            chunks = []
            async for chunk in router.stream(_simple_messages()):
                chunks.append(chunk)
            await router.close()

        assert attempt_count["n"] == 3
        text_chunks = [c for c in chunks if c.content]
        assert any("success after retry" in c.content for c in text_chunks)

    async def test_retry_on_500(self):
        """A 500 response triggers a retry that eventually succeeds."""
        success_body = _sse_lines([
            _openai_text_chunk("recovered"),
            _openai_finish_chunk("stop"),
        ])

        attempt_count = {"n": 0}

        async def mock_stream_provider(self, provider, messages, tools, system):
            attempt_count["n"] += 1
            if attempt_count["n"] == 1:
                request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
                response = httpx.Response(500, request=request)
                raise httpx.HTTPStatusError("Internal Server Error", request=request, response=response)
            transport = _make_mock_transport(success_body)
            client = httpx.AsyncClient(transport=transport)
            self._client = client
            async for chunk in LLMRouter._stream_openai(self, provider, messages, tools, system):
                yield chunk

        with (
            patch.object(LLMRouter, "_stream_provider", mock_stream_provider),
            patch("omnigent.router.asyncio.sleep", new_callable=AsyncMock),
        ):
            router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)
            router._client = httpx.AsyncClient()

            chunks = []
            async for chunk in router.stream(_simple_messages()):
                chunks.append(chunk)
            await router.close()

        assert attempt_count["n"] == 2
        text = "".join(c.content for c in chunks if c.content)
        assert "recovered" in text

    async def test_max_retries_exceeded_raises(self):
        """When all retries are exhausted, the error propagates."""

        async def mock_stream_provider(self, provider, messages, tools, system):
            request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
            response = httpx.Response(429, request=request)
            raise httpx.HTTPStatusError("Rate limited", request=request, response=response)
            yield  # pragma: no cover

        with (
            patch.object(LLMRouter, "_stream_provider", mock_stream_provider),
            patch("omnigent.router.asyncio.sleep", new_callable=AsyncMock),
        ):
            router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)

            with pytest.raises(Exception, match="All providers failed|Rate limited"):
                async for _ in router.stream(_simple_messages()):
                    pass
            await router.close()

    async def test_non_retryable_error_not_retried(self):
        """A 400 error is not retryable and should propagate immediately."""
        attempt_count = {"n": 0}

        async def mock_stream_provider(self, provider, messages, tools, system):
            attempt_count["n"] += 1
            request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
            response = httpx.Response(400, request=request)
            raise httpx.HTTPStatusError("Bad Request", request=request, response=response)
            yield  # pragma: no cover

        with (
            patch.object(LLMRouter, "_stream_provider", mock_stream_provider),
            patch("omnigent.router.asyncio.sleep", new_callable=AsyncMock),
        ):
            # Pass system=None to avoid the 400 retry-without-system logic in stream()
            # We also set fallback=None so only the primary is tried
            router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)

            with pytest.raises(Exception, match="All providers failed|Bad Request"):
                async for _ in router.stream(_simple_messages(), system=None):
                    pass
            await router.close()

        # Should only have been called once (no retry for 400)
        assert attempt_count["n"] == 1

    async def test_retry_sleeps_with_backoff(self):
        """Verify that asyncio.sleep is called with exponential backoff delays."""
        sleep_calls: list[float] = []
        original_sleep = AsyncMock(side_effect=lambda d: sleep_calls.append(d))

        attempt_count = {"n": 0}

        async def mock_stream_provider(self, provider, messages, tools, system):
            attempt_count["n"] += 1
            if attempt_count["n"] <= 3:
                request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
                response = httpx.Response(503, request=request)
                raise httpx.HTTPStatusError("Service Unavailable", request=request, response=response)
            # Fourth attempt (attempt index 3) succeeds
            body = _sse_lines([
                _openai_text_chunk("ok"),
                _openai_finish_chunk("stop"),
            ])
            transport = _make_mock_transport(body)
            self._client = httpx.AsyncClient(transport=transport)
            async for chunk in LLMRouter._stream_openai(self, provider, messages, tools, system):
                yield chunk

        with (
            patch.object(LLMRouter, "_stream_provider", mock_stream_provider),
            patch("omnigent.router.asyncio.sleep", original_sleep),
        ):
            router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)
            router._client = httpx.AsyncClient()

            async for _ in router.stream(_simple_messages()):
                pass
            await router.close()

        # 3 failures = 3 sleep calls (attempts 0, 1, 2)
        assert len(sleep_calls) == 3
        # Delays should be roughly: 1*2^0+jitter, 1*2^1+jitter, 1*2^2+jitter
        assert sleep_calls[0] >= 1.0  # base_delay * 2^0 = 1.0
        assert sleep_calls[1] >= 2.0  # base_delay * 2^1 = 2.0
        assert sleep_calls[2] >= 4.0  # base_delay * 2^2 = 4.0

    async def test_connection_error_retried(self):
        """httpx.ConnectError is also retried."""
        attempt_count = {"n": 0}

        async def mock_stream_provider(self, provider, messages, tools, system):
            attempt_count["n"] += 1
            if attempt_count["n"] == 1:
                raise httpx.ConnectError("Connection refused")
            body = _sse_lines([
                _openai_text_chunk("reconnected"),
                _openai_finish_chunk("stop"),
            ])
            transport = _make_mock_transport(body)
            self._client = httpx.AsyncClient(transport=transport)
            async for chunk in LLMRouter._stream_openai(self, provider, messages, tools, system):
                yield chunk

        with (
            patch.object(LLMRouter, "_stream_provider", mock_stream_provider),
            patch("omnigent.router.asyncio.sleep", new_callable=AsyncMock),
        ):
            router = LLMRouter(primary=Provider.DEEPSEEK, fallback=None)
            router._client = httpx.AsyncClient()

            text = ""
            async for chunk in router.stream(_simple_messages()):
                if chunk.content:
                    text += chunk.content
            await router.close()

        assert attempt_count["n"] == 2
        assert "reconnected" in text


# ═══════════════════════════════════════════════════════════════════════════
# Bonus: TestAnthropicSystemPromptPayload
# ═══════════════════════════════════════════════════════════════════════════


class TestAnthropicSystemPromptPayload:
    """Verify Anthropic system prompt is sent in the payload's 'system' field."""

    async def test_system_string_sent_as_cacheable_list(self):
        """A string system prompt is wrapped in cacheable format for Anthropic."""
        body = _sse_lines_anthropic([
            _anthropic_message_start(input_tokens=5),
            _anthropic_text_delta("ok"),
            _anthropic_message_stop(),
        ])
        transport, captured = _make_capture_transport(body)

        router = LLMRouter(primary=Provider.CLAUDE, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        async for _ in router.stream(
            _simple_messages(),
            system="You are a security expert.",
            provider_override=Provider.CLAUDE,
        ):
            pass
        await router.close()

        assert len(captured) == 1
        payload = json.loads(captured[0].content.decode())

        # System should be in the payload as a list with cache_control
        assert "system" in payload
        system_val = payload["system"]
        assert isinstance(system_val, list)
        assert len(system_val) == 1
        assert system_val[0]["type"] == "text"
        assert system_val[0]["text"] == "You are a security expert."
        assert system_val[0]["cache_control"] == {"type": "ephemeral"}

    async def test_system_list_sent_directly(self):
        """A list-format system prompt is passed through unchanged."""
        body = _sse_lines_anthropic([
            _anthropic_message_start(input_tokens=5),
            _anthropic_text_delta("ok"),
            _anthropic_message_stop(),
        ])
        transport, captured = _make_capture_transport(body)

        router = LLMRouter(primary=Provider.CLAUDE, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        system_blocks = [
            {"type": "text", "text": "Block 1", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Block 2"},
        ]
        async for _ in router.stream(
            _simple_messages(),
            system=system_blocks,
            provider_override=Provider.CLAUDE,
        ):
            pass
        await router.close()

        payload = json.loads(captured[0].content.decode())
        assert payload["system"] == system_blocks

    async def test_no_system_no_system_key(self):
        """When no system prompt is given, the payload should not have a 'system' key."""
        body = _sse_lines_anthropic([
            _anthropic_message_start(input_tokens=1),
            _anthropic_text_delta("ok"),
            _anthropic_message_stop(),
        ])
        transport, captured = _make_capture_transport(body)

        router = LLMRouter(primary=Provider.CLAUDE, fallback=None)
        router._client = httpx.AsyncClient(transport=transport)

        async for _ in router.stream(
            _simple_messages(),
            system=None,
            provider_override=Provider.CLAUDE,
        ):
            pass
        await router.close()

        payload = json.loads(captured[0].content.decode())
        assert "system" not in payload
