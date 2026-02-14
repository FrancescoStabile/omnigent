"""
Omnigent — LLM Router

Multi-provider LLM routing with streaming, fallback, task-based selection, and cost tracking.

Supports: DeepSeek, Claude, OpenAI, Ollama (local)
Task-based routing: SELECT best provider per task type.

This module is 100% domain-agnostic — no domain-specific logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("omnigent.router")


# ═══════════════════════════════════════════════════════════════════════════
# Provider Protocol — extend to add new LLM providers without modifying router
# ═══════════════════════════════════════════════════════════════════════════


class LLMProvider:
    """Abstract base for LLM providers.

    Subclass this to add a new provider (e.g., Gemini, Mistral, Groq)
    without modifying LLMRouter. Register it via LLMRouter.register_provider().
    """

    async def stream(
        self,
        client: httpx.AsyncClient,
        config: dict,
        messages: list[dict],
        tools: list[dict] | None,
        system: str | list[dict] | None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a response from this provider.

        Args:
            client: Shared httpx.AsyncClient
            config: Provider config dict (base_url, model, api_key_env, etc.)
            messages: Conversation messages
            tools: Tool schemas (OpenAI format)
            system: System prompt

        Yields:
            StreamChunk objects
        """
        raise NotImplementedError
        # Make this an async generator
        yield  # pragma: no cover


# ═══════════════════════════════════════════════════════════════════════════
# Security Helper
# ═══════════════════════════════════════════════════════════════════════════


def redact_api_keys(text: str) -> str:
    """Redact API keys from error messages for security."""
    import re
    patterns = [
        (r'sk-[a-zA-Z0-9]{20,}', 'sk-***REDACTED***'),
        (r'sk-ant-[a-zA-Z0-9-]{40,}', 'sk-ant-***REDACTED***'),
        (r'Bearer\s+[a-zA-Z0-9_-]+', 'Bearer ***REDACTED***'),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return text


# ═══════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════


class Provider(str, Enum):
    """Supported LLM providers."""
    DEEPSEEK = "deepseek"
    CLAUDE = "claude"
    OPENAI = "openai"
    LOCAL = "local"  # Ollama


class TaskType(str, Enum):
    """Task types for smart routing.

    Extend this enum in your domain to add task-specific routing.
    """
    PLANNING = "planning"       # Plan generation
    TOOL_USE = "tool_use"       # Standard tool-calling iteration
    ANALYSIS = "analysis"       # Deep analysis of results
    REFLECTION = "reflection"   # Self-evaluation
    REPORT = "report"           # Report writing


@dataclass
class StreamChunk:
    """Single streaming chunk from LLM."""
    content: str | None = None
    tool_call: dict | None = None  # {id, name, arguments}
    done: bool = False
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Provider Configs
# ═══════════════════════════════════════════════════════════════════════════


PROVIDERS = {
    Provider.DEEPSEEK: {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "cost_per_1k_in": 0.00014,
        "cost_per_1k_out": 0.00028,
    },
    Provider.CLAUDE: {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-20250514",
        "api_key_env": "ANTHROPIC_API_KEY",
        "cost_per_1k_in": 0.003,
        "cost_per_1k_out": 0.015,
    },
    Provider.OPENAI: {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "cost_per_1k_in": 0.00015,
        "cost_per_1k_out": 0.0006,
    },
    Provider.LOCAL: {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5-coder:3b-instruct",
        "api_key_env": "",
        "cost_per_1k_in": 0.0,
        "cost_per_1k_out": 0.0,
    },
}


# Task-based routing rules: task_type → preferred provider (if available)
TASK_ROUTING: dict[TaskType, list[Provider]] = {
    TaskType.PLANNING: [Provider.CLAUDE, Provider.DEEPSEEK],
    TaskType.TOOL_USE: [Provider.DEEPSEEK, Provider.OPENAI],
    TaskType.ANALYSIS: [Provider.CLAUDE, Provider.DEEPSEEK],
    TaskType.REFLECTION: [Provider.DEEPSEEK, Provider.LOCAL],
    TaskType.REPORT: [Provider.CLAUDE, Provider.OPENAI],
}


# ═══════════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════════


class LLMRouter:
    """
    Multi-provider LLM router with streaming, fallback, and task-based routing.

    Usage:
        router = LLMRouter(primary=Provider.DEEPSEEK)

        async for chunk in router.stream(messages, tools, system, task_type=TaskType.TOOL_USE):
            print(chunk.content, end="", flush=True)
    """

    def __init__(
        self,
        primary: Provider = Provider.DEEPSEEK,
        fallback: Provider | None = Provider.CLAUDE,
    ):
        self.primary = primary
        if fallback and not self._get_api_key(fallback):
            logger.warning(f"No API key for {fallback.value}, disabling fallback")
            self.fallback = None
        else:
            self.fallback = fallback
        self._client: httpx.AsyncClient | None = None
        self.current_provider: Provider = primary
        # Custom provider implementations (override built-in streaming)
        self._custom_providers: dict[Provider, LLMProvider] = {}

    def register_provider(self, provider: Provider, impl: LLMProvider) -> None:
        """Register a custom LLMProvider implementation for a provider.

        This allows adding new providers (e.g., Gemini, Mistral) or replacing
        the built-in OpenAI/Anthropic implementations without modifying router.py.
        """
        self._custom_providers[provider] = impl
        logger.info(f"Registered custom provider: {provider.value} → {type(impl).__name__}")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=180.0)
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_api_key(self, provider: Provider) -> str | None:
        """Get API key for provider."""
        config = PROVIDERS[provider]
        env_var = config["api_key_env"]
        if not env_var:
            return None
        return os.environ.get(env_var)

    def select_provider(self, task_type: TaskType | None = None) -> Provider:
        """Select best provider for a task type."""
        if task_type and task_type in TASK_ROUTING:
            for provider in TASK_ROUTING[task_type]:
                if self._get_api_key(provider):
                    return provider
        return self.primary

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | list[dict] | None = None,
        task_type: TaskType | None = None,
        provider_override: Provider | None = None,
        thinking_budget: int | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream LLM response with real-time tokens.

        Args:
            messages: Conversation messages
            tools: Tool schemas (OpenAI format)
            system: System prompt (string or Anthropic cacheable format)
            task_type: Optional task type for smart routing
            provider_override: Force a specific provider
            thinking_budget: Optional extended thinking budget (tokens) for Claude.
                If None, uses automatic budget for PLANNING/ANALYSIS tasks with Claude.
                Set to 0 to disable thinking.
        """
        self._current_thinking_budget = thinking_budget
        self._current_task_type = task_type

        if provider_override:
            selected = provider_override
        elif task_type:
            selected = self.select_provider(task_type)
        else:
            selected = self.primary

        self.current_provider = selected

        providers = [selected]
        if self.fallback and self.fallback != selected:
            providers.append(self.fallback)
        elif self.primary != selected:
            providers.append(self.primary)

        for provider in providers:
            try:
                async for chunk in self._stream_with_retry(provider, messages, tools, system):
                    yield chunk
                return
            except Exception as e:
                error_msg = redact_api_keys(str(e))

                if "400" in error_msg and provider == self.primary and system:
                    logger.warning(f"Got 400 from {provider}, retrying without system prompt...")
                    try:
                        async for chunk in self._stream_with_retry(provider, messages, tools, None):
                            yield chunk
                        return
                    except Exception as e2:
                        logger.error(f"Retry also failed: {redact_api_keys(str(e2))}")

                logger.error(f"Streaming failed with {provider}: {error_msg}", exc_info=True)
                if provider == providers[-1]:
                    raise Exception(f"All providers failed. Last error: {error_msg}")

    async def _stream_with_retry(
        self,
        provider: Provider,
        messages: list[dict],
        tools: list[dict] | None,
        system: str | list[dict] | None,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream from provider with exponential backoff retry on transient errors."""
        retryable_codes = {429, 500, 502, 503, 504}

        for attempt in range(max_retries + 1):
            try:
                async for chunk in self._stream_provider(provider, messages, tools, system):
                    yield chunk
                return
            except httpx.HTTPStatusError as e:
                if e.response.status_code not in retryable_codes or attempt == max_retries:
                    raise
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Retryable error {e.response.status_code} from {provider.value}, "
                    f"retry {attempt + 1}/{max_retries} in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                if attempt == max_retries:
                    raise
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Connection error from {provider.value}: {e}, "
                    f"retry {attempt + 1}/{max_retries} in {delay:.1f}s"
                )
                await asyncio.sleep(delay)

    async def _stream_provider(
        self,
        provider: Provider,
        messages: list[dict],
        tools: list[dict] | None,
        system: str | list[dict] | None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream from specific provider.

        Checks for a custom LLMProvider first, then falls back to
        built-in OpenAI/Anthropic implementations.
        """
        if provider in self._custom_providers:
            client = await self._get_client()
            config = PROVIDERS.get(provider, {})
            async for chunk in self._custom_providers[provider].stream(
                client, config, messages, tools, system
            ):
                yield chunk
        elif provider == Provider.CLAUDE:
            async for chunk in self._stream_anthropic(provider, messages, tools, system):
                yield chunk
        else:
            async for chunk in self._stream_openai(provider, messages, tools, system):
                yield chunk

    async def _stream_openai(
        self,
        provider: Provider,
        messages: list[dict],
        tools: list[dict] | None,
        system: str | list[dict] | None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream from OpenAI-compatible API (DeepSeek, OpenAI, Ollama)."""
        client = await self._get_client()
        config = PROVIDERS[provider]

        headers = {"Content-Type": "application/json"}
        api_key = self._get_api_key(provider)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Extract system text
        system_text = None
        if system:
            if isinstance(system, list):
                if system and len(system) > 0 and "text" in system[0]:
                    system_text = system[0]["text"]
            else:
                system_text = system

        # Build messages — always use native system message role
        all_messages = []
        if system_text:
            all_messages.append({"role": "system", "content": system_text})
        all_messages.extend(messages)

        # Normalize messages for OpenAI format
        normalized_messages = self._normalize_messages_openai(all_messages)

        payload = {
            "model": config["model"],
            "messages": normalized_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 4096,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # Validate message sequence before sending
        self._validate_message_sequence(normalized_messages)

        tool_calls_buffer = {}

        async with client.stream(
            "POST",
            f"{config['base_url']}/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                error_body = await response.aread()
                error_msg = redact_api_keys(error_body.decode())
                logger.error(f"Provider {provider} HTTP {e.response.status_code}: {error_msg}")
                raise

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    continue

                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                usage = data.get("usage")
                if usage:
                    yield StreamChunk(
                        model=config["model"],
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                    )

                choice = data.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                finish = choice.get("finish_reason")

                if delta.get("content"):
                    yield StreamChunk(content=delta["content"], model=config["model"])

                if delta.get("tool_calls"):
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc.get("id"):
                            tool_calls_buffer[idx]["id"] = tc["id"]
                        if tc.get("function", {}).get("name"):
                            tool_calls_buffer[idx]["name"] = tc["function"]["name"]
                        if tc.get("function", {}).get("arguments"):
                            tool_calls_buffer[idx]["arguments"] += tc["function"]["arguments"]

                if finish:
                    for tc in tool_calls_buffer.values():
                        try:
                            args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning(
                                f"Malformed JSON in tool arguments for '{tc.get('name', '?')}': {e}. "
                                f"Raw: {tc['arguments'][:200]}"
                            )
                            args = {}
                        yield StreamChunk(
                            tool_call={"id": tc["id"], "name": tc["name"], "arguments": args},
                            model=config["model"]
                        )
                    yield StreamChunk(done=True, model=config["model"])
                    return

    async def _stream_anthropic(
        self,
        provider: Provider,
        messages: list[dict],
        tools: list[dict] | None,
        system: str | list[dict] | None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream from Anthropic Claude."""
        client = await self._get_client()
        config = PROVIDERS[provider]

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._get_api_key(provider) or "",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
        }

        # Determine thinking budget
        thinking_budget = getattr(self, "_current_thinking_budget", None)
        task_type = getattr(self, "_current_task_type", None)

        # Auto-enable thinking for planning/analysis tasks if not explicitly disabled
        if thinking_budget is None and task_type in (TaskType.PLANNING, TaskType.ANALYSIS):
            thinking_budget = 10000  # Default thinking budget for complex tasks

        payload = {
            "model": config["model"],
            "max_tokens": 16384 if thinking_budget else 4096,
            "messages": messages,
            "stream": True,
        }

        # Extended thinking support
        if thinking_budget and thinking_budget > 0:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

        if system:
            if isinstance(system, list):
                payload["system"] = system
            else:
                payload["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]

        if tools:
            payload["tools"] = self._convert_tools_anthropic(tools)

        current_tool = None

        async with client.stream(
            "POST",
            f"{config['base_url']}/messages",
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                event_type = data.get("type")

                if event_type == "message_start":
                    message = data.get("message", {})
                    usage = message.get("usage", {})
                    if usage:
                        yield StreamChunk(
                            model=config["model"],
                            input_tokens=usage.get("input_tokens", 0),
                            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                            cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
                        )

                elif event_type == "content_block_start":
                    block = data.get("content_block", {})
                    if block.get("type") == "tool_use":
                        current_tool = {
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "arguments": ""
                        }
                    elif block.get("type") == "thinking":
                        # Extended thinking block — track but don't emit to user
                        pass

                elif event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield StreamChunk(content=delta.get("text"), model=config["model"])
                    elif delta.get("type") == "input_json_delta" and current_tool:
                        current_tool["arguments"] += delta.get("partial_json", "")
                    elif delta.get("type") == "thinking_delta":
                        # Extended thinking tokens — don't emit to user
                        pass

                elif event_type == "content_block_stop":
                    if current_tool:
                        try:
                            args = json.loads(current_tool["arguments"]) if current_tool["arguments"] else {}
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning(
                                f"Malformed JSON in tool arguments for '{current_tool.get('name', '?')}': {e}. "
                                f"Raw: {current_tool['arguments'][:200]}"
                            )
                            args = {}
                        yield StreamChunk(
                            tool_call={"id": current_tool["id"], "name": current_tool["name"], "arguments": args},
                            model=config["model"]
                        )
                        current_tool = None

                elif event_type == "message_delta":
                    usage = data.get("usage", {})
                    if usage:
                        yield StreamChunk(
                            model=config["model"],
                            input_tokens=usage.get("input_tokens", 0),
                            output_tokens=usage.get("output_tokens", 0),
                            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                            cache_creation_tokens=usage.get("cache_creation_input_tokens", 0)
                        )

                elif event_type == "message_stop":
                    yield StreamChunk(done=True, model=config["model"])
                    return

    def _normalize_messages_openai(self, messages: list[dict]) -> list[dict]:
        """Normalize messages to OpenAI wire format."""
        normalized = []
        for msg in messages:
            if msg["role"] == "tool":
                content_data = msg.get("content", {})
                if isinstance(content_data, dict):
                    tool_call_id = content_data.get("tool_call_id", "call_unknown")
                    content_str = content_data.get("content", "")
                    if isinstance(content_str, dict):
                        content_str = json.dumps(content_str)
                    normalized.append({
                        "role": "tool",
                        "content": str(content_str),
                        "tool_call_id": tool_call_id
                    })
                else:
                    normalized.append({
                        "role": "tool",
                        "content": str(content_data),
                        "tool_call_id": "call_" + uuid.uuid4().hex[:8]
                    })
            elif msg["role"] == "assistant" and isinstance(msg.get("content"), list):
                text_content = ""
                tool_calls_list = []
                for item in msg["content"]:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_content += item.get("text", "")
                        elif item.get("type") == "tool_use":
                            tool_calls_list.append({
                                "id": item["id"],
                                "type": "function",
                                "function": {
                                    "name": item["name"],
                                    "arguments": json.dumps(item.get("input", {}))
                                }
                            })
                if tool_calls_list:
                    msg_dict = {"role": "assistant", "tool_calls": tool_calls_list}
                    if text_content:
                        msg_dict["content"] = text_content
                    normalized.append(msg_dict)
                else:
                    normalized.append({"role": "assistant", "content": text_content or ""})
            else:
                normalized.append(msg)
        return normalized

    def _validate_message_sequence(self, messages: list[dict]):
        """Validate assistant→tool message sequence to prevent 400 errors."""
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                expected_ids = {tc["id"] for tc in msg["tool_calls"]}
                actual_ids = set()
                j = i + 1
                while j < len(messages) and messages[j].get("role") == "tool":
                    actual_ids.add(messages[j].get("tool_call_id"))
                    j += 1
                missing = expected_ids - actual_ids
                if missing:
                    logger.error(
                        f"Message sequence validation failed at position {i+1}: "
                        f"assistant expects {len(expected_ids)} tools, got {len(actual_ids)}. "
                        f"Missing: {missing}"
                    )

    def _convert_tools_anthropic(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI tool format to Anthropic format."""
        return [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "input_schema": t["function"].get("parameters", {}),
            }
            for t in tools
        ]
