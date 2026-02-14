"""
Omnigent — Smart Context Management

3-level context strategy:
1. DomainProfile summary → always in system prompt (never trimmed)
2. Middle messages → summarized heuristically (tool results compressed)
3. Recent window → kept intact (last N messages)

Preserves assistant+tool atomic groups to avoid breaking LLM validation.

This module is 100% domain-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Module-level custom tokenizer hook
_custom_tokenizer: Callable[[str], int] | None = None


def set_tokenizer(fn: Callable[[str], int] | None) -> None:
    """Set a custom tokenizer function for accurate token counting.

    Args:
        fn: A function that takes a string and returns token count.
            Pass None to revert to the heuristic estimator.

    Example:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        set_tokenizer(lambda text: len(enc.encode(text)))
    """
    global _custom_tokenizer
    _custom_tokenizer = fn


def trim_context_window(
    messages: list[dict[str, Any]],
    max_messages: int = 20,
    preserve_first: int = 2,
) -> list[dict[str, Any]]:
    """
    Trim conversation history to recent messages.

    CRITICAL: Preserves assistant+tools sequences to avoid breaking API validation.

    Strategy:
    1. Identify all message "groups" (user, assistant, assistant+tools)
    2. Never split an assistant+tools group
    3. Keep first N groups + last M groups within budget
    """
    if len(messages) <= max_messages:
        return messages

    groups = _group_messages(messages)

    if len(groups) <= 2:
        return messages

    first_groups_count = 0
    first_msgs_count = 0
    while first_groups_count < len(groups) and first_msgs_count < preserve_first:
        first_msgs_count += len(groups[first_groups_count])
        first_groups_count += 1

    first_msgs = _flatten_groups(groups[:first_groups_count])

    remaining_budget = max_messages - len(first_msgs)
    if remaining_budget <= 0:
        return first_msgs[:max_messages]

    recent_groups = []
    recent_msgs_count = 0
    for g in reversed(groups[first_groups_count:]):
        if recent_msgs_count + len(g) <= remaining_budget:
            recent_groups.insert(0, g)
            recent_msgs_count += len(g)
        else:
            break

    recent_msgs = _flatten_groups(recent_groups)
    return first_msgs + recent_msgs


def smart_trim_context(
    messages: list[dict[str, Any]],
    max_tokens: int = 80000,
    recent_window: int = 12,
) -> list[dict[str, Any]]:
    """
    Smart context trimming with 3 levels:

    Level 1: DomainProfile + Plan are in system prompt (not in messages)
    Level 2: Middle messages get tool results compressed
    Level 3: Recent window (last N messages) kept intact
    """
    total_tokens = sum(estimate_tokens(msg.get("content", "")) for msg in messages)

    if total_tokens <= max_tokens:
        return messages

    groups = _group_messages(messages)

    if len(groups) <= 3:
        return messages

    recent_group_count = min(recent_window // 2, len(groups) - 2)
    first_groups = groups[:1]
    middle_groups = groups[1:-recent_group_count] if recent_group_count > 0 else groups[1:]
    recent_groups = groups[-recent_group_count:] if recent_group_count > 0 else []

    # Compress middle groups: summarize tool results
    compressed_middle = []
    for group in middle_groups:
        compressed_group = []
        for msg in group:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, dict):
                    tool_content = content.get("content", "")
                    if isinstance(tool_content, str) and len(tool_content) > 500:
                        compressed_content = _summarize_tool_result(tool_content)
                        compressed_msg = dict(msg)
                        compressed_msg["content"] = {
                            "tool_call_id": content.get("tool_call_id", ""),
                            "content": compressed_content,
                        }
                        compressed_group.append(compressed_msg)
                        continue
                elif isinstance(content, str) and len(content) > 500:
                    compressed_msg = dict(msg)
                    compressed_msg["content"] = _summarize_tool_result(content)
                    compressed_group.append(compressed_msg)
                    continue
            compressed_group.append(msg)
        compressed_middle.append(compressed_group)

    result = _flatten_groups(first_groups + compressed_middle + recent_groups)
    result_tokens = sum(estimate_tokens(msg.get("content", "")) for msg in result)

    if result_tokens <= max_tokens:
        return result

    # Still too large — drop middle groups from oldest
    while compressed_middle and result_tokens > max_tokens:
        dropped = compressed_middle.pop(0)
        dropped_tokens = sum(estimate_tokens(msg.get("content", "")) for msg in dropped)
        result_tokens -= dropped_tokens

    return _flatten_groups(first_groups + compressed_middle + recent_groups)


def _group_messages(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group messages into atomic units that can't be split."""
    groups = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")

        if role == "assistant":
            content = msg.get("content", [])
            has_tool_calls = False
            if isinstance(content, list):
                has_tool_calls = any(
                    isinstance(item, dict) and item.get("type") == "tool_use"
                    for item in content
                )
            if has_tool_calls:
                group = [messages[i]]
                j = i + 1
                while j < len(messages) and messages[j].get("role") == "tool":
                    group.append(messages[j])
                    j += 1
                groups.append(group)
                i = j
            else:
                groups.append([messages[i]])
                i += 1
        else:
            groups.append([messages[i]])
            i += 1

    return groups


def _flatten_groups(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Flatten a list of groups into a flat message list."""
    result = []
    for g in groups:
        result.extend(g)
    return result


def _summarize_tool_result(result: str, max_chars: int = 300) -> str:
    """Heuristically summarize a tool result for middle context."""
    if len(result) <= max_chars:
        return result

    if result.strip().startswith("{") or result.strip().startswith("["):
        lines = result.split("\n")
        summary_lines = lines[:8]
        summary = "\n".join(summary_lines)
        if len(summary) > max_chars:
            summary = summary[:max_chars]
        return summary + f"\n[... {len(result) - len(summary)} chars compressed ...]"

    return result[:max_chars] + f"\n[... {len(result) - max_chars} chars compressed ...]"


# ═══════════════════════════════════════════════════════════════════════════
# Semantic Context Compression (LLM-based)
# ═══════════════════════════════════════════════════════════════════════════


async def semantic_compress_messages(
    messages: list[dict[str, Any]],
    router: Any,
    max_tokens: int = 80000,
    recent_window: int = 12,
) -> list[dict[str, Any]]:
    """Compress old messages using LLM-based summarization.

    Unlike heuristic trimming, this uses a cheap/fast LLM to summarize
    old tool results and conversations while preserving key facts.

    Strategy:
    1. Keep recent_window messages intact.
    2. For older messages, group tool results and summarize them via LLM.
    3. Replace the original messages with a single summary message.

    Args:
        messages: Conversation messages.
        router: LLMRouter instance for summarization calls.
        max_tokens: Token budget.
        recent_window: Number of recent messages to preserve intact.

    Returns:
        Compressed message list.
    """
    total_tokens = sum(estimate_tokens(msg.get("content", "")) for msg in messages)
    if total_tokens <= max_tokens:
        return messages

    # Split into old and recent
    groups = _group_messages(messages)
    if len(groups) <= 3:
        return messages

    recent_count = min(recent_window // 2, len(groups) - 1)
    old_groups = groups[:-recent_count] if recent_count > 0 else groups
    recent_groups = groups[-recent_count:] if recent_count > 0 else []

    # Collect old tool results for summarization
    old_content_parts: list[str] = []
    for group in old_groups:
        for msg in group:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                old_content_parts.append(f"[{role}]: {content[:500]}")
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text", "") or item.get("content", "")
                        if text:
                            old_content_parts.append(f"[{role}]: {str(text)[:500]}")

    if not old_content_parts:
        return messages

    # Build summarization prompt
    old_text = "\n".join(old_content_parts[:50])  # Cap at 50 entries
    summary_prompt = (
        "Summarize the following conversation history into a concise factual summary. "
        "Keep all key findings, tool results, decisions, and important details. "
        "Omit verbose tool output and repetitive content. "
        "Output ONLY the summary, no commentary.\n\n"
        f"{old_text}"
    )

    try:
        from omnigent.router import TaskType
        summary_text = ""
        import asyncio
        async with asyncio.timeout(30):
            async for chunk in router.stream(
                messages=[{"role": "user", "content": summary_prompt}],
                tools=None,
                system="You are a concise summarizer. Extract key facts only.",
                task_type=TaskType.REFLECTION,
            ):
                if chunk.content:
                    summary_text += chunk.content
                if chunk.done:
                    break

        if summary_text:
            summary_msg = {
                "role": "user",
                "content": f"[CONTEXT SUMMARY of {len(old_groups)} earlier message groups]:\n{summary_text}",
            }
            return [summary_msg] + _flatten_groups(recent_groups)
    except Exception:
        pass

    # Fallback to heuristic compression
    return smart_trim_context(messages, max_tokens=max_tokens, recent_window=recent_window)


def estimate_tokens(content: Any) -> int:
    """Estimate token count for content (~1 token ≈ 3 characters).

    Uses custom tokenizer if set via set_tokenizer(), otherwise falls back
    to a heuristic of len(content) // 3 which is closer to empirical
    measurements across cl100k and o200k tokenizers.
    """
    if isinstance(content, str):
        if _custom_tokenizer:
            try:
                return _custom_tokenizer(content)
            except Exception:
                pass
        return len(content) // 3
    elif isinstance(content, list):
        return sum(estimate_tokens(item) for item in content)
    elif isinstance(content, dict):
        return sum(estimate_tokens(v) for v in content.values())
    return 0


def calculate_context_cost(
    messages: list[dict[str, Any]],
    system_prompt: str,
    cost_per_1k_tokens: float = 0.00014,
) -> float:
    """Calculate input token cost for current context."""
    system_tokens = estimate_tokens(system_prompt)
    message_tokens = sum(estimate_tokens(msg.get("content", "")) for msg in messages)
    total_tokens = system_tokens + message_tokens
    return (total_tokens / 1000) * cost_per_1k_tokens


def should_trim_context(
    messages: list[dict[str, Any]],
    threshold: int = 25,
    max_tokens: int = 100000,
) -> tuple[bool, int]:
    """
    Check if context should be trimmed.

    Returns (needs_trim, current_token_count).
    """
    total_tokens = sum(estimate_tokens(msg.get("content", "")) for msg in messages)
    needs_trim = len(messages) > threshold or total_tokens > max_tokens
    return needs_trim, total_tokens
