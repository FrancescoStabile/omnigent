"""Tests for context trimming atomic group preservation.

Verifies that assistant+tool message groups are never split during
context trimming operations.
"""

import pytest

from omnigent.context import (
    _group_messages,
    smart_trim_context,
    trim_context_window,
)


class TestGroupMessages:
    """Tests for _group_messages() — the core grouping logic."""

    def test_user_messages_separate_groups(self):
        msgs = [
            {"role": "user", "content": "msg1"},
            {"role": "user", "content": "msg2"},
        ]
        groups = _group_messages(msgs)
        assert len(groups) == 2
        assert len(groups[0]) == 1
        assert len(groups[1]) == 1

    def test_assistant_text_separate_group(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        groups = _group_messages(msgs)
        assert len(groups) == 2

    def test_assistant_with_tool_use_grouped(self):
        """Assistant message with tool_use blocks must stay grouped with subsequent tool messages."""
        msgs = [
            {"role": "user", "content": "do something"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll run a scan"},
                    {"type": "tool_use", "id": "tc1", "name": "scan", "input": {}},
                ],
            },
            {"role": "tool", "content": {"tool_call_id": "tc1", "content": "scan result"}},
        ]
        groups = _group_messages(msgs)
        assert len(groups) == 2  # user group + (assistant+tool) group
        assert len(groups[1]) == 2  # assistant + tool in one group

    def test_multiple_tool_results_grouped(self):
        """Multiple tool results after one assistant message stay in same group."""
        msgs = [
            {"role": "user", "content": "run scans"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tc1", "name": "tool_a", "input": {}},
                    {"type": "tool_use", "id": "tc2", "name": "tool_b", "input": {}},
                ],
            },
            {"role": "tool", "content": {"tool_call_id": "tc1", "content": "result a"}},
            {"role": "tool", "content": {"tool_call_id": "tc2", "content": "result b"}},
        ]
        groups = _group_messages(msgs)
        assert len(groups) == 2
        assert len(groups[1]) == 3  # assistant + 2 tool messages

    def test_consecutive_assistant_tool_groups(self):
        """Two separate assistant+tool groups remain separate."""
        msgs = [
            {"role": "user", "content": "start"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tc1", "name": "scan", "input": {}},
                ],
            },
            {"role": "tool", "content": {"tool_call_id": "tc1", "content": "result1"}},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tc2", "name": "scan2", "input": {}},
                ],
            },
            {"role": "tool", "content": {"tool_call_id": "tc2", "content": "result2"}},
        ]
        groups = _group_messages(msgs)
        assert len(groups) == 3  # user + group1 + group2
        assert len(groups[1]) == 2
        assert len(groups[2]) == 2


class TestTrimContextWindowAtomic:
    """Verify trim_context_window never splits atomic groups."""

    def _make_tool_group(self, index: int) -> list[dict]:
        """Create an assistant+tool atomic group."""
        return [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"tc_{index}", "name": f"tool_{index}", "input": {}},
                ],
            },
            {"role": "tool", "content": {"tool_call_id": f"tc_{index}", "content": f"result {index}"}},
        ]

    def test_atomic_groups_preserved_during_trim(self):
        """After trimming, no assistant+tool_use message should lack its tool result."""
        msgs = [{"role": "user", "content": "initial"}]
        for i in range(20):
            msgs.extend(self._make_tool_group(i))

        result = trim_context_window(msgs, max_messages=10, preserve_first=1)

        # Verify: every assistant with tool_use is followed by its tool result
        for idx, msg in enumerate(result):
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                has_tool_use = any(
                    isinstance(item, dict) and item.get("type") == "tool_use"
                    for item in msg["content"]
                )
                if has_tool_use:
                    # Next message(s) must be tool messages
                    assert idx + 1 < len(result), "Assistant with tool_use has no following tool message"
                    assert result[idx + 1].get("role") == "tool"

    def test_smart_trim_preserves_atomic_groups(self):
        """smart_trim_context also preserves assistant+tool groups."""
        msgs = [{"role": "user", "content": "start"}]
        for i in range(15):
            msgs.extend(self._make_tool_group(i))
            msgs.append({"role": "user", "content": f"follow up {i}"})

        result = smart_trim_context(msgs, max_tokens=2000, recent_window=4)

        for idx, msg in enumerate(result):
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                has_tool_use = any(
                    isinstance(item, dict) and item.get("type") == "tool_use"
                    for item in msg["content"]
                )
                if has_tool_use:
                    assert idx + 1 < len(result), "Atomic group split: tool_use without tool result"
                    assert result[idx + 1].get("role") == "tool"

    def test_tool_message_never_orphaned(self):
        """A tool message should never appear without a preceding assistant+tool_use."""
        msgs = [{"role": "user", "content": "go"}]
        for i in range(10):
            msgs.extend(self._make_tool_group(i))

        result = trim_context_window(msgs, max_messages=8, preserve_first=1)

        for idx, msg in enumerate(result):
            if msg.get("role") == "tool":
                # Must be preceded by assistant with tool_use or another tool message
                assert idx > 0, "Tool message at index 0"
                prev = result[idx - 1]
                assert prev.get("role") in ("assistant", "tool"), (
                    f"Tool message at index {idx} not preceded by assistant or tool"
                )
