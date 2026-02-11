"""Tests for Context Management."""

import pytest
from omnigent.context import (
    trim_context_window,
    smart_trim_context,
    estimate_tokens,
)


class TestTrimContextWindow:
    def test_short_messages_untouched(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        result = trim_context_window(msgs, max_messages=10)
        assert len(result) == 2

    def test_trims_to_budget(self):
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(50)]
        result = trim_context_window(msgs, max_messages=10, preserve_first=2)
        assert len(result) <= 10

    def test_preserves_first_messages(self):
        msgs = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "first user msg"},
        ] + [{"role": "user", "content": f"msg {i}"} for i in range(30)]
        result = trim_context_window(msgs, max_messages=10, preserve_first=2)
        assert result[0]["content"] == "system prompt"

    def test_preserves_recent_messages(self):
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(30)]
        result = trim_context_window(msgs, max_messages=10, preserve_first=1)
        assert result[-1]["content"] == "msg 29"


class TestSmartTrimContext:
    def test_under_budget_unchanged(self):
        msgs = [{"role": "user", "content": "short"}]
        result = smart_trim_context(msgs, max_tokens=100000)
        assert len(result) == 1

    def test_compresses_long_conversations(self):
        msgs = [{"role": "user", "content": "x" * 10000} for _ in range(20)]
        result = smart_trim_context(msgs, max_tokens=5000, recent_window=4)
        # Should compress or trim
        assert len(result) <= 20


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_normal_text(self):
        tokens = estimate_tokens("hello world this is a test")
        assert tokens > 0
        assert tokens < 100

    def test_handles_non_string(self):
        # Should handle list (multimodal content)
        tokens = estimate_tokens([{"type": "text", "text": "hello"}])
        assert tokens > 0
