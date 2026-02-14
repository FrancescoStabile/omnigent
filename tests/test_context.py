"""Tests for Context Management."""

import pytest
from omnigent.context import (
    trim_context_window,
    smart_trim_context,
    estimate_tokens,
    set_tokenizer,
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

    def test_divisor_is_3(self):
        """Token estimate should use //3 (not //4) for closer approximation."""
        text = "a" * 300
        tokens = estimate_tokens(text)
        assert tokens == 100  # 300 // 3 = 100

    def test_estimate_bounds_realistic(self):
        """For English prose, ~1 token per 3-4 chars; //3 is a conservative estimate."""
        text = "The quick brown fox jumps over the lazy dog"  # 43 chars
        tokens = estimate_tokens(text)
        # 43 // 3 = 14
        assert tokens == 14


class TestCustomTokenizer:
    """Tests for set_tokenizer() hook."""

    def teardown_method(self):
        """Reset tokenizer after each test."""
        set_tokenizer(None)

    def test_custom_tokenizer_used(self):
        """Custom tokenizer function should override heuristic."""
        set_tokenizer(lambda text: 42)
        assert estimate_tokens("anything") == 42

    def test_custom_tokenizer_reset(self):
        """Passing None should revert to heuristic."""
        set_tokenizer(lambda text: 99)
        assert estimate_tokens("abc") == 99
        set_tokenizer(None)
        assert estimate_tokens("abc") == 1  # 3 // 3 = 1

    def test_custom_tokenizer_error_fallback(self):
        """If custom tokenizer raises, should fall back to heuristic."""
        def broken_tokenizer(text):
            raise ValueError("broken")

        set_tokenizer(broken_tokenizer)
        text = "a" * 30
        tokens = estimate_tokens(text)
        assert tokens == 10  # fallback: 30 // 3

    def test_custom_tokenizer_only_for_strings(self):
        """Custom tokenizer should only be called for string content."""
        calls = []
        def tracking_tokenizer(text):
            calls.append(text)
            return len(text)

        set_tokenizer(tracking_tokenizer)
        # Dict content should recurse, not call tokenizer directly
        estimate_tokens({"key": "val"})
        assert "val" in calls
        assert {"key": "val"} not in calls
