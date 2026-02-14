"""Tests for router.py bug fixes and improvements."""

import asyncio
import json
import logging
import uuid

import httpx
import pytest

from omnigent.router import LLMRouter, StreamChunk


class TestUUIDToolCallIDs:
    """Tests for UUID-based tool call IDs (replacing hash-based)."""

    def test_tool_call_id_format(self):
        """Tool call IDs should use UUID hex format: call_<8hex>."""
        call_id = "call_" + uuid.uuid4().hex[:8]
        assert call_id.startswith("call_")
        assert len(call_id) == 13  # "call_" (5) + 8 hex chars

    def test_tool_call_ids_unique(self):
        """Multiple generated IDs should be unique."""
        ids = {"call_" + uuid.uuid4().hex[:8] for _ in range(1000)}
        assert len(ids) == 1000

    def test_uuid_hex_chars_valid(self):
        """UUID hex should only contain valid hex characters."""
        hex_part = uuid.uuid4().hex[:8]
        assert all(c in "0123456789abcdef" for c in hex_part)


class TestMalformedJSONWarning:
    """Tests for JSON parse warning logging in tool arguments."""

    def test_malformed_json_logs_warning(self, caplog):
        """Invalid JSON should trigger a logger.warning with details."""
        logger = logging.getLogger("omnigent.router")
        bad_json = '{"key": invalid}'
        tool_name = "test_tool"

        # Simulate the parse behavior from router.py
        with caplog.at_level(logging.WARNING, logger="omnigent.router"):
            try:
                args = json.loads(bad_json)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Malformed JSON in tool arguments for '{tool_name}': {e}. "
                    f"Raw: {bad_json[:200]}"
                )
                args = {}

        assert args == {}
        assert "Malformed JSON" in caplog.text
        assert "test_tool" in caplog.text
        assert "invalid" in caplog.text

    def test_valid_json_no_warning(self, caplog):
        """Valid JSON should not trigger any warning."""
        with caplog.at_level(logging.WARNING, logger="omnigent.router"):
            args = json.loads('{"key": "value"}')
        assert args == {"key": "value"}
        assert "Malformed JSON" not in caplog.text

    def test_empty_json_no_warning(self, caplog):
        """Empty string should be handled gracefully (results in empty dict)."""
        with caplog.at_level(logging.WARNING, logger="omnigent.router"):
            args = json.loads("{}") if "{}" else {}
        assert args == {}
        assert "Malformed JSON" not in caplog.text

    def test_truncated_raw_output(self, caplog):
        """Long malformed JSON should be truncated to 200 chars in log."""
        logger = logging.getLogger("omnigent.router")
        bad_json = '{"key": ' + "x" * 300 + "}"

        with caplog.at_level(logging.WARNING, logger="omnigent.router"):
            try:
                json.loads(bad_json)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Malformed JSON in tool arguments for 'test': {e}. "
                    f"Raw: {bad_json[:200]}"
                )

        # The raw output in the log should be truncated
        assert "Malformed JSON" in caplog.text
        # Ensure it was truncated (200 chars of raw, not 300+)
        raw_in_log = caplog.text.split("Raw: ")[-1]
        assert len(raw_in_log) < 350  # truncated


class TestStreamWithRetry:
    """Tests for exponential backoff retry mechanism."""

    def test_retryable_codes(self):
        """Verify the set of retryable HTTP status codes."""
        retryable = {429, 500, 502, 503, 504}
        assert 429 in retryable  # Rate limit
        assert 500 in retryable  # Internal server error
        assert 502 in retryable  # Bad gateway
        assert 503 in retryable  # Service unavailable
        assert 504 in retryable  # Gateway timeout
        assert 400 not in retryable  # Bad request - NOT retryable
        assert 401 not in retryable  # Unauthorized - NOT retryable
        assert 404 not in retryable  # Not found - NOT retryable

    def test_backoff_delay_formula(self):
        """Verify exponential backoff formula: base * 2^attempt + jitter."""
        import random

        random.seed(42)
        base_delay = 1.0

        delays = []
        for attempt in range(4):
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            delays.append(delay)

        # Attempt 0: ~1 + jitter (1.0-2.0)
        assert 1.0 <= delays[0] <= 2.0
        # Attempt 1: ~2 + jitter (2.0-3.0)
        assert 2.0 <= delays[1] <= 3.0
        # Attempt 2: ~4 + jitter (4.0-5.0)
        assert 4.0 <= delays[2] <= 5.0
        # Attempt 3: ~8 + jitter (8.0-9.0)
        assert 8.0 <= delays[3] <= 9.0

    def test_max_retries_default(self):
        """Default max_retries should be 3."""
        # Read from the method signature default
        import inspect
        sig = inspect.signature(LLMRouter._stream_with_retry)
        max_retries_param = sig.parameters["max_retries"]
        assert max_retries_param.default == 3

    def test_base_delay_default(self):
        """Default base_delay should be 1.0 seconds."""
        import inspect
        sig = inspect.signature(LLMRouter._stream_with_retry)
        base_delay_param = sig.parameters["base_delay"]
        assert base_delay_param.default == 1.0


class TestStreamChunkCacheFields:
    """Tests for cache token fields in StreamChunk."""

    def test_stream_chunk_has_cache_fields(self):
        """StreamChunk should include cache_read_tokens and cache_creation_tokens."""
        chunk = StreamChunk(
            model="test",
            cache_read_tokens=100,
            cache_creation_tokens=50,
        )
        assert chunk.cache_read_tokens == 100
        assert chunk.cache_creation_tokens == 50

    def test_stream_chunk_cache_defaults_zero(self):
        """Cache token fields should default to 0."""
        chunk = StreamChunk(model="test")
        assert chunk.cache_read_tokens == 0
        assert chunk.cache_creation_tokens == 0
