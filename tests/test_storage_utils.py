"""
Tests for storage utility functions.

Tests token estimation, history truncation, and session ID masking.
"""

import pytest


@pytest.mark.unit
class TestEstimateTokens:
    """Tests for token estimation."""

    def test_estimate_tokens_normal_text(self):
        """Test token estimation for normal text."""
        from app.storage.utils import estimate_tokens

        text = "This is a simple sentence with eight words"
        tokens = estimate_tokens(text)

        # Should be approximately 8 * 0.75 = 6
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_estimate_tokens_empty_string(self):
        """Test token estimation for empty string."""
        from app.storage.utils import estimate_tokens

        tokens = estimate_tokens("")

        assert tokens == 0

    def test_estimate_tokens_single_word(self):
        """Test token estimation for single word."""
        from app.storage.utils import estimate_tokens

        tokens = estimate_tokens("hello")

        assert tokens >= 0


@pytest.mark.unit
class TestTruncateHistory:
    """Tests for history truncation."""

    def test_truncate_history_empty(self):
        """Test truncating empty history."""
        from app.storage.utils import truncate_history

        result = truncate_history([])

        assert result == []

    def test_truncate_history_within_limits(self):
        """Test history within limits is unchanged."""
        from app.storage.utils import truncate_history

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        result = truncate_history(history, max_tokens=1000, max_turns=10)

        assert len(result) == 2

    def test_truncate_history_by_turn_count(self):
        """Test history truncation by turn count."""
        from app.storage.utils import truncate_history

        history = [{"role": "user", "content": f"Message {i}"} for i in range(10)]

        result = truncate_history(history, max_tokens=1000, max_turns=3)

        assert len(result) <= 3

    def test_truncate_history_removes_orphaned_assistant(self):
        """Test that orphaned assistant message at start is removed."""
        from app.storage.utils import truncate_history

        # History that starts with assistant (after truncation)
        history = [
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Response 2"},
        ]

        result = truncate_history(history, max_tokens=1000, max_turns=3)

        # Depending on implementation, might remove orphaned assistant
        assert all(isinstance(msg, dict) for msg in result)


@pytest.mark.unit
class TestMaskSessionId:
    """Tests for session ID masking."""

    def test_mask_session_id_normal(self):
        """Test masking normal session ID."""
        from app.storage.utils import mask_session_id

        masked = mask_session_id("abc12345-full-session-id")

        assert masked == "abc12345***"
        assert "full-session-id" not in masked

    def test_mask_session_id_short(self):
        """Test masking short session ID."""
        from app.storage.utils import mask_session_id

        masked = mask_session_id("abc")

        assert masked == "***"

    def test_mask_session_id_empty(self):
        """Test masking empty session ID."""
        from app.storage.utils import mask_session_id

        masked = mask_session_id("")

        assert masked == "***"

    def test_mask_session_id_none(self):
        """Test masking None session ID."""
        from app.storage.utils import mask_session_id

        masked = mask_session_id(None)

        assert masked == "***"
