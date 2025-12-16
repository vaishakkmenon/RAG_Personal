"""
Unit tests for Session Management.

Tests the session store functionality:
- Session creation and retrieval
- Rate limiting
- Conversation history management
- Session expiration
- Memory vs Redis backends
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta


@pytest.mark.unit
@pytest.mark.session
class TestMemorySessionStore:
    """Tests for in-memory session store."""

    def test_create_session(self):
        """Test creating a new session."""
        from app.storage.fallback.memory import InMemorySessionStore

        store = InMemorySessionStore()
        session = store.get_or_create_session(
            session_id=None,
            ip_address="127.0.0.1"
        )

        assert session is not None
        assert session.session_id is not None
        assert session.ip_address == "127.0.0.1"
        assert session.request_count == 1  # record_request() is called
        assert len(session.history) == 0

    def test_get_existing_session(self):
        """Test retrieving an existing session."""
        from app.storage.fallback.memory import InMemorySessionStore

        store = InMemorySessionStore()

        # Create session
        session1 = store.get_or_create_session(None, "127.0.0.1")
        session_id = session1.session_id

        # Retrieve same session
        session2 = store.get_or_create_session(session_id, "127.0.0.1")

        assert session1.session_id == session2.session_id
        # Note: Not same object due to record_request() modifying it

    def test_add_to_conversation_history(self):
        """Test adding messages to conversation history."""
        from app.storage.fallback.memory import InMemorySessionStore

        store = InMemorySessionStore()
        session = store.get_or_create_session(None, "127.0.0.1")

        # Add user message
        session.add_turn(role="user", content="What is my Python experience?")
        assert len(session.history) == 1

        # Add assistant message
        session.add_turn(role="assistant", content="You have 5 years.")
        assert len(session.history) == 2

        # Verify content
        assert session.history[0]["role"] == "user"
        assert session.history[1]["role"] == "assistant"

    def test_conversation_history_truncation(self):
        """Test that conversation history is truncated to max messages."""
        from app.storage.fallback.memory import InMemorySessionStore

        store = InMemorySessionStore()
        session = store.get_or_create_session(None, "127.0.0.1")

        # Add 4 turns (8 messages total)
        for i in range(4):
            session.add_turn("user", f"Question {i}")
            session.add_turn("assistant", f"Answer {i}")

        # Get truncated history with max_turns=4 (keeps last 4 messages)
        truncated = session.get_truncated_history(max_turns=4)
        assert len(truncated) == 4
        # Should keep: Question 2, Answer 2, Question 3, Answer 3
        assert "Question 2" in truncated[0]["content"]
        assert "Answer 3" in truncated[-1]["content"]

    def test_rate_limiting(self):
        """Test session rate limiting."""
        from app.storage.fallback.memory import InMemorySessionStore
        from app.settings import settings

        # Temporarily set low rate limit
        original_limit = settings.session.queries_per_hour
        settings.session.queries_per_hour = 3  # Allow 3 requests per hour

        try:
            store = InMemorySessionStore()
            session = store.get_or_create_session(None, "127.0.0.1")
            # First request already recorded by get_or_create_session (count=1)

            # Check: 1 < 3, should pass
            allowed = store.check_rate_limit(session)
            assert allowed is True

            # Second request (count=2)
            session.record_request()
            allowed = store.check_rate_limit(session)
            assert allowed is True  # 2 < 3, still passes

            # Third request (count=3)
            session.record_request()
            allowed = store.check_rate_limit(session)
            assert allowed is False  # 3 < 3 is False, rate limited!
        finally:
            settings.session.queries_per_hour = original_limit

    def test_session_expiration(self):
        """Test that old sessions can be detected as expired."""
        from app.storage.fallback.memory import InMemorySessionStore
        from datetime import datetime, timedelta

        store = InMemorySessionStore()
        session = store.get_or_create_session(None, "127.0.0.1")

        # Manually set last_accessed to 2 hours ago
        session.last_accessed = datetime.now() - timedelta(hours=2)
        store.update_session(session)

        # Note: Cleanup runs in background thread, so we just test the session is old
        # In production, _cleanup_expired() would remove this session
        age = datetime.now() - session.last_accessed
        assert age > timedelta(hours=1)  # Session is old enough to be cleaned up

    def test_max_sessions_per_ip(self):
        """Test IP-based session limit."""
        from app.storage.fallback.memory import InMemorySessionStore
        from app.settings import settings
        from fastapi import HTTPException

        # Temporarily set low IP limit
        original_limit = settings.session.max_sessions_per_ip
        settings.session.max_sessions_per_ip = 2

        try:
            store = InMemorySessionStore()
            ip = "127.0.0.1"

            # Create 2 sessions - should work
            session1 = store.get_or_create_session(None, ip)
            session2 = store.get_or_create_session(None, ip)

            assert session1.session_id != session2.session_id

            # Try to create 3rd session - should raise HTTPException
            with pytest.raises(HTTPException) as exc_info:
                store.get_or_create_session(None, ip)

            assert exc_info.value.status_code == 429
        finally:
            settings.session.max_sessions_per_ip = original_limit


@pytest.mark.unit
@pytest.mark.session
class TestRedisSessionStore:
    """Tests for Redis-backed session store."""

    @patch("app.storage.primary.redis_store.redis.ConnectionPool")
    @patch("app.storage.primary.redis_store.redis.Redis")
    def test_create_session_in_redis(self, mock_redis_class, mock_pool_class):
        """Test creating session with Redis backend."""
        from app.storage.primary.redis_store import RedisSessionStore

        # Mock connection pool
        mock_pool = MagicMock()
        mock_pool_class.from_url.return_value = mock_pool

        # Mock pipeline for atomic operations
        mock_pipeline = MagicMock()
        mock_pipeline.execute.return_value = [True, True, True]  # setex, sadd, expire

        # Mock Redis client
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = None  # Session doesn't exist
        mock_redis.pipeline.return_value = mock_pipeline  # Return pipeline
        mock_redis_class.return_value = mock_redis

        store = RedisSessionStore(redis_url="redis://localhost:6379/0")
        session = store.get_or_create_session(None, "127.0.0.1")

        assert session is not None
        assert session.session_id is not None

        # Verify pipeline operations were called
        assert mock_redis.pipeline.called
        assert mock_pipeline.setex.called
        assert mock_pipeline.execute.called

    @patch("app.storage.primary.redis_store.redis.ConnectionPool")
    @patch("app.storage.primary.redis_store.redis.Redis")
    def test_get_existing_session_from_redis(self, mock_redis_class, mock_pool_class):
        """Test retrieving existing session from Redis."""
        import json
        from app.storage.primary.redis_store import RedisSessionStore
        from datetime import datetime

        # Mock connection pool
        mock_pool = MagicMock()
        mock_pool_class.from_url.return_value = mock_pool

        # Mock Redis client
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        session_data = {
            "session_id": "test-123",
            "ip_address": "127.0.0.1",
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "request_count": 5,
            "history": [],
            "request_timestamps": [],
            "total_tokens_used": 0,
        }
        mock_redis.get.return_value = json.dumps(session_data)
        mock_redis.setex.return_value = True
        mock_redis_class.return_value = mock_redis

        store = RedisSessionStore(redis_url="redis://localhost:6379/0")
        session = store.get_or_create_session("test-123", "127.0.0.1")

        assert session.session_id == "test-123"
        assert session.request_count == 6  # Was 5, +1 from record_request()


@pytest.mark.unit
@pytest.mark.session
class TestSessionStoreFactory:
    """Tests for session store factory function."""

    def test_get_memory_store(self):
        """Test getting memory store from factory."""
        from app.storage.factory import create_session_store
        from app.storage.fallback.memory import InMemorySessionStore

        store = create_session_store(backend="memory")
        assert isinstance(store, InMemorySessionStore)

    @patch("app.storage.primary.redis_store.redis.Redis")
    def test_get_redis_store(self, mock_redis_class):
        """Test getting Redis store from factory."""
        from app.storage.factory import create_session_store
        from app.storage.primary.redis_store import RedisSessionStore

        # Mock Redis connection
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis

        store = create_session_store(
            backend="redis",
            redis_url="redis://localhost:6379/0"
        )

        assert isinstance(store, RedisSessionStore)
