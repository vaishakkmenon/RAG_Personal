"""
Unit tests for Response Caching.

Tests:
- Cache hit/miss
- Cache key generation
- TTL expiration
- Cache invalidation
- Redis integration
"""

import pytest
from unittest.mock import MagicMock, patch
import json
import time


@pytest.mark.unit
class TestResponseCache:
    """Tests for response caching functionality."""

    @patch("app.services.response_cache.redis.Redis")
    def test_cache_miss(self, mock_redis_class):
        """Test cache miss returns None."""
        from app.services.response_cache import ResponseCache

        # Mock Redis client (cache miss)
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis_class.return_value = mock_redis

        cache = ResponseCache(redis_url="redis://localhost:6379/0")
        result = cache.get(
            question="What is my Python experience?",
            session_id=None,
            params={"top_k": 5, "temperature": 0.1}
        )

        assert result is None
        mock_redis.get.assert_called_once()

    @patch("app.services.response_cache.redis.Redis")
    def test_cache_hit(self, mock_redis_class):
        """Test cache hit returns cached response."""
        from app.services.response_cache import ResponseCache

        # Mock cached response
        cached_data = {
            "answer": "You have 5 years of Python experience.",
            "sources": [],
            "grounded": True,
            "metadata": {"cached": True}
        }

        # Mock Redis client (cache hit)
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(cached_data).encode()
        mock_redis_class.return_value = mock_redis

        cache = ResponseCache(redis_url="redis://localhost:6379/0")
        result = cache.get(
            question="What is my Python experience?",
            session_id=None,
            params={"top_k": 5, "temperature": 0.1}
        )

        assert result is not None
        assert result["answer"] == "You have 5 years of Python experience."
        assert result["grounded"] is True

    @patch("app.services.response_cache.redis.Redis")
    def test_cache_set(self, mock_redis_class):
        """Test setting cache value."""
        from app.services.response_cache import ResponseCache

        # Mock Redis client
        mock_redis = MagicMock()
        mock_redis.setex.return_value = True
        mock_redis_class.return_value = mock_redis

        cache = ResponseCache(
            redis_url="redis://localhost:6379/0",
            ttl_seconds=3600
        )

        response = {
            "answer": "Test answer",
            "sources": [],
            "grounded": True,
            "metadata": {}
        }

        cache.set(
            question="Test question",
            session_id=None,
            params={"top_k": 5},
            response=response
        )

        # Verify Redis setex was called with TTL
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args.args[1] == 3600  # TTL

    @patch("app.services.response_cache.redis.Redis")
    def test_cache_key_generation(self, mock_redis_class):
        """Test that cache keys are generated consistently."""
        from app.services.response_cache import ResponseCache

        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis_class.return_value = mock_redis

        cache = ResponseCache(redis_url="redis://localhost:6379/0")

        # Same question and params should generate same key
        cache.get(
            question="What is my Python experience?",
            session_id=None,
            params={"top_k": 5, "temperature": 0.1}
        )
        call1_key = mock_redis.get.call_args_list[0].args[0]

        cache.get(
            question="What is my Python experience?",
            session_id=None,
            params={"top_k": 5, "temperature": 0.1}
        )
        call2_key = mock_redis.get.call_args_list[1].args[0]

        assert call1_key == call2_key

    @patch("app.services.response_cache.redis.Redis")
    def test_cache_key_differs_with_params(self, mock_redis_class):
        """Test that different params generate different cache keys."""
        from app.services.response_cache import ResponseCache

        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis_class.return_value = mock_redis

        cache = ResponseCache(redis_url="redis://localhost:6379/0")

        # Different params should generate different keys
        cache.get(
            question="What is my Python experience?",
            session_id=None,
            params={"top_k": 5}
        )
        key1 = mock_redis.get.call_args_list[0].args[0]

        cache.get(
            question="What is my Python experience?",
            session_id=None,
            params={"top_k": 10}  # Different!
        )
        key2 = mock_redis.get.call_args_list[1].args[0]

        assert key1 != key2

    @patch("app.services.response_cache.redis.Redis")
    def test_cache_with_session_id(self, mock_redis_class):
        """Test that session_id is included in cache key."""
        from app.services.response_cache import ResponseCache

        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis_class.return_value = mock_redis

        cache = ResponseCache(redis_url="redis://localhost:6379/0")

        # Same question but different session should generate different keys
        cache.get(
            question="What is my Python experience?",
            session_id="session-123",
            params={"top_k": 5}
        )
        key1 = mock_redis.get.call_args_list[0].args[0]

        cache.get(
            question="What is my Python experience?",
            session_id="session-456",  # Different!
            params={"top_k": 5}
        )
        key2 = mock_redis.get.call_args_list[1].args[0]

        assert key1 != key2

    @patch("app.services.response_cache.redis.Redis")
    def test_cache_disabled(self, mock_redis_class):
        """Test that cache can be disabled."""
        from app.services.response_cache import ResponseCache

        mock_redis = MagicMock()
        mock_redis_class.return_value = mock_redis

        cache = ResponseCache(
            redis_url="redis://localhost:6379/0",
            enabled=False
        )

        # Get should return None when disabled
        result = cache.get(
            question="Test",
            session_id=None,
            params={}
        )

        assert result is None
        # Redis should not be called
        mock_redis.get.assert_not_called()

    @patch("app.services.response_cache.redis.Redis")
    def test_cache_handles_redis_failure(self, mock_redis_class):
        """Test graceful handling of Redis failures."""
        from app.services.response_cache import ResponseCache

        # Mock Redis failure
        mock_redis = MagicMock()
        mock_redis.get.side_effect = Exception("Redis connection failed")
        mock_redis_class.return_value = mock_redis

        cache = ResponseCache(redis_url="redis://localhost:6379/0")

        # Should return None on Redis failure (graceful degradation)
        result = cache.get(
            question="Test",
            session_id=None,
            params={}
        )

        assert result is None


@pytest.mark.unit
class TestResponseCacheSingleton:
    """Tests for response cache singleton pattern."""

    def test_get_response_cache_returns_singleton(self):
        """Test that get_response_cache returns the same instance."""
        from app.services.response_cache import get_response_cache

        cache1 = get_response_cache()
        cache2 = get_response_cache()

        assert cache1 is cache2
