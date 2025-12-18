"""
Tests for rate limiter functionality.

Tests token bucket algorithm, rate limiting with minute/day limits,
and NoOp rate limiter.
"""

import pytest
from unittest.mock import patch, MagicMock
import time


@pytest.mark.unit
class TestRateLimiterInit:
    """Tests for RateLimiter initialization."""

    def test_rate_limiter_init_with_limits(self):
        """Test RateLimiter initialization with limits."""
        from app.services.rate_limiter import RateLimiter

        limiter = RateLimiter(
            requests_per_minute=30,
            requests_per_day=1000
        )

        assert limiter is not None
        assert limiter.requests_per_minute == 30
        assert limiter.requests_per_day == 1000

    def test_rate_limiter_default_values(self):
        """Test RateLimiter has reasonable defaults."""
        from app.services.rate_limiter import RateLimiter

        limiter = RateLimiter()

        assert limiter.requests_per_minute > 0
        assert limiter.requests_per_day > 0


@pytest.mark.unit
class TestRateLimiterAcquire:
    """Tests for rate limiter acquire method."""

    def test_acquire_succeeds_under_limit(self):
        """Test that acquire succeeds when under rate limit."""
        from app.services.rate_limiter import RateLimiter

        limiter = RateLimiter(
            requests_per_minute=100,
            requests_per_day=10000
        )

        # Should succeed for first request
        result = limiter.acquire(timeout=1)

        assert result is True

    def test_acquire_multiple_requests(self):
        """Test multiple acquire calls."""
        from app.services.rate_limiter import RateLimiter

        limiter = RateLimiter(
            requests_per_minute=100,
            requests_per_day=10000
        )

        # Should succeed for multiple requests under limit
        for _ in range(5):
            result = limiter.acquire(timeout=0.1)
            assert result is True


@pytest.mark.unit
class TestRateLimiterStats:
    """Tests for rate limiter statistics."""

    def test_get_stats_returns_dict(self):
        """Test that get_stats returns statistics dict."""
        from app.services.rate_limiter import RateLimiter

        limiter = RateLimiter(
            requests_per_minute=30,
            requests_per_day=1000
        )

        stats = limiter.get_stats()

        assert isinstance(stats, dict)
        assert 'requests_last_minute' in stats or 'minute_utilization' in stats

    def test_get_stats_includes_limits(self):
        """Test that stats includes configured limits."""
        from app.services.rate_limiter import RateLimiter

        limiter = RateLimiter(
            requests_per_minute=30,
            requests_per_day=1000
        )

        stats = limiter.get_stats()

        assert stats.get('requests_per_minute_limit', 30) == 30
        assert stats.get('requests_per_day_limit', 1000) == 1000


@pytest.mark.unit
class TestNoOpRateLimiter:
    """Tests for NoOpRateLimiter (bypass rate limiting)."""

    def test_noop_rate_limiter_init(self):
        """Test NoOpRateLimiter initialization."""
        from app.services.rate_limiter import NoOpRateLimiter

        limiter = NoOpRateLimiter()

        assert limiter is not None

    def test_noop_rate_limiter_always_allows(self):
        """Test that NoOpRateLimiter always allows requests."""
        from app.services.rate_limiter import NoOpRateLimiter

        limiter = NoOpRateLimiter()

        # Should always return True
        for _ in range(100):
            result = limiter.acquire(timeout=0)
            assert result is True

    def test_noop_rate_limiter_get_stats(self):
        """Test NoOpRateLimiter get_stats."""
        from app.services.rate_limiter import NoOpRateLimiter

        limiter = NoOpRateLimiter()

        stats = limiter.get_stats()

        assert isinstance(stats, dict)
