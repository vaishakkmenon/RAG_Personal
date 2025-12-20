"""
Tests for Redis degradation and fallback scenarios.

Verifies that the application gracefully handles Redis unavailability:
- Session storage falls back to in-memory store
- Response cache degrades gracefully
- Health checks report degraded status
- Application continues to function
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError


@pytest.mark.integration
class TestRedisSessionStorageFallback:
    """Tests for session storage fallback when Redis is unavailable."""

    def test_session_storage_falls_back_to_memory_on_redis_failure(self):
        """Test that session storage automatically falls back to in-memory when Redis fails."""
        from app.storage.factory import create_session_store

        # Simulate Redis connection failure
        with patch(
            "app.storage.primary.RedisSessionStore.__init__",
            side_effect=RedisConnectionError("Connection refused"),
        ):
            store = create_session_store(
                backend="redis", redis_url="redis://localhost:6379/0"
            )

            # Should return in-memory store as fallback
            from app.storage.fallback import InMemorySessionStore

            assert isinstance(store, InMemorySessionStore)

    def test_sessions_continue_working_with_memory_fallback(self):
        """Test that sessions continue to work when using in-memory fallback."""
        from app.storage.factory import create_session_store

        # Create in-memory store (fallback)
        store = create_session_store(backend="memory")

        # Create a session
        session = store.get_or_create_session(session_id=None, ip_address="127.0.0.1")

        assert session is not None
        assert session.session_id is not None

        # Add messages
        session.add_turn(role="user", content="Test question")
        session.add_turn(role="assistant", content="Test answer")

        # Update session
        store.update_session(session)

        # Retrieve session
        retrieved = store.get_session(session.session_id)
        assert retrieved is not None
        assert len(retrieved.history) == 2

    def test_session_rate_limiting_works_with_memory_fallback(self):
        """Test that rate limiting works correctly with in-memory fallback."""
        from app.storage.factory import create_session_store
        from app.settings import settings

        store = create_session_store(backend="memory")

        # Temporarily lower the rate limit for testing
        original_limit = settings.session.queries_per_hour
        settings.session.queries_per_hour = 5

        try:
            session = store.get_or_create_session(
                session_id=None, ip_address="127.0.0.1"
            )

            # Should be able to make requests up to the limit (but not exceed it)
            # The get_or_create_session already recorded 1 request
            for i in range(4):  # Only 4 more since we already have 1
                assert store.check_rate_limit(session)
                session.record_request()

            # Should not be able to query after reaching limit
            assert not store.check_rate_limit(session)
        finally:
            # Restore original limit
            settings.session.queries_per_hour = original_limit

    def test_ip_based_session_lookup_works_with_memory_fallback(self):
        """Test that IP-based session lookup works with in-memory store."""
        from app.storage.factory import create_session_store

        store = create_session_store(backend="memory")

        # Create multiple sessions for same IP
        session1 = store.get_or_create_session(None, "192.168.1.100")
        session2 = store.get_or_create_session(None, "192.168.1.100")

        # Get sessions by IP
        sessions_for_ip = store.get_sessions_by_ip("192.168.1.100")
        assert len(sessions_for_ip) == 2
        assert session1.session_id in [s.session_id for s in sessions_for_ip]
        assert session2.session_id in [s.session_id for s in sessions_for_ip]

    def test_session_cleanup_works_with_memory_fallback(self):
        """Test that expired session cleanup works with in-memory store."""
        from app.storage.factory import create_session_store
        from datetime import datetime, timedelta

        store = create_session_store(backend="memory")

        # Create a session
        session = store.get_or_create_session(None, "127.0.0.1")
        session_id = session.session_id

        # Manually expire the session by setting old last_accessed time
        session.last_accessed = datetime.now() - timedelta(hours=7)
        store.update_session(session)

        # Manually delete the expired session (simulating cleanup)
        store.delete_session(session_id)

        # Session should be removed
        retrieved = store.get_session(session_id)
        assert retrieved is None


@pytest.mark.integration
class TestRedisCacheDegradation:
    """Tests for response cache degradation when Redis is unavailable."""

    def test_cache_degrades_gracefully_on_redis_failure(self):
        """Test that response cache gracefully disables when Redis fails."""
        from app.services.response_cache import ResponseCache

        # Simulate Redis connection failure
        with patch(
            "redis.Redis", side_effect=RedisConnectionError("Connection refused")
        ):
            cache = ResponseCache(
                redis_url="redis://localhost:6379/0", ttl_seconds=3600, enabled=True
            )

            # Cache should be disabled but not crash
            assert cache.enabled is False

    def test_cache_operations_safe_when_disabled(self):
        """Test that cache operations are safe when cache is disabled."""
        from app.services.response_cache import ResponseCache

        # Create disabled cache
        cache = ResponseCache(
            redis_url="redis://localhost:6379/0", ttl_seconds=3600, enabled=False
        )

        # All operations should be safe no-ops
        assert cache.get("test-key") is None
        cache.set("test-key", "test-value")  # Should not crash
        cache.invalidate_pattern("test-*")  # Should not crash

    @patch("app.services.response_cache.get_response_cache")
    @patch("app.api.routes.chat.search")
    @patch("app.api.routes.chat.generate_with_llm")
    @patch("app.api.routes.chat.get_prompt_guard")
    def test_chat_endpoint_works_without_cache(
        self,
        mock_guard,
        mock_generate,
        mock_search,
        mock_cache,
        client: TestClient,
        auth_headers: dict,
        sample_chunks: list,
    ):
        """Test that chat endpoint works when cache is unavailable."""
        # Mock disabled cache
        mock_cache_instance = MagicMock()
        mock_cache_instance.get.return_value = None
        mock_cache_instance._enabled = False
        mock_cache.return_value = mock_cache_instance

        # Mock prompt guard
        mock_guard_instance = MagicMock()
        mock_guard_instance.check_input.return_value = {
            "blocked": False,
            "label": "safe",
        }
        mock_guard.return_value = mock_guard_instance

        # Mock search and generation
        mock_search.return_value = sample_chunks
        mock_generate.return_value = "Test response"

        response = client.post(
            "/chat/simple",
            json={"question": "What is my Python experience?"},
            headers=auth_headers,
        )

        # Should still work without cache
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data


@pytest.mark.integration
class TestRedisHealthCheckDegradation:
    """Tests for health check reporting when Redis is degraded."""

    def test_health_check_reports_redis_degraded(self, client: TestClient):
        """Test that detailed health check reports Redis as degraded when unavailable."""
        # Patch Redis connection to fail
        with patch("app.services.response_cache.get_response_cache") as mock_cache:
            mock_cache_instance = MagicMock()
            # Health check accesses cache._redis.ping()
            mock_cache_instance._redis.ping.side_effect = RedisConnectionError(
                "Connection refused"
            )
            mock_cache.return_value = mock_cache_instance

            response = client.get("/health/detailed")

            assert response.status_code == 200
            data = response.json()

            # Overall status should be degraded
            assert data["status"] == "degraded"

            # Redis should be marked as degraded
            assert "dependencies" in data
            assert data["dependencies"]["redis"] == "degraded"

    def test_basic_health_check_always_succeeds(self, client: TestClient):
        """Test that basic health check always returns 200 even if Redis is down."""
        response = client.get("/health")

        # Basic health check should always succeed if API is running
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_readiness_probe_succeeds_without_redis(self, client: TestClient):
        """Test that readiness probe focuses on critical dependencies, not Redis."""
        with patch("app.services.response_cache.get_response_cache") as mock_cache:
            mock_cache_instance = MagicMock()
            mock_cache_instance._client.ping.side_effect = RedisConnectionError(
                "Connection refused"
            )
            mock_cache.return_value = mock_cache_instance

            response = client.get("/health/ready")

            # Should be ready as long as ChromaDB is available
            # Redis is not critical for readiness
            assert response.status_code == 200


@pytest.mark.integration
class TestRedisMetricsWithFallback:
    """Tests for Prometheus metrics when using memory fallback."""

    def test_memory_fallback_exposes_metrics(self):
        """Test that in-memory store exposes Prometheus metrics like Redis."""
        from app.storage.factory import create_session_store

        store = create_session_store(backend="memory")

        # Create some sessions
        for i in range(3):
            store.get_or_create_session(None, f"192.168.1.{i}")

        # Get session count
        count = store.get_session_count()

        # Should have sessions
        assert count == 3

    def test_fallback_metrics_match_redis_interface(self):
        """Test that fallback store has same interface as Redis store."""
        from app.storage.factory import create_session_store

        memory_store = create_session_store(
            backend="memory"
        )  # Using memory as fallback

        # Should have same core methods as Redis store
        assert hasattr(memory_store, "get_or_create_session")
        assert hasattr(memory_store, "get_session")
        assert hasattr(memory_store, "get_sessions_by_ip")
        assert hasattr(memory_store, "delete_session")
        assert hasattr(memory_store, "get_session_count")
        assert hasattr(memory_store, "update_session")
        assert hasattr(memory_store, "create_session")
        # Note: cleanup_expired_sessions is handled by background thread, not exposed as public API


@pytest.mark.integration
class TestRedisFailoverScenarios:
    """Tests for various Redis failure and recovery scenarios."""

    def test_application_starts_without_redis(self):
        """Test that application can start when Redis is unavailable at startup."""
        # This test verifies the factory handles Redis unavailability at startup
        from app.storage.factory import create_session_store

        with patch(
            "app.storage.primary.RedisSessionStore.__init__",
            side_effect=RedisConnectionError("Connection refused"),
        ):
            store = create_session_store(
                backend="redis", redis_url="redis://localhost:6379/0"
            )

            # Should successfully create fallback store
            assert store is not None

            # Should be able to use it
            session = store.get_or_create_session(None, "127.0.0.1")
            assert session is not None

    def test_redis_failure_logged_appropriately(self, caplog):
        """Test that Redis connection failures are logged with appropriate severity."""
        from app.storage.factory import create_session_store
        import logging

        with caplog.at_level(logging.WARNING):
            with patch(
                "app.storage.primary.RedisSessionStore.__init__",
                side_effect=RedisConnectionError("Connection refused"),
            ):
                create_session_store(
                    backend="redis", redis_url="redis://localhost:6379/0"
                )

                # Should log error about Redis failure
                assert any(
                    "[FAILED]" in record.message
                    and "Failed to initialize Redis store" in record.message
                    for record in caplog.records
                )

                # Should log warning about fallback activation
                assert any(
                    "[FALLBACK ACTIVATED]" in record.message
                    for record in caplog.records
                )
