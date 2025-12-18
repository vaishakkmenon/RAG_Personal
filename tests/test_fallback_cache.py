"""
Tests for retrieval fallback cache.

Tests the LRU cache used for fallback when ChromaDB is unavailable,
including fuzzy matching, TTL expiration, and statistics.
"""

import pytest
import time
from datetime import datetime, timedelta


@pytest.mark.unit
class TestFallbackCacheBasics:
    """Basic functionality tests for fallback cache."""

    def test_cache_initialization(self):
        """Test that cache initializes with correct parameters."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        cache = RetrievalFallbackCache(
            max_size=50,
            ttl_seconds=1800,
            similarity_threshold=0.75
        )

        assert cache.max_size == 50
        assert cache.ttl_seconds == 1800
        assert cache.similarity_threshold == 0.75

        stats = cache.get_stats()
        assert stats["cache_size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_cache_stores_results(self):
        """Test that cache stores retrieval results."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        cache = RetrievalFallbackCache()
        results = [
            {"id": "1", "text": "Python programming", "distance": 0.1},
            {"id": "2", "text": "Machine learning basics", "distance": 0.2},
        ]

        cache.cache_results("What is Python?", results)

        stats = cache.get_stats()
        assert stats["cache_size"] == 1

    def test_cache_doesnt_store_empty_results(self):
        """Test that empty results are not cached."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        cache = RetrievalFallbackCache()
        cache.cache_results("test query", [])

        stats = cache.get_stats()
        assert stats["cache_size"] == 0


@pytest.mark.unit
class TestFallbackCacheRetrieval:
    """Tests for retrieving cached results."""

    def test_exact_match_retrieval(self):
        """Test retrieving results with exact query match."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        cache = RetrievalFallbackCache()
        results = [{"id": "1", "text": "test content"}]

        cache.cache_results("test query", results)
        retrieved = cache.get_cached_results("test query")

        assert retrieved is not None
        assert len(retrieved) == 1
        assert retrieved[0]["id"] == "1"

        # Should record a hit
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        cache = RetrievalFallbackCache()
        results = [{"id": "1", "text": "test"}]

        cache.cache_results("Test Query", results)

        # Should match with different casing
        retrieved = cache.get_cached_results("test query")
        assert retrieved is not None
        assert len(retrieved) == 1

    def test_fuzzy_matching(self):
        """Test fuzzy matching for similar queries."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        cache = RetrievalFallbackCache(similarity_threshold=0.8)
        results = [{"id": "1", "text": "Python programming"}]

        cache.cache_results("What is Python programming?", results)

        # Similar query should match
        retrieved = cache.get_cached_results("What is Python programming")
        assert retrieved is not None

        stats = cache.get_stats()
        assert stats["fallback_uses"] == 1

    def test_fuzzy_matching_respects_threshold(self):
        """Test that fuzzy matching respects similarity threshold."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        cache = RetrievalFallbackCache(similarity_threshold=0.9)
        results = [{"id": "1", "text": "test"}]

        cache.cache_results("Python programming", results)

        # Very different query should not match
        retrieved = cache.get_cached_results("JavaScript basics")
        assert retrieved is None

        stats = cache.get_stats()
        assert stats["misses"] == 1

    def test_disable_fuzzy_matching(self):
        """Test retrieving with fuzzy matching disabled."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        cache = RetrievalFallbackCache()
        results = [{"id": "1", "text": "test"}]

        cache.cache_results("Python programming", results)

        # Disable fuzzy matching - should only match exact
        retrieved = cache.get_cached_results("Python programmin", use_fuzzy_match=False)
        assert retrieved is None

    def test_cache_miss(self):
        """Test cache miss for non-existent query."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        cache = RetrievalFallbackCache()

        retrieved = cache.get_cached_results("non-existent query")
        assert retrieved is None

        stats = cache.get_stats()
        assert stats["misses"] == 1


@pytest.mark.unit
class TestFallbackCacheLRU:
    """Tests for LRU (Least Recently Used) behavior."""

    def test_lru_eviction(self):
        """Test that oldest entries are evicted when cache is full."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        cache = RetrievalFallbackCache(max_size=3, similarity_threshold=0.99)

        # Fill cache to capacity with very distinct queries
        cache.cache_results("python programming language tutorial", [{"id": "0"}])
        cache.cache_results("javascript web development guide", [{"id": "1"}])
        cache.cache_results("rust systems programming book", [{"id": "2"}])

        assert cache.get_stats()["cache_size"] == 3

        # Add one more - should evict oldest
        cache.cache_results("java object oriented design", [{"id": "3"}])

        assert cache.get_stats()["cache_size"] == 3

        # First query should be evicted (using exact match to avoid fuzzy)
        assert cache.get_cached_results("python programming language tutorial", use_fuzzy_match=False) is None

    def test_lru_access_updates_order(self):
        """Test that accessing an entry moves it to the end (most recent)."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        cache = RetrievalFallbackCache(max_size=3, similarity_threshold=0.99)

        # Fill cache with distinct queries
        cache.cache_results("python programming language", [{"id": "0"}])
        cache.cache_results("javascript web development", [{"id": "1"}])
        cache.cache_results("rust systems programming", [{"id": "2"}])

        # Access first query to move it to end
        cache.get_cached_results("python programming language", use_fuzzy_match=False)

        # Add new entry - should evict query 1 (now oldest)
        cache.cache_results("java object oriented", [{"id": "3"}])

        # First query should still exist
        assert cache.get_cached_results("python programming language", use_fuzzy_match=False) is not None
        # Second query should be evicted
        assert cache.get_cached_results("javascript web development", use_fuzzy_match=False) is None


@pytest.mark.unit
class TestFallbackCacheTTL:
    """Tests for TTL (Time To Live) expiration."""

    def test_expired_entries_not_returned(self):
        """Test that expired entries are not returned."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        cache = RetrievalFallbackCache(ttl_seconds=1)
        cache.cache_results("test query", [{"id": "1"}])

        # Immediately should be available
        assert cache.get_cached_results("test query") is not None

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get_cached_results("test query") is None

    def test_cleanup_expired_entries(self):
        """Test manual cleanup of expired entries."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        cache = RetrievalFallbackCache(ttl_seconds=1)
        cache.clear()  # Ensure clean state

        # Add some entries
        for i in range(5):
            cache.cache_results(f"query {i}", [{"id": str(i)}])

        assert cache.get_stats()["cache_size"] == 5

        # Wait for expiration
        time.sleep(1.1)

        # Cleanup expired
        removed = cache.cleanup_expired()
        assert removed == 5
        assert cache.get_stats()["cache_size"] == 0

    def test_fuzzy_match_skips_expired(self):
        """Test that fuzzy matching skips expired entries."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        cache = RetrievalFallbackCache(ttl_seconds=1, similarity_threshold=0.8)

        cache.cache_results("Python programming", [{"id": "1"}])

        # Wait for expiration
        time.sleep(1.1)

        # Fuzzy match should not find expired entry
        retrieved = cache.get_cached_results("Python programing")
        assert retrieved is None


@pytest.mark.unit
class TestFallbackCacheFallbackMode:
    """Tests for fallback mode (when ChromaDB is unavailable)."""

    def test_get_fallback_results(self):
        """Test getting fallback results with appropriate logging."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        cache = RetrievalFallbackCache(similarity_threshold=0.75)
        results = [{"id": "1", "text": "cached result"}]

        cache.cache_results("what is python programming language", results)

        # Get fallback results with very similar query (will use fuzzy matching)
        fallback = cache.get_fallback_results("what is python programming")
        assert fallback is not None
        assert len(fallback) == 1

        stats = cache.get_stats()
        # Should have used fuzzy matching, so fallback_uses should be incremented
        assert stats["fallback_uses"] >= 1

    def test_get_fallback_results_uses_fuzzy_match(self):
        """Test that fallback mode always uses fuzzy matching."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        cache = RetrievalFallbackCache(similarity_threshold=0.8)
        results = [{"id": "1", "text": "test"}]

        cache.cache_results("Python programming language", results)

        # Should match with fuzzy
        fallback = cache.get_fallback_results("Python programming")
        assert fallback is not None

    def test_get_fallback_results_returns_none_when_empty(self):
        """Test fallback returns None when no cache available."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        cache = RetrievalFallbackCache()

        fallback = cache.get_fallback_results("non-existent query")
        assert fallback is None


@pytest.mark.unit
class TestFallbackCacheStatistics:
    """Tests for cache statistics and monitoring."""

    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        # Create fresh cache for clean test
        cache = RetrievalFallbackCache(similarity_threshold=0.99)

        # Get initial stats
        initial_stats = cache.get_stats()
        initial_hits = initial_stats["hits"]
        initial_misses = initial_stats["misses"]

        # Add entries
        cache.cache_results("unique query alpha", [{"id": "1"}])
        cache.cache_results("unique query beta", [{"id": "2"}])

        # Generate hits and misses (disable fuzzy to avoid interference)
        cache.get_cached_results("unique query alpha", use_fuzzy_match=False)  # Hit
        cache.get_cached_results("unique query beta", use_fuzzy_match=False)  # Hit
        cache.get_cached_results("unique query gamma", use_fuzzy_match=False)  # Miss

        stats = cache.get_stats()
        assert stats["cache_size"] >= 2
        assert stats["hits"] == initial_hits + 2
        assert stats["misses"] == initial_misses + 1

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        cache = RetrievalFallbackCache()

        # No requests yet
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.0

        # Add and retrieve
        cache.cache_results("query", [{"id": "1"}])
        cache.get_cached_results("query")  # Hit

        stats = cache.get_stats()
        assert stats["hit_rate"] == 1.0

    def test_clear_resets_cache(self):
        """Test that clear removes all entries."""
        from app.retrieval.fallback_cache import RetrievalFallbackCache

        cache = RetrievalFallbackCache()

        # Add entries
        for i in range(5):
            cache.cache_results(f"query {i}", [{"id": str(i)}])

        assert cache.get_stats()["cache_size"] == 5

        # Clear
        cache.clear()

        stats = cache.get_stats()
        assert stats["cache_size"] == 0


@pytest.mark.unit
class TestFallbackCacheGlobalInstance:
    """Tests for global cache instance management."""

    def test_get_fallback_cache_singleton(self):
        """Test that get_fallback_cache returns singleton."""
        from app.retrieval.fallback_cache import get_fallback_cache

        cache1 = get_fallback_cache()
        cache2 = get_fallback_cache()

        # Should be same instance
        assert cache1 is cache2

    def test_global_cache_persists_data(self):
        """Test that global cache persists data across calls."""
        from app.retrieval.fallback_cache import get_fallback_cache

        cache = get_fallback_cache()
        cache.clear()  # Start fresh

        # Add data
        cache.cache_results("test", [{"id": "1"}])

        # Get cache again
        cache2 = get_fallback_cache()

        # Should have the data
        assert cache2.get_stats()["cache_size"] >= 1
