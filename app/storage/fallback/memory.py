"""
============================================================================
FALLBACK IMPLEMENTATION: In-Memory Session Store (High-Availability)
============================================================================

This is the FALLBACK session storage backend that activates when Redis fails.

PURPOSE:
--------
Ensures the application remains operational even when Redis is unavailable,
providing graceful degradation with zero external dependencies.

FEATURES:
---------
* Zero Dependencies: No external services required
* High Concurrency: 16-shard architecture reduces lock contention
* IP Indexing: Fast O(K) lookups for sessions by IP address
* Auto-Cleanup: Background thread removes expired sessions
* Prometheus Metrics: Same observability as Redis backend

LIMITATIONS:
------------
* Non-Persistent: Sessions lost on server restart
* Single-Instance: Cannot be shared across multiple servers
* Memory-Bound: Limited by available RAM

WHEN THIS IS USED:
------------------
1. Redis connection fails during startup
2. Redis URL not provided in configuration
3. Redis library not installed
4. Explicitly set STORAGE_BACKEND=memory (dev/testing)

See: app/storage/factory.py for fallback activation logic
See: app/storage/redis_store.py for primary implementation
"""

import logging
import threading
import time
import uuid
import zlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

from app.settings import settings
from app.storage.base import SessionStore
from app.storage.models import Session
from app.storage.utils import mask_session_id

logger = logging.getLogger(__name__)


class InMemorySessionStore(SessionStore):
    """In-memory session storage with sharding for high concurrency.
    
    Uses 16 shards to reduce lock contention.
    Maintains a separate index for IP lookups to avoid O(N) scans.
    """

    NUM_SHARDS = 16

    def __init__(self):
        """Initialize in-memory store."""
        # Session storage shards: session_id -> Session
        self._shards: List[Dict[str, Session]] = [{} for _ in range(self.NUM_SHARDS)]
        self._shard_locks: List[threading.RLock] = [threading.RLock() for _ in range(self.NUM_SHARDS)]

        # IP index shards: ip_address -> Set[session_id]
        self._ip_index: List[Dict[str, Set[str]]] = [{} for _ in range(self.NUM_SHARDS)]
        self._ip_locks: List[threading.RLock] = [threading.RLock() for _ in range(self.NUM_SHARDS)]

        # Prometheus metrics
        try:
            from prometheus_client import Counter, Gauge
            self._metrics_enabled = True
            self.session_count_metric = Gauge(
                'sessions_active', 
                'Number of active sessions'
            )
            self.session_creates_metric = Counter(
                'sessions_created_total', 
                'Total sessions created'
            )
            self.session_cleanups_metric = Counter(
                'sessions_expired_total', 
                'Total expired sessions'
            )
            self.memory_usage_metric = Gauge(
                'sessions_memory_mb', 
                'Estimated memory usage in MB'
            )
        except ImportError:
            logger.warning("Prometheus client not available, metrics disabled")
            self._metrics_enabled = False

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="SessionCleanup"
        )
        self._cleanup_thread.start()
        logger.info(f"Started sharded in-memory session store ({self.NUM_SHARDS} shards)")

    def _get_shard_index(self, key: str) -> int:
        """Get shard index for a string key."""
        return zlib.crc32(key.encode()) % self.NUM_SHARDS

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve session by ID."""
        shard_idx = self._get_shard_index(session_id)
        with self._shard_locks[shard_idx]:
            return self._shards[shard_idx].get(session_id)

    def create_session(
        self,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Session:
        """Create new session."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        now = datetime.now()
        session = Session(
            session_id=session_id,
            created_at=now,
            last_accessed=now,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        # Add to session storage
        shard_idx = self._get_shard_index(session_id)
        with self._shard_locks[shard_idx]:
            self._shards[shard_idx][session_id] = session

        # Add to IP index if IP is present
        if ip_address:
            ip_shard_idx = self._get_shard_index(ip_address)
            with self._ip_locks[ip_shard_idx]:
                if ip_address not in self._ip_index[ip_shard_idx]:
                    self._ip_index[ip_shard_idx][ip_address] = set()
                self._ip_index[ip_shard_idx][ip_address].add(session_id)

        logger.debug(f"Created session {mask_session_id(session_id)} (Shard {shard_idx})")
        
        # Update metrics
        if self._metrics_enabled:
            self.session_creates_metric.inc()
            self.session_count_metric.set(self.get_session_count())
        
        return session

    def update_session(self, session: Session) -> None:
        """Update existing session."""
        shard_idx = self._get_shard_index(session.session_id)
        with self._shard_locks[shard_idx]:
            self._shards[shard_idx][session.session_id] = session

    def delete_session(self, session_id: str) -> None:
        """Delete session and update indexes."""
        shard_idx = self._get_shard_index(session_id)
        session = None
        
        # Remove from storage
        with self._shard_locks[shard_idx]:
            if session_id in self._shards[shard_idx]:
                session = self._shards[shard_idx].pop(session_id)

        # Remove from IP index if needed
        if session and session.ip_address:
            ip_shard_idx = self._get_shard_index(session.ip_address)
            with self._ip_locks[ip_shard_idx]:
                if session.ip_address in self._ip_index[ip_shard_idx]:
                    self._ip_index[ip_shard_idx][session.ip_address].discard(session_id)
                    # Clean up empty sets
                    if not self._ip_index[ip_shard_idx][session.ip_address]:
                        del self._ip_index[ip_shard_idx][session.ip_address]

        if session:
            logger.debug(f"Deleted session {mask_session_id(session_id)}")

    def get_session_count(self) -> int:
        """Get total number of active sessions across all shards."""
        count = 0
        for i in range(self.NUM_SHARDS):
            with self._shard_locks[i]:
                count += len(self._shards[i])
        return count

    def get_sessions_by_ip(self, ip_address: str) -> List[Session]:
        """Get all sessions for an IP address using the index."""
        ip_shard_idx = self._get_shard_index(ip_address)
        session_ids = set()

        # 1. Get session IDs from index (fast)
        with self._ip_locks[ip_shard_idx]:
            if ip_address in self._ip_index[ip_shard_idx]:
                session_ids = self._ip_index[ip_shard_idx][ip_address].copy()

        # 2. Retrieve actual sessions
        sessions = []
        for sid in session_ids:
            session = self.get_session(sid)
            if session:
                sessions.append(session)
        
        return sessions

    def get_memory_stats(self) -> Dict[str, any]:
        """Get memory usage statistics.
        
        Returns:
            Dictionary with session counts and estimated memory usage
        """
        import sys
        
        total_sessions = self.get_session_count()
        
        # Estimate memory usage based on sample session
        sample_session = None
        for shard in self._shards:
            if shard:
                sample_session = next(iter(shard.values()), None)
                break
        
        avg_session_size = sys.getsizeof(sample_session) if sample_session else 1000
        estimated_memory_mb = (total_sessions * avg_session_size) / (1024 * 1024)
        
        return {
            "total_sessions": total_sessions,
            "estimated_memory_mb": round(estimated_memory_mb, 2),
            "shards": self.NUM_SHARDS
        }

    def _cleanup_loop(self) -> None:
        """Background thread to clean up expired sessions."""
        interval = settings.session.cleanup_interval_seconds

        while True:
            try:
                time.sleep(interval)
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")

    def _cleanup_expired(self) -> None:
        """Remove expired sessions from all shards in parallel."""
        import concurrent.futures
        
        now = datetime.now()
        ttl = timedelta(seconds=settings.session.ttl_seconds)
        
        def cleanup_shard(shard_idx: int) -> int:
            """Cleanup single shard, return count of expired sessions."""
            expired = []
            
            # Identify expired sessions in this shard
            with self._shard_locks[shard_idx]:
                for session_id, session in self._shards[shard_idx].items():
                    age = now - session.last_accessed
                    if age > ttl:
                        expired.append(session)
            
            # Delete them (this handles index cleanup too)
            for session in expired:
                self.delete_session(session.session_id)
            
            return len(expired)
        
        # Process all shards concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.NUM_SHARDS) as executor:
            futures = [executor.submit(cleanup_shard, i) for i in range(self.NUM_SHARDS)]
            total_expired = sum(f.result() for f in concurrent.futures.as_completed(futures))
        
        if total_expired > 0:
            logger.info(f"Cleaned up {total_expired} expired sessions")
            
            # Update metrics
            if self._metrics_enabled:
                self.session_cleanups_metric.inc(total_expired)
                self.session_count_metric.set(self.get_session_count())
                stats = self.get_memory_stats()
                self.memory_usage_metric.set(stats["estimated_memory_mb"])
