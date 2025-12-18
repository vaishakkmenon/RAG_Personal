"""
Query Rewriter - Pattern-based query rewriting engine

Main entry point for query rewriting with:
- YAML configuration loading
- Hot-reload support
- Pattern matching orchestration
- Latency tracking
- Singleton instance management
"""

import time
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from app.retrieval.pattern_matcher import PatternMatcher

if TYPE_CHECKING:
    from app.models import RewriteMetadata

logger = logging.getLogger(__name__)

# Import query rewriting metrics
try:
    from app.metrics import (
        rag_query_rewrite_total,
        rag_query_rewrite_pattern_matches_total,
        rag_query_rewrite_latency_seconds,
    )

    REWRITE_METRICS_ENABLED = True
except ImportError:
    REWRITE_METRICS_ENABLED = False
    logger.warning("Query rewrite metrics not available")


class QueryRewriter:
    """Pattern-based query rewriting engine with hot-reload support."""

    def __init__(
        self, config_path: Optional[str] = None, settings_obj: Optional[object] = None
    ):
        """
        Initialize query rewriter.

        Args:
            config_path: Path to pattern YAML config (overrides settings)
            settings_obj: Settings object (optional, for dependency injection)
        """
        # Import settings here to avoid circular imports
        if settings_obj is None:
            from app.settings import settings as settings_obj

        self.settings = settings_obj
        self.config_path = (
            config_path or self.settings.query_rewriter.pattern_config_path
        )
        self.patterns: List[PatternMatcher] = []
        self.enabled = self.settings.query_rewriter.enabled
        self.hot_reload_interval = (
            self.settings.query_rewriter.hot_reload_interval_seconds
        )
        self.last_reload_time = 0

        # Load patterns on initialization
        self._load_config()

    def _load_config(self):
        """
        Load pattern configuration from YAML file.

        Creates PatternMatcher instances for all enabled patterns
        and sorts them by priority (highest first).
        """
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"Pattern config not found: {self.config_path}")
                logger.warning("Query rewriting will be disabled")
                self.patterns = []
                return

            with open(config_file, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if not config or "patterns" not in config:
                logger.error("Invalid pattern config: missing 'patterns' key")
                self.patterns = []
                return

            # Create PatternMatcher instances for enabled patterns
            pattern_configs = config.get("patterns", [])
            enabled_patterns = [p for p in pattern_configs if p.get("enabled", True)]

            # Sort by priority (highest first)
            enabled_patterns.sort(key=lambda p: p.get("priority", 0), reverse=True)

            self.patterns = [PatternMatcher(p) for p in enabled_patterns]
            self.last_reload_time = time.time()

            logger.info(
                f"Loaded {len(self.patterns)} query rewriting patterns from {self.config_path}"
            )

            # Log pattern names and priorities
            if self.patterns:
                pattern_summary = [f"{p.name}({p.priority})" for p in self.patterns[:5]]
                logger.debug(f"Top patterns: {', '.join(pattern_summary)}")

        except yaml.YAMLError as e:
            logger.error(f"YAML syntax error in pattern config: {e}")
            self.patterns = []
        except Exception as e:
            logger.error(f"Failed to load pattern config: {e}", exc_info=True)
            self.patterns = []

    def _check_reload(self):
        """
        Check if config should be hot-reloaded.

        Reloads configuration if:
        - Hot-reload is enabled in settings
        - Sufficient time has elapsed since last reload
        """
        if not self.settings.query_rewriter.hot_reload:
            return

        elapsed = time.time() - self.last_reload_time
        if elapsed >= self.hot_reload_interval:
            logger.info("Hot-reloading pattern configuration...")
            old_count = len(self.patterns)
            self._load_config()
            new_count = len(self.patterns)

            if old_count != new_count:
                logger.info(f"Pattern count changed: {old_count} -> {new_count}")

    def rewrite_query(
        self, query: str, metadata_filter: Optional[Dict] = None
    ) -> Tuple[str, Optional["RewriteMetadata"]]:
        """
        Main entry point for query rewriting.

        Tries patterns in priority order (first match wins).
        Tracks latency and validates against max_latency constraint.

        Args:
            query: Original user query
            metadata_filter: Optional metadata filter dict (for future enhancements)

        Returns:
            (rewritten_query, RewriteMetadata or None) tuple
        """
        start_time = time.time()

        # Check if enabled
        if not self.enabled:
            logger.debug("Query rewriting is disabled")
            return query, None

        if not self.patterns:
            logger.debug("No patterns loaded, skipping rewrite")
            return query, None

        # Hot-reload check
        self._check_reload()

        # Try each pattern in priority order (first match wins)
        for pattern_matcher in self.patterns:
            match_result = pattern_matcher.match(query)

            if match_result.matched:
                # Apply rewrite transformation
                rewritten_query, rewrite_metadata_dict = pattern_matcher.apply_rewrite(
                    query, match_result
                )

                latency_ms = (time.time() - start_time) * 1000

                # Import RewriteMetadata here to avoid circular imports
                from app.models import RewriteMetadata

                # Build RewriteMetadata
                metadata = RewriteMetadata(
                    original_query=query,
                    rewritten_query=rewritten_query,
                    pattern_name=pattern_matcher.name,
                    pattern_type=pattern_matcher.matching_config["type"],
                    matched_entities=match_result.extracted_entities or {},
                    rewrite_hint=rewrite_metadata_dict.get("hint"),
                    metadata_filter_addition=rewrite_metadata_dict.get(
                        "metadata_filter"
                    ),
                    latency_ms=latency_ms,
                    confidence=match_result.score,
                )

                # Check latency constraint
                max_latency = self.settings.query_rewriter.max_latency_ms
                if latency_ms > max_latency:
                    logger.warning(
                        f"Query rewriting exceeded max latency: {latency_ms:.2f}ms > {max_latency}ms "
                        f"(pattern: {pattern_matcher.name})"
                    )

                logger.info(
                    f"Query matched pattern '{pattern_matcher.name}' "
                    f"(latency: {latency_ms:.2f}ms, confidence: {match_result.score:.2f})"
                )

                # Track pattern match metrics
                if REWRITE_METRICS_ENABLED:
                    rag_query_rewrite_total.labels(matched="true").inc()
                    rag_query_rewrite_pattern_matches_total.labels(
                        pattern_name=pattern_matcher.name
                    ).inc()
                    rag_query_rewrite_latency_seconds.observe(latency_ms / 1000.0)

                return rewritten_query, metadata

        # No pattern matched
        latency_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"No pattern matched query: '{query[:50]}...' (latency: {latency_ms:.2f}ms)"
        )

        # Track no-match metrics
        if REWRITE_METRICS_ENABLED:
            rag_query_rewrite_total.labels(matched="false").inc()
            rag_query_rewrite_latency_seconds.observe(latency_ms / 1000.0)

        return query, None


# ============================================================================
# Singleton Instance Management
# ============================================================================

_query_rewriter_instance: Optional[QueryRewriter] = None


def get_query_rewriter() -> QueryRewriter:
    """
    Get singleton QueryRewriter instance.

    Returns:
        QueryRewriter instance
    """
    global _query_rewriter_instance
    if _query_rewriter_instance is None:
        _query_rewriter_instance = QueryRewriter()
        logger.debug("Created QueryRewriter singleton instance")
    return _query_rewriter_instance


def reset_query_rewriter():
    """
    Reset singleton instance (useful for testing).

    This will force reload of configuration on next get_query_rewriter() call.
    """
    global _query_rewriter_instance
    _query_rewriter_instance = None
    logger.debug("Reset QueryRewriter singleton instance")
