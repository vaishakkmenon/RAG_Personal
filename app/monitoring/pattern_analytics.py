"""
Pattern Analytics - Track pattern effectiveness and retrieval improvements

Tracks metrics for each pattern:
- Match rate (how often pattern matches)
- Success rate (how often retrieval succeeds after rewrite)
- Average latency (performance monitoring)
- Average retrieval distance (quality improvement)
- Distance improvements (before/after comparison)

Stores metrics in JSON file for persistence.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class PatternAnalytics:
    """Track pattern effectiveness and retrieval improvements."""

    def __init__(self, storage_path: Optional[str] = None, settings_obj: Optional[object] = None):
        """
        Initialize pattern analytics.

        Args:
            storage_path: Path to analytics JSON file (overrides settings)
            settings_obj: Settings object (optional, for dependency injection)
        """
        # Import settings here to avoid circular imports
        if settings_obj is None:
            from app.settings import settings as settings_obj

        self.settings = settings_obj
        self.storage_path = storage_path or self.settings.query_rewriter.analytics_storage_path
        self.metrics: Dict = self._load_metrics()
        self.save_counter = 0
        self.save_frequency = 10  # Save every 10 queries

    def _load_metrics(self) -> Dict:
        """
        Load metrics from JSON file.

        Returns:
            Metrics dict with pattern-specific data
        """
        storage_file = Path(self.storage_path)
        if storage_file.exists():
            try:
                with open(storage_file, encoding='utf-8') as f:
                    metrics = json.load(f)
                    logger.info(f"Loaded analytics from {self.storage_path}")
                    return metrics
            except Exception as e:
                logger.error(f"Failed to load analytics: {e}")

        # Initialize with empty structure
        return {
            '_total': {'total_queries': 0},
            '_no_match': {'total_queries': 0}
        }

    def _save_metrics(self):
        """Save metrics to JSON file."""
        try:
            storage_file = Path(self.storage_path)
            storage_file.parent.mkdir(parents=True, exist_ok=True)

            with open(storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2)

            logger.debug(f"Saved analytics to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save analytics: {e}")

    def log_query(
        self,
        query: str,
        rewrite_metadata: Optional[object],
        retrieval_distance: Optional[float],
        grounded: bool
    ):
        """
        Log a query outcome for analytics.

        Args:
            query: Original user query
            rewrite_metadata: RewriteMetadata object (or None if no rewrite)
            retrieval_distance: Best retrieval distance (or None if no results)
            grounded: Whether response was grounded in documents
        """
        # Update total queries
        self.metrics['_total']['total_queries'] = self.metrics['_total'].get('total_queries', 0) + 1

        if rewrite_metadata is None:
            # No pattern matched
            self.metrics['_no_match']['total_queries'] = self.metrics['_no_match'].get('total_queries', 0) + 1
        else:
            # Pattern matched - track metrics
            pattern_name = rewrite_metadata.pattern_name

            # Initialize pattern metrics if first time
            if pattern_name not in self.metrics:
                self.metrics[pattern_name] = {
                    'total_matches': 0,
                    'grounded_responses': 0,
                    'failed_responses': 0,
                    'total_latency_ms': 0.0,
                    'distances': [],
                    'last_updated': None
                }

            pattern_metrics = self.metrics[pattern_name]
            pattern_metrics['total_matches'] += 1
            pattern_metrics['total_latency_ms'] += rewrite_metadata.latency_ms

            # Track success/failure
            if grounded:
                pattern_metrics['grounded_responses'] += 1
            else:
                pattern_metrics['failed_responses'] += 1

            # Track retrieval distance
            if retrieval_distance is not None:
                pattern_metrics['distances'].append(retrieval_distance)

            # Update timestamp
            pattern_metrics['last_updated'] = datetime.now().isoformat()

            logger.debug(
                f"Logged analytics for pattern '{pattern_name}': "
                f"grounded={grounded}, distance={retrieval_distance}"
            )

        # Periodic save (every N queries)
        self.save_counter += 1
        if self.save_counter >= self.save_frequency:
            self._save_metrics()
            self.save_counter = 0

    def get_pattern_effectiveness(self, pattern_name: str) -> Dict:
        """
        Get effectiveness metrics for a specific pattern.

        Args:
            pattern_name: Name of the pattern

        Returns:
            Dict with effectiveness metrics
        """
        if pattern_name not in self.metrics:
            return {}

        metrics = self.metrics[pattern_name]
        total_queries = self.metrics['_total']['total_queries']

        return {
            'pattern_name': pattern_name,
            'total_matches': metrics['total_matches'],
            'match_rate': metrics['total_matches'] / total_queries if total_queries > 0 else 0.0,
            'success_rate': metrics['grounded_responses'] / metrics['total_matches'] if metrics['total_matches'] > 0 else 0.0,
            'avg_latency_ms': metrics['total_latency_ms'] / metrics['total_matches'] if metrics['total_matches'] > 0 else 0.0,
            'avg_distance': sum(metrics['distances']) / len(metrics['distances']) if metrics['distances'] else None,
            'last_updated': metrics['last_updated']
        }

    def get_all_pattern_effectiveness(self) -> List[Dict]:
        """
        Get effectiveness metrics for all patterns.

        Returns:
            List of pattern effectiveness dicts, sorted by match count
        """
        pattern_stats = []

        for pattern_name in self.metrics:
            if pattern_name.startswith('_'):
                continue  # Skip special keys

            stats = self.get_pattern_effectiveness(pattern_name)
            if stats:
                pattern_stats.append(stats)

        # Sort by match count (descending)
        pattern_stats.sort(key=lambda x: x['total_matches'], reverse=True)

        return pattern_stats

    def generate_report(self) -> str:
        """
        Generate human-readable effectiveness report.

        Returns:
            ASCII-formatted report string
        """
        report_lines = [
            "PATTERN EFFECTIVENESS REPORT",
            "=" * 80,
            f"Total Queries: {self.metrics['_total']['total_queries']}",
            f"No Pattern Match: {self.metrics['_no_match']['total_queries']} "
            f"({self.metrics['_no_match']['total_queries'] / max(self.metrics['_total']['total_queries'], 1) * 100:.1f}%)",
            "",
            f"{'Pattern':<30} {'Matches':<10} {'Success':<10} {'Latency':<12} {'Avg Dist':<10}",
            "-" * 80
        ]

        # Get all patterns sorted by match count
        pattern_stats = self.get_all_pattern_effectiveness()

        for stats in pattern_stats:
            avg_dist_str = f"{stats['avg_distance']:.3f}" if stats['avg_distance'] is not None else "N/A"

            report_lines.append(
                f"{stats['pattern_name']:<30} "
                f"{stats['total_matches']:<10} "
                f"{stats['success_rate']*100:>6.1f}%   "
                f"{stats['avg_latency_ms']:>8.1f}ms   "
                f"{avg_dist_str:<10}"
            )

        report_lines.append("=" * 80)
        return "\n".join(report_lines)

    def force_save(self):
        """Force save metrics to disk (useful at shutdown)."""
        self._save_metrics()
        logger.info("Forced save of analytics metrics")


# ============================================================================
# Singleton Instance Management
# ============================================================================

_pattern_analytics_instance: Optional[PatternAnalytics] = None


def get_pattern_analytics() -> PatternAnalytics:
    """
    Get singleton PatternAnalytics instance.

    Returns:
        PatternAnalytics instance
    """
    global _pattern_analytics_instance
    if _pattern_analytics_instance is None:
        _pattern_analytics_instance = PatternAnalytics()
        logger.debug("Created PatternAnalytics singleton instance")
    return _pattern_analytics_instance


def reset_pattern_analytics():
    """
    Reset singleton instance (useful for testing).

    This will force reload of metrics on next get_pattern_analytics() call.
    """
    global _pattern_analytics_instance
    if _pattern_analytics_instance is not None:
        _pattern_analytics_instance.force_save()  # Save before reset
    _pattern_analytics_instance = None
    logger.debug("Reset PatternAnalytics singleton instance")
