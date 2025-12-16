"""
Pattern Suggester - Analyze failed queries and suggest new patterns

Analyzes queries that fail to retrieve good results and clusters them
to identify opportunities for new query rewriting patterns.

Tracks:
- Failed queries (distance > threshold OR grounded = False)
- Keyword-based clustering
- Pattern suggestions (minimum 3 queries per cluster)
- Timestamp tracking for trend analysis

Stores failed queries in JSON file for persistence.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class PatternSuggester:
    """Analyze failed queries and suggest new patterns."""

    def __init__(self, storage_path: Optional[str] = None, settings_obj: Optional[object] = None):
        """
        Initialize pattern suggester.

        Args:
            storage_path: Path to failed queries JSON file (overrides settings)
            settings_obj: Settings object (optional, for dependency injection)
        """
        # Import settings here to avoid circular imports
        if settings_obj is None:
            from app.settings import settings as settings_obj

        self.settings = settings_obj
        self.storage_path = storage_path or self.settings.query_rewriter.failed_queries_storage_path
        self.failed_queries: List[Dict] = self._load_failed_queries()
        self.max_failed_queries = 1000
        self.save_counter = 0
        self.save_frequency = 10  # Save every 10 captures

    def _load_failed_queries(self) -> List[Dict]:
        """
        Load failed queries from JSON file.

        Returns:
            List of failed query dicts
        """
        storage_file = Path(self.storage_path)
        if storage_file.exists():
            try:
                with open(storage_file, encoding='utf-8') as f:
                    queries = json.load(f)
                    logger.info(f"Loaded {len(queries)} failed queries from {self.storage_path}")
                    return queries
            except Exception as e:
                logger.error(f"Failed to load failed queries: {e}")

        return []

    def _save_failed_queries(self):
        """Save failed queries to JSON file."""
        try:
            storage_file = Path(self.storage_path)
            storage_file.parent.mkdir(parents=True, exist_ok=True)

            # Keep only the most recent N queries
            queries_to_save = self.failed_queries[-self.max_failed_queries:]

            with open(storage_file, 'w', encoding='utf-8') as f:
                json.dump(queries_to_save, f, indent=2)

            logger.debug(f"Saved {len(queries_to_save)} failed queries to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save failed queries: {e}")

    def capture_failed_query(
        self,
        query: str,
        distance: Optional[float],
        pattern_matched: Optional[str],
        grounded: bool
    ):
        """
        Capture a failed query for analysis.

        Args:
            query: Original user query
            distance: Best retrieval distance (or None if no results)
            pattern_matched: Pattern that matched (or None if no match)
            grounded: Whether response was grounded
        """
        # Check if this qualifies as a failure
        distance_threshold = self.settings.query_rewriter.capture_failed_threshold

        is_failure = (
            not grounded or
            (distance is not None and distance > distance_threshold) or
            distance is None
        )

        if not is_failure:
            return

        # Capture the failed query
        self.failed_queries.append({
            'query': query,
            'distance': distance,
            'pattern_matched': pattern_matched,
            'grounded': grounded,
            'timestamp': datetime.now().isoformat()
        })

        logger.debug(
            f"Captured failed query: '{query[:50]}...' "
            f"(distance: {distance}, grounded: {grounded})"
        )

        # Periodic save
        self.save_counter += 1
        if self.save_counter >= self.save_frequency:
            self._save_failed_queries()
            self.save_counter = 0

    def analyze_and_suggest(self, min_cluster_size: int = 3) -> Dict:
        """
        Analyze failed queries and generate pattern suggestions.

        Args:
            min_cluster_size: Minimum number of queries to form a cluster

        Returns:
            Dict with analysis results and pattern suggestions
        """
        if not self.failed_queries:
            return {
                'total_failed': 0,
                'clusters': [],
                'suggestions': []
            }

        # Keyword-based clustering
        keyword_clusters = defaultdict(list)

        for failed in self.failed_queries:
            query = failed['query']
            query_lower = query.lower()

            # Identify query patterns based on keywords
            clustered = False

            # Counting queries
            if any(kw in query_lower for kw in ['how many', 'count', 'number of']):
                keyword_clusters['counting'].append(failed)
                clustered = True

            # Temporal queries
            elif any(kw in query_lower for kw in ['when did', 'when was', 'what year', 'what date']):
                keyword_clusters['temporal_past'].append(failed)
                clustered = True
            elif any(kw in query_lower for kw in ['when does', 'when will', 'expire', 'expiration']):
                keyword_clusters['temporal_future'].append(failed)
                clustered = True

            # Comparison queries
            elif any(kw in query_lower for kw in ['compare', 'versus', 'vs', 'difference between', 'better than']):
                keyword_clusters['comparison'].append(failed)
                clustered = True

            # List queries
            elif any(kw in query_lower for kw in ['list all', 'show all', 'what are all', 'give me all']):
                keyword_clusters['list_request'].append(failed)
                clustered = True

            # Ranking/superlative queries
            elif any(kw in query_lower for kw in ['best', 'worst', 'highest', 'lowest', 'most', 'least']):
                keyword_clusters['ranking'].append(failed)
                clustered = True

            # Attribute queries
            elif any(kw in query_lower for kw in ['what grade', 'what score', 'what gpa', 'which courses']):
                keyword_clusters['attribute_filter'].append(failed)
                clustered = True

            # Existence/verification queries
            elif any(kw in query_lower for kw in ['do i have', 'did i take', 'have i done', 'did i get']):
                keyword_clusters['existence_check'].append(failed)
                clustered = True

            # If no cluster matched, put in unknown
            if not clustered:
                keyword_clusters['unknown'].append(failed)

        # Generate suggestions for clusters with enough queries
        suggestions = []

        for cluster_name, queries in keyword_clusters.items():
            if len(queries) >= min_cluster_size:
                # Calculate average distance for this cluster
                distances = [q['distance'] for q in queries if q['distance'] is not None]
                avg_distance = sum(distances) / len(distances) if distances else None

                # Get example queries (up to 5)
                example_queries = [q['query'] for q in queries[:5]]

                # Check if any pattern already matched these queries
                patterns_that_matched = [q['pattern_matched'] for q in queries if q['pattern_matched']]
                pattern_coverage = len(patterns_that_matched) / len(queries) if queries else 0.0

                suggestions.append({
                    'cluster_type': cluster_name,
                    'frequency': len(queries),
                    'example_queries': example_queries,
                    'avg_distance': avg_distance,
                    'pattern_coverage': pattern_coverage,
                    'suggestion': self._generate_suggestion_text(cluster_name, len(queries), pattern_coverage)
                })

        # Sort suggestions by frequency (descending)
        suggestions.sort(key=lambda x: x['frequency'], reverse=True)

        return {
            'total_failed': len(self.failed_queries),
            'clusters': list(keyword_clusters.keys()),
            'cluster_sizes': {k: len(v) for k, v in keyword_clusters.items()},
            'suggestions': suggestions
        }

    def _generate_suggestion_text(
        self,
        cluster_type: str,
        frequency: int,
        pattern_coverage: float
    ) -> str:
        """
        Generate human-readable suggestion text.

        Args:
            cluster_type: Type of cluster
            frequency: Number of queries in cluster
            pattern_coverage: Fraction of queries with pattern matches

        Returns:
            Suggestion text
        """
        if pattern_coverage > 0.7:
            return (
                f"Pattern exists for '{cluster_type}' but failing for {frequency} queries. "
                f"Consider improving the existing pattern or rewrite strategy."
            )
        elif pattern_coverage > 0.3:
            return (
                f"Partial pattern coverage for '{cluster_type}' ({pattern_coverage*100:.0f}%). "
                f"Consider expanding pattern rules to cover {frequency} failed queries."
            )
        else:
            return (
                f"No pattern coverage for '{cluster_type}' ({frequency} queries). "
                f"Consider adding a new pattern to handle this query type."
            )

    def generate_report(self) -> str:
        """
        Generate human-readable failed query report.

        Returns:
            ASCII-formatted report string
        """
        analysis = self.analyze_and_suggest()

        report_lines = [
            "FAILED QUERY ANALYSIS REPORT",
            "=" * 80,
            f"Total Failed Queries Captured: {analysis['total_failed']}",
            f"Query Clusters Identified: {len(analysis['clusters'])}",
            "",
            "Cluster Sizes:",
            "-" * 80
        ]

        # Show cluster sizes
        for cluster_name, size in sorted(
            analysis['cluster_sizes'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            report_lines.append(f"  {cluster_name:<25} {size:>5} queries")

        report_lines.extend([
            "",
            "Pattern Suggestions:",
            "=" * 80
        ])

        if not analysis['suggestions']:
            report_lines.append("No suggestions available (need at least 3 queries per cluster)")
        else:
            for i, suggestion in enumerate(analysis['suggestions'], 1):
                avg_dist_str = f"{suggestion['avg_distance']:.3f}" if suggestion['avg_distance'] is not None else "N/A"

                report_lines.extend([
                    f"\n{i}. {suggestion['cluster_type'].upper()} ({suggestion['frequency']} queries)",
                    f"   Coverage: {suggestion['pattern_coverage']*100:.0f}% | Avg Distance: {avg_dist_str}",
                    f"   Suggestion: {suggestion['suggestion']}",
                    "   Example queries:"
                ])

                for example in suggestion['example_queries']:
                    report_lines.append(f"     - {example}")

        report_lines.append("=" * 80)
        return "\n".join(report_lines)

    def force_save(self):
        """Force save failed queries to disk (useful at shutdown)."""
        self._save_failed_queries()
        logger.info("Forced save of failed queries")


# ============================================================================
# Singleton Instance Management
# ============================================================================

_pattern_suggester_instance: Optional[PatternSuggester] = None


def get_pattern_suggester() -> PatternSuggester:
    """
    Get singleton PatternSuggester instance.

    Returns:
        PatternSuggester instance
    """
    global _pattern_suggester_instance
    if _pattern_suggester_instance is None:
        _pattern_suggester_instance = PatternSuggester()
        logger.debug("Created PatternSuggester singleton instance")
    return _pattern_suggester_instance


def reset_pattern_suggester():
    """
    Reset singleton instance (useful for testing).

    This will force reload of failed queries on next get_pattern_suggester() call.
    """
    global _pattern_suggester_instance
    if _pattern_suggester_instance is not None:
        _pattern_suggester_instance.force_save()  # Save before reset
    _pattern_suggester_instance = None
    logger.debug("Reset PatternSuggester singleton instance")
