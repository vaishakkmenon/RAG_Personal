"""
Tests for adaptive threshold calculation for entity existence detection.

Tests all three threshold methods:
- Gap-based threshold
- Percentile-based threshold
- Context-aware threshold
"""

import pytest
from unittest.mock import patch

from app.retrieval.adaptive_threshold import (
    calculate_percentile_threshold,
    calculate_gap_based_threshold,
    calculate_context_aware_threshold,
    check_entity_exists_adaptive,
)


@pytest.mark.unit
class TestPercentileThreshold:
    """Tests for percentile-based threshold calculation."""

    @patch("app.retrieval.adaptive_threshold.search")
    def test_percentile_with_results(self, mock_search):
        """Test percentile calculation with normal results."""
        # Mock search results with varying distances
        mock_search.return_value = [
            {"distance": 0.20, "content": "chunk1"},
            {"distance": 0.30, "content": "chunk2"},
            {"distance": 0.40, "content": "chunk3"},
            {"distance": 0.50, "content": "chunk4"},
            {"distance": 0.60, "content": "chunk5"},
        ]

        best_distance, threshold = calculate_percentile_threshold(
            entity="Python", sample_size=5, percentile=0.85
        )

        assert best_distance == 0.20
        # Percentile at 85% of 5 items = index 4, which is 0.60 after sorting
        assert threshold == 0.60

    @patch("app.retrieval.adaptive_threshold.search")
    def test_percentile_with_no_results(self, mock_search):
        """Test percentile calculation when no results found."""
        mock_search.return_value = []

        best_distance, threshold = calculate_percentile_threshold("NonExistent")

        assert best_distance == 1.0
        assert threshold == 0.85

    @patch("app.retrieval.adaptive_threshold.search")
    def test_percentile_with_single_result(self, mock_search):
        """Test percentile calculation with only one result."""
        mock_search.return_value = [{"distance": 0.35, "content": "chunk1"}]

        best_distance, threshold = calculate_percentile_threshold(
            entity="Rare", sample_size=10, percentile=0.85
        )

        assert best_distance == 0.35
        assert threshold <= 1.0  # Should handle edge case gracefully


@pytest.mark.unit
class TestGapBasedThreshold:
    """Tests for gap-based threshold calculation."""

    @patch("app.retrieval.adaptive_threshold.search")
    def test_strong_match_low_distance(self, mock_search):
        """Test strong match with very low distance (<0.30)."""
        mock_search.return_value = [
            {"distance": 0.25, "content": "Python programming"},
            {"distance": 0.28, "content": "Python language"},
        ]

        exists, distance, reason = calculate_gap_based_threshold("Python")

        assert exists is True
        assert distance == 0.25
        assert "Strong match" in reason

    @patch("app.retrieval.adaptive_threshold.search")
    def test_weak_match_high_distance(self, mock_search):
        """Test weak match with high distance (>0.42)."""
        mock_search.return_value = [
            {"distance": 0.50, "content": "unrelated content"},
            {"distance": 0.55, "content": "also unrelated"},
        ]

        exists, distance, reason = calculate_gap_based_threshold("NonExistent")

        assert exists is False
        assert distance == 0.50
        assert "Weak match" in reason

    @patch("app.retrieval.adaptive_threshold.search")
    def test_consistent_results_small_gap(self, mock_search):
        """Test entity existence with small gap between top results."""
        mock_search.return_value = [
            {"distance": 0.35, "content": "Docker containers"},
            {"distance": 0.37, "content": "Docker images"},  # gap = 0.02
            {"distance": 0.45, "content": "Docker compose"},
        ]

        exists, distance, reason = calculate_gap_based_threshold("Docker")

        assert exists is True
        assert distance == 0.35
        assert "Consistent results" in reason

    @patch("app.retrieval.adaptive_threshold.search")
    def test_inconsistent_results_large_gap(self, mock_search):
        """Test entity doesn't exist with large gap between results."""
        mock_search.return_value = [
            {"distance": 0.38, "content": "something"},
            {"distance": 0.55, "content": "unrelated"},  # gap = 0.17
        ]

        exists, distance, reason = calculate_gap_based_threshold("Fake")

        assert exists is False
        assert distance == 0.38
        assert "Inconsistent results" in reason

    @patch("app.retrieval.adaptive_threshold.search")
    def test_moderate_match_gray_area(self, mock_search):
        """Test moderate match in gray area."""
        mock_search.return_value = [
            {"distance": 0.36, "content": "content1"},
            {
                "distance": 0.50,
                "content": "content2",
            },  # gap > 0.08, will go to gray area logic
        ]

        exists, distance, reason = calculate_gap_based_threshold("Moderate")

        assert exists is True
        assert distance == 0.36
        assert "Moderate match" in reason

    @patch("app.retrieval.adaptive_threshold.search")
    def test_below_threshold_gray_area(self, mock_search):
        """Test below threshold in gray area."""
        mock_search.return_value = [
            {"distance": 0.39, "content": "content1"},
            {"distance": 0.45, "content": "content2"},
        ]

        exists, distance, reason = calculate_gap_based_threshold("Below")

        assert exists is False
        assert distance == 0.39
        assert "Below threshold" in reason

    @patch("app.retrieval.adaptive_threshold.search")
    def test_no_results(self, mock_search):
        """Test when search returns no results."""
        mock_search.return_value = []

        exists, distance, reason = calculate_gap_based_threshold("Nothing")

        assert exists is False
        assert distance == 1.0
        assert "No results found" in reason


@pytest.mark.unit
class TestContextAwareThreshold:
    """Tests for context-aware threshold calculation."""

    @patch("app.retrieval.adaptive_threshold.search")
    def test_acronym_detection(self, mock_search):
        """Test acronym entity type detection and threshold."""
        mock_search.return_value = [
            {"distance": 0.37, "content": "Certified Kubernetes Administrator"},
        ]

        exists, distance, reason = calculate_context_aware_threshold("CKA")

        assert distance == 0.37
        assert "Type=acronym" in reason
        assert "threshold=0.38" in reason
        # 0.37 <= 0.38, so should exist
        assert exists is True

    @patch("app.retrieval.adaptive_threshold.search")
    def test_proper_noun_detection(self, mock_search):
        """Test proper noun entity type detection."""
        mock_search.return_value = [
            {"distance": 0.39, "content": "Google Cloud Platform"},
        ]

        exists, distance, reason = calculate_context_aware_threshold("Google")

        assert distance == 0.39
        assert "Type=proper_noun" in reason
        assert "threshold=0.40" in reason
        # 0.39 <= 0.40, so should exist
        assert exists is True

    @patch("app.retrieval.adaptive_threshold.search")
    def test_credential_detection(self, mock_search):
        """Test credential entity type detection."""
        mock_search.return_value = [
            {"distance": 0.34, "content": "Doctor of Philosophy"},
        ]

        exists, distance, reason = calculate_context_aware_threshold("PhD degree")

        assert distance == 0.34
        assert "Type=credential" in reason
        assert "threshold=0.35" in reason
        assert exists is True

    @patch("app.retrieval.adaptive_threshold.search")
    def test_general_type_fallback(self, mock_search):
        """Test general entity type as fallback."""
        mock_search.return_value = [
            {"distance": 0.36, "content": "machine learning"},
        ]

        exists, distance, reason = calculate_context_aware_threshold("machine learning")

        assert distance == 0.36
        assert "Type=general" in reason
        assert "threshold=0.37" in reason
        assert exists is True

    @patch("app.retrieval.adaptive_threshold.search")
    def test_explicit_entity_type(self, mock_search):
        """Test with explicitly provided entity type."""
        mock_search.return_value = [
            {"distance": 0.35, "content": "React framework"},
        ]

        exists, distance, reason = calculate_context_aware_threshold(
            "React", entity_type="technology"
        )

        assert distance == 0.35
        assert "Type=technology" in reason
        assert "threshold=0.36" in reason
        assert exists is True

    @patch("app.retrieval.adaptive_threshold.search")
    def test_no_results(self, mock_search):
        """Test when search returns no results."""
        mock_search.return_value = []

        exists, distance, reason = calculate_context_aware_threshold("Nothing")

        assert exists is False
        assert distance == 1.0
        assert "No results found" in reason


@pytest.mark.unit
class TestCheckEntityExistsAdaptive:
    """Tests for the main adaptive check function."""

    @patch("app.retrieval.adaptive_threshold.calculate_gap_based_threshold")
    def test_gap_based_method(self, mock_gap):
        """Test using gap_based method."""
        mock_gap.return_value = (True, 0.30, "Strong match")

        exists, distance = check_entity_exists_adaptive("Python", method="gap_based")

        assert exists is True
        assert distance == 0.30
        mock_gap.assert_called_once_with("Python")

    @patch("app.retrieval.adaptive_threshold.calculate_percentile_threshold")
    def test_percentile_method(self, mock_percentile):
        """Test using percentile method."""
        mock_percentile.return_value = (0.35, 0.50)  # best_distance, threshold

        exists, distance = check_entity_exists_adaptive("Docker", method="percentile")

        assert exists is True  # 0.35 <= 0.50
        assert distance == 0.35
        mock_percentile.assert_called_once_with("Docker")

    @patch("app.retrieval.adaptive_threshold.calculate_percentile_threshold")
    def test_percentile_method_does_not_exist(self, mock_percentile):
        """Test percentile method when entity doesn't exist."""
        mock_percentile.return_value = (0.60, 0.50)  # distance > threshold

        exists, distance = check_entity_exists_adaptive("Fake", method="percentile")

        assert exists is False  # 0.60 > 0.50
        assert distance == 0.60

    @patch("app.retrieval.adaptive_threshold.calculate_context_aware_threshold")
    def test_context_aware_method(self, mock_context):
        """Test using context_aware method."""
        mock_context.return_value = (True, 0.32, "Type=acronym")

        exists, distance = check_entity_exists_adaptive("AWS", method="context_aware")

        assert exists is True
        assert distance == 0.32
        mock_context.assert_called_once_with("AWS")

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            check_entity_exists_adaptive("Python", method="invalid_method")

    @patch("app.retrieval.adaptive_threshold.calculate_gap_based_threshold")
    def test_default_method_is_gap_based(self, mock_gap):
        """Test that default method is gap_based."""
        mock_gap.return_value = (True, 0.28, "Strong match")

        exists, distance = check_entity_exists_adaptive("Test")

        # Should call gap_based by default
        mock_gap.assert_called_once_with("Test")
