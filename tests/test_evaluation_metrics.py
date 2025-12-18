"""
Tests for evaluation metrics.

Tests all IR metrics: Recall@K, Precision@K, MRR, NDCG, F1.
Tests ID matching logic including prefix matching.
"""

import pytest


@pytest.mark.unit
class TestIDMatching:
    """Tests for ID matching logic."""

    def test_exact_id_match(self):
        """Test exact ID matching."""
        from app.evaluation.metrics import _is_match

        assert _is_match("chunk-1", "chunk-1") is True
        assert _is_match("chunk-1", "chunk-2") is False

    def test_prefix_id_match(self):
        """Test prefix matching for versioned chunks."""
        from app.evaluation.metrics import _is_match

        # Full chunk ID matches prefix
        assert _is_match("resume@v1#experience:0", "resume@v1#experience") is True
        assert _is_match("resume@v1#experience:5", "resume@v1#experience") is True

        # Different sections don't match
        assert _is_match("resume@v1#education:0", "resume@v1#experience") is False

    def test_count_matches(self):
        """Test counting matching IDs."""
        from app.evaluation.metrics import _count_matches

        retrieved = ["chunk-1", "chunk-2", "chunk-3"]
        relevant = ["chunk-1", "chunk-3", "chunk-5"]

        count = _count_matches(retrieved, relevant)
        assert count == 2  # chunk-1 and chunk-3


@pytest.mark.unit
class TestRecallAtK:
    """Tests for Recall@K metric."""

    def test_recall_at_k_perfect(self):
        """Test recall when all relevant docs are retrieved."""
        from app.evaluation.metrics import calculate_recall_at_k

        retrieved = ["chunk-1", "chunk-2", "chunk-3"]
        relevant = ["chunk-1", "chunk-2"]

        recall = calculate_recall_at_k(retrieved, relevant, k=3)
        assert recall == 1.0  # Found all relevant

    def test_recall_at_k_partial(self):
        """Test recall when some relevant docs are missing."""
        from app.evaluation.metrics import calculate_recall_at_k

        retrieved = ["chunk-1", "chunk-2", "chunk-3"]
        relevant = ["chunk-1", "chunk-4", "chunk-5"]

        recall = calculate_recall_at_k(retrieved, relevant, k=3)
        assert recall == pytest.approx(1 / 3)  # Found 1 out of 3 relevant

    def test_recall_at_k_no_relevant(self):
        """Test recall when no relevant docs exist."""
        from app.evaluation.metrics import calculate_recall_at_k

        retrieved = ["chunk-1", "chunk-2"]
        relevant = []

        recall = calculate_recall_at_k(retrieved, relevant, k=2)
        assert recall == 0.0


@pytest.mark.unit
class TestPrecisionAtK:
    """Tests for Precision@K metric."""

    def test_precision_at_k_perfect(self):
        """Test precision when all retrieved are relevant."""
        from app.evaluation.metrics import calculate_precision_at_k

        retrieved = ["chunk-1", "chunk-2"]
        relevant = ["chunk-1", "chunk-2", "chunk-3"]

        precision = calculate_precision_at_k(retrieved, relevant, k=2)
        assert precision == 1.0  # All retrieved are relevant

    def test_precision_at_k_partial(self):
        """Test precision when some retrieved are not relevant."""
        from app.evaluation.metrics import calculate_precision_at_k

        retrieved = ["chunk-1", "chunk-2", "chunk-3"]
        relevant = ["chunk-1"]

        precision = calculate_precision_at_k(retrieved, relevant, k=3)
        assert precision == pytest.approx(1 / 3)  # Only 1 out of 3 is relevant


@pytest.mark.unit
class TestMRR:
    """Tests for Mean Reciprocal Rank."""

    def test_mrr_first_position(self):
        """Test MRR when relevant doc is first."""
        from app.evaluation.metrics import calculate_mrr

        retrieved = ["relevant", "irrelevant", "irrelevant"]
        relevant = ["relevant"]

        mrr = calculate_mrr(retrieved, relevant)
        assert mrr == 1.0  # 1/1

    def test_mrr_third_position(self):
        """Test MRR when relevant doc is third."""
        from app.evaluation.metrics import calculate_mrr

        retrieved = ["a", "b", "relevant", "d"]
        relevant = ["relevant"]

        mrr = calculate_mrr(retrieved, relevant)
        assert mrr == pytest.approx(1 / 3)  # 1/3

    def test_mrr_no_match(self):
        """Test MRR when no relevant docs retrieved."""
        from app.evaluation.metrics import calculate_mrr

        retrieved = ["a", "b", "c"]
        relevant = ["d"]

        mrr = calculate_mrr(retrieved, relevant)
        assert mrr == 0.0


@pytest.mark.unit
class TestNDCG:
    """Tests for Normalized Discounted Cumulative Gain."""

    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        from app.evaluation.metrics import calculate_ndcg_at_k

        # All relevant docs at top
        retrieved = ["relevant-1", "relevant-2", "irrelevant"]
        relevant = ["relevant-1", "relevant-2"]

        ndcg = calculate_ndcg_at_k(retrieved, relevant, k=3)
        assert ndcg == 1.0  # Perfect ranking

    def test_ndcg_suboptimal_ranking(self):
        """Test NDCG with suboptimal ranking."""
        from app.evaluation.metrics import calculate_ndcg_at_k

        # Relevant docs not at top
        retrieved = ["irrelevant", "relevant-1", "relevant-2"]
        relevant = ["relevant-1", "relevant-2"]

        ndcg = calculate_ndcg_at_k(retrieved, relevant, k=3)
        assert 0.0 < ndcg < 1.0  # Less than perfect

    def test_ndcg_no_relevant(self):
        """Test NDCG when no relevant docs retrieved."""
        from app.evaluation.metrics import calculate_ndcg_at_k

        retrieved = ["a", "b", "c"]
        relevant = []

        ndcg = calculate_ndcg_at_k(retrieved, relevant, k=3)
        assert ndcg == 0.0


@pytest.mark.unit
class TestF1Score:
    """Tests for F1 score calculation."""

    def test_f1_score_balanced(self):
        """Test F1 with balanced precision and recall."""
        from app.evaluation.metrics import calculate_f1_score

        f1 = calculate_f1_score(precision=0.8, recall=0.8)
        assert f1 == pytest.approx(0.8)

    def test_f1_score_unbalanced(self):
        """Test F1 with unbalanced precision and recall."""
        from app.evaluation.metrics import calculate_f1_score

        f1 = calculate_f1_score(precision=1.0, recall=0.5)
        assert 0.6 < f1 < 0.7  # Harmonic mean

    def test_f1_score_zero_division(self):
        """Test F1 with zero precision and recall."""
        from app.evaluation.metrics import calculate_f1_score

        f1 = calculate_f1_score(precision=0.0, recall=0.0)
        assert f1 == 0.0


@pytest.mark.unit
class TestIDMatchingEdgeCases:
    """Tests for additional ID matching edge cases."""

    def test_is_match_prefix_starts_with(self):
        """Test prefix.startswith(expected_id) branch."""
        from app.evaluation.metrics import _is_match

        # retrieved_id has index, and is a longer form of expected_id
        # This tests line 37-38: prefix.startswith(expected_id)
        assert _is_match("doc@v1#section-subsection:0", "doc@v1#section") is True

    def test_is_match_expected_has_index(self):
        """Test when expected_id has an index suffix."""
        from app.evaluation.metrics import _is_match

        # Tests lines 41-48: expected_id has index suffix
        # Case: retrieved_id matches expected prefix
        assert _is_match("doc@v1#education", "doc@v1#education:0") is True

        # Case: both have indices but same prefix
        assert _is_match("doc@v1#education:1", "doc@v1#education:0") is True

    def test_is_match_both_have_different_prefixes(self):
        """Test when both have indices but different prefixes."""
        from app.evaluation.metrics import _is_match

        # Different prefixes should not match
        assert _is_match("doc@v1#edu:0", "doc@v1#exp:0") is False


@pytest.mark.unit
class TestPrecisionEdgeCases:
    """Tests for precision edge cases."""

    def test_precision_empty_retrieved(self):
        """Test precision when no docs retrieved."""
        from app.evaluation.metrics import calculate_precision_at_k

        # Tests line 132-133: empty retrieved list
        retrieved = []
        relevant = ["doc-1", "doc-2"]

        precision = calculate_precision_at_k(retrieved, relevant)
        assert precision == 0.0

    def test_precision_with_none_k(self):
        """Test precision with k=None (use all retrieved)."""
        from app.evaluation.metrics import calculate_precision_at_k

        retrieved = ["doc-1", "doc-2", "doc-3", "doc-4"]
        relevant = ["doc-1", "doc-2"]

        precision = calculate_precision_at_k(retrieved, relevant, k=None)
        assert precision == pytest.approx(0.5)  # 2 out of 4


@pytest.mark.unit
class TestFindFirstMatchRank:
    """Tests for first match rank finding."""

    def test_find_first_match_rank(self):
        """Test finding first match rank."""
        from app.evaluation.metrics import _find_first_match_rank

        retrieved = ["a", "b", "relevant", "d"]
        relevant = ["relevant"]

        rank = _find_first_match_rank(retrieved, relevant)
        assert rank == 3  # 1-indexed

    def test_find_first_match_rank_no_match(self):
        """Test no match returns 0."""
        from app.evaluation.metrics import _find_first_match_rank

        retrieved = ["a", "b", "c"]
        relevant = ["x", "y"]

        rank = _find_first_match_rank(retrieved, relevant)
        assert rank == 0


@pytest.mark.unit
class TestHasAnyMatch:
    """Tests for has_any_match helper."""

    def test_has_any_match_true(self):
        """Test has_any_match returns True when match exists."""
        from app.evaluation.metrics import _has_any_match

        retrieved = ["a", "b", "match"]
        relevant = ["match", "other"]

        assert _has_any_match(retrieved, relevant) is True

    def test_has_any_match_false(self):
        """Test has_any_match returns False when no match."""
        from app.evaluation.metrics import _has_any_match

        retrieved = ["a", "b", "c"]
        relevant = ["x", "y", "z"]

        assert _has_any_match(retrieved, relevant) is False
