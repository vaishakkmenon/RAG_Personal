"""
Unit tests for Reranker Service.

Tests:
- Lexical overlap calculation
- Hybrid scoring logic
- Reranking order and edge cases
"""

import pytest
from app.services.reranker import RerankerService, rerank_chunks


@pytest.mark.unit
class TestRerankerService:
    """Tests for RerankerService class."""

    def test_lexical_overlap(self):
        """Test lexical overlap calculation."""
        # Exact match
        score = RerankerService.calculate_lexical_overlap("test query", "test query")
        assert score == 1.0

        # Partial match
        score = RerankerService.calculate_lexical_overlap("test query", "test document")
        assert score == 0.5  # "test" matches, "query" != "document"

        # No match
        score = RerankerService.calculate_lexical_overlap(
            "test query", "completely different"
        )
        assert score == 0.0

        # Empty query/text
        assert RerankerService.calculate_lexical_overlap("", "text") == 0.0
        assert RerankerService.calculate_lexical_overlap("query", "") == 0.0

        # Stopword removal (assuming "the" is a stopword)
        score = RerankerService.calculate_lexical_overlap("the query", "the document")
        # "the" is removed. "query" vs "document" -> 0.0
        assert score == 0.0

    def test_hybrid_score(self):
        """Test hybrid score calculation."""
        # Case 1: Pure Lexical (weight=1.0)
        chunk = {"text": "test document", "distance": 0.5}
        score = RerankerService.calculate_hybrid_score(
            "test query", chunk, lex_weight=1.0
        )
        # Overlap: "test" matches (0.5). Score = 1.0 * 0.5 + 0 = 0.5
        assert score == 0.5

        # Case 2: Pure Semantic (weight=0.0)
        chunk = {"text": "test document", "distance": 0.2}
        score = RerankerService.calculate_hybrid_score(
            "test query", chunk, lex_weight=0.0
        )
        # Similarity = 1 - 0.2 = 0.8. Score = 0 + 1.0 * 0.8 = 0.8
        assert score == 0.8

        # Case 3: Balanced (weight=0.5)
        # Overlap = 0.5 ("test" matches)
        # Similarity = 1 - 0.2 = 0.8
        # Score = 0.5*0.5 + 0.5*0.8 = 0.25 + 0.4 = 0.65
        score = RerankerService.calculate_hybrid_score(
            "test query", chunk, lex_weight=0.5
        )
        assert abs(score - 0.65) < 1e-6

        # Case 4: Missing distance
        chunk_no_dist = {"text": "test document"}  # distance defaults to 0.5 -> sim 0.5
        score = RerankerService.calculate_hybrid_score(
            "test query", chunk_no_dist, lex_weight=0.5
        )
        # Overlap = 0.5
        # Sim = 0.5
        # Score = 0.5*0.5 + 0.5*0.5 = 0.5
        assert score == 0.5

    def test_rerank_logic(self):
        """Test reranking order."""
        chunks = [
            {
                "id": "1",
                "text": "irrelevant text",
                "distance": 0.9,
            },  # Low overlap, Low sim
            {"id": "2", "text": "perfect match", "distance": 0.1},  # High sim
            {"id": "3", "text": "test query match", "distance": 0.5},  # High overlap
        ]

        # Test with balanced weight
        reranked = RerankerService.rerank("test query", chunks, lex_weight=0.5)

        # Chunk 3: Overlap=2/2=1.0. Sim=0.5. Score = 0.5*1 + 0.5*0.5 = 0.75
        # Chunk 2: Overlap=0.0. Sim=0.9. Score = 0.5*0 + 0.5*0.9 = 0.45
        # Chunk 1: Overlap=0.0. Sim=0.1. Score = 0.5*0 + 0.5*0.1 = 0.05

        assert reranked[0]["id"] == "3"
        assert reranked[1]["id"] == "2"
        assert reranked[2]["id"] == "1"

    def test_rerank_empty_list(self):
        """Test reranking empty list."""
        assert RerankerService.rerank("query", []) == []

    def test_convenience_function(self):
        """Test wrapper function."""
        chunks = [{"text": "test"}]
        result = rerank_chunks("test", chunks)
        assert len(result) == 1
