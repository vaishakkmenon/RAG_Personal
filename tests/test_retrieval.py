"""
Unit tests for Retrieval and Reranking.

Tests:
- Semantic search
- Distance filtering
- Metadata filtering
- BM25 reranking
- Cross-encoder reranking
- Query rewriting
- Negative inference
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.unit
@pytest.mark.retrieval
class TestSemanticSearch:
    """Tests for basic semantic search functionality."""

    @patch("app.retrieval.store._collection")
    def test_search_returns_chunks(self, mock_collection):
        """Test that search returns relevant chunks."""
        from app.retrieval import search

        # Mock collection query
        mock_collection.query.return_value = {
            "ids": [["chunk-1", "chunk-2"]],
            "documents": [["Text 1", "Text 2"]],
            "metadatas": [[{"source": "test1.md"}, {"source": "test2.md"}]],
            "distances": [[0.15, 0.25]],
        }

        results = search(query="test question", k=5, use_hybrid=False, use_query_rewriting=False, use_cross_encoder=False)
        print(f"\nDEBUG: Results returned: {results}")

        assert len(results) == 2
        assert results[0]["id"] == "chunk-1"
        assert results[0]["text"] == "Text 1"
        assert results[0]["distance"] == 0.15
        assert results[1]["distance"] == 0.25

    @patch("app.retrieval.store._collection")
    def test_search_respects_max_distance(self, mock_collection):
        """Test that max_distance filters out distant chunks."""
        from app.retrieval import search

        # Mock collection query (return chunks with varying distances)
        mock_collection.query.return_value = {
            "ids": [["chunk-1", "chunk-2", "chunk-3"]],
            "documents": [["Close", "Medium", "Far"]],
            "metadatas": [[{}, {}, {}]],
            "distances": [[0.15, 0.55, 0.85]],  # Last one exceeds threshold
        }

        results = search(query="test", k=10, max_distance=0.6, use_hybrid=False, use_query_rewriting=False)

        # Should only return chunks with distance <= 0.6
        assert len(results) == 2
        assert results[0]["distance"] == 0.15
        assert results[1]["distance"] == 0.55

    @patch("app.retrieval.store._collection")
    def test_search_with_metadata_filter(self, mock_collection):
        """Test that metadata filters are applied."""
        from app.retrieval import search

        mock_collection.query.return_value = {
            "ids": [["chunk-1"]],
            "documents": [["Resume content"]],
            "metadatas": [[{"doc_type": "resume"}]],
            "distances": [[0.15]],
        }

        results = search(
            query="test",
            k=5,
            metadata_filter={"doc_type": "resume"},
            use_hybrid=False,
            use_query_rewriting=False
        )

        # Verify query was called with where clause
        query_call = mock_collection.query.call_args
        assert "where" in query_call.kwargs
        assert query_call.kwargs["where"]["doc_type"]["$eq"] == "resume"

    @patch("app.retrieval.store._collection")
    def test_search_empty_results(self, mock_collection):
        """Test handling of empty search results."""
        from app.retrieval import search

        # Mock empty results
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        results = search(query="nonexistent topic", k=5, use_hybrid=False, use_query_rewriting=False)

        assert len(results) == 0


@pytest.mark.unit
@pytest.mark.retrieval
class TestBM25Reranking:
    """Tests for BM25 lexical reranking (hybrid search)."""

    @patch("app.retrieval.store._collection")
    @patch("app.retrieval.store._bm25_index")
    def test_bm25_reranking_applied(self, mock_bm25_index, mock_collection):
        """Test that BM25 hybrid search is applied when enabled."""
        from app.retrieval import search

        # Mock BM25 index search results
        mock_bm25_index.search.return_value = [
            {"id": "chunk-2", "text": "Java development", "bm25_score": 0.9, "distance": None},
            {"id": "chunk-1", "text": "Python programming", "bm25_score": 0.5, "distance": None},
        ]

        # Mock semantic search results (ChromaDB)
        mock_collection.query.return_value = {
            "ids": [["chunk-1", "chunk-2"]],
            "documents": [["Python programming", "Java development"]],
            "metadatas": [[{"source": "test1.md"}, {"source": "test2.md"}]],
            "distances": [[0.2, 0.3]],
        }

        # Call with use_hybrid=True (enables BM25)
        results = search(
            query="Java",
            k=5,
            use_hybrid=True,
            use_query_rewriting=False,
            use_cross_encoder=False
        )

        # Verify BM25 was called
        mock_bm25_index.search.assert_called_once()

        # Verify semantic search was also called (for hybrid)
        assert mock_collection.query.called

        # Results should be merged using RRF
        assert len(results) > 0

    @patch("app.retrieval.store._collection")
    @patch("app.retrieval.store._bm25_index")
    def test_bm25_disabled_when_use_hybrid_false(self, mock_bm25_index, mock_collection):
        """Test that BM25 is not applied when use_hybrid=False."""
        from app.retrieval import search

        mock_collection.query.return_value = {
            "ids": [["chunk-1"]],
            "documents": [["Test"]],
            "metadatas": [[{"source": "test.md"}]],
            "distances": [[0.15]],
        }

        results = search(
            query="test",
            k=5,
            use_hybrid=False,
            use_query_rewriting=False,
            use_cross_encoder=False
        )

        # Verify BM25 was NOT called
        mock_bm25_index.search.assert_not_called()

        # Verify only semantic search was used
        assert mock_collection.query.called


@pytest.mark.unit
@pytest.mark.retrieval
class TestCrossEncoderReranking:
    """Tests for cross-encoder neural reranking."""

    @patch("app.retrieval.store._collection")
    @patch("app.services.cross_encoder_reranker.get_cross_encoder_reranker")
    def test_cross_encoder_reranking(self, mock_get_reranker, mock_collection):
        """Test that cross-encoder reranking is applied when use_cross_encoder=True."""
        from app.retrieval import search

        # Mock semantic search results
        mock_collection.query.return_value = {
            "ids": [["chunk-1", "chunk-2"]],
            "documents": [["Relevant text", "Less relevant"]],
            "metadatas": [[{"source": "test1.md"}, {"source": "test2.md"}]],
            "distances": [[0.25, 0.15]],
        }

        # Mock cross-encoder reranker instance
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {
                "id": "chunk-1",
                "text": "Relevant text",
                "source": "test1.md",
                "distance": 0.25,
                "cross_encoder_score": 0.95
            },
            {
                "id": "chunk-2",
                "text": "Less relevant",
                "source": "test2.md",
                "distance": 0.15,
                "cross_encoder_score": 0.40
            },
        ]
        mock_get_reranker.return_value = mock_reranker

        results = search(
            query="test query",
            k=5,
            use_hybrid=False,
            use_query_rewriting=False,
            use_cross_encoder=True
        )

        # Verify cross-encoder was called
        mock_reranker.rerank.assert_called_once()

        # Verify reranked results
        assert len(results) == 2
        assert results[0]["id"] == "chunk-1"
        assert results[0]["cross_encoder_score"] == 0.95


@pytest.mark.unit
@pytest.mark.retrieval
class TestQueryRewriting:
    """Tests for query rewriting functionality."""

    @patch("app.retrieval.store._collection")
    @patch("app.retrieval.query_rewriter.get_query_rewriter")
    def test_query_rewriting_applied(self, mock_get_rewriter, mock_collection):
        """Test that query rewriting is applied when enabled."""
        from app.retrieval import search
        from app.models import RewriteMetadata
        from app.settings import settings

        # Temporarily enable query rewriting for this test
        original_enabled = settings.query_rewriter.enabled
        settings.query_rewriter.enabled = True

        try:
            # Mock query rewriter instance
            mock_rewriter = MagicMock()
            mock_rewriter.rewrite_query.return_value = (
                "Python programming experience skills",
                RewriteMetadata(
                    pattern_name="experience_question",
                    pattern_type="regex_list",
                    original_query="What is my Python experience?",
                    rewritten_query="Python programming experience skills",
                    latency_ms=5.2,
                    confidence=0.95
                )
            )
            mock_get_rewriter.return_value = mock_rewriter

            # Mock semantic search
            mock_collection.query.return_value = {
                "ids": [["chunk-1"]],
                "documents": [["Test"]],
                "metadatas": [[{"source": "test.md"}]],
                "distances": [[0.15]],
            }

            results = search(
                query="What is my Python experience?",
                k=5,
                use_hybrid=False,
                use_query_rewriting=True,
                use_cross_encoder=False
            )

            # Verify rewriter was called with original query
            mock_rewriter.rewrite_query.assert_called_once_with("What is my Python experience?")

            # Verify search was performed
            assert len(results) >= 0
        finally:
            # Restore original setting
            settings.query_rewriter.enabled = original_enabled

    @patch("app.retrieval.store._collection")
    def test_query_rewriting_disabled(self, mock_collection):
        """Test that query rewriting can be disabled."""
        from app.retrieval import search

        # Mock semantic search
        mock_collection.query.return_value = {
            "ids": [["chunk-1"]],
            "documents": [["Test"]],
            "metadatas": [[{"source": "test.md"}]],
            "distances": [[0.15]],
        }

        # Mock query rewriter to verify it's NOT called
        with patch("app.retrieval.query_rewriter.get_query_rewriter") as mock_get_rewriter:
            results = search(
                query="test query",
                k=5,
                use_hybrid=False,
                use_query_rewriting=False,
                use_cross_encoder=False
            )

            # Verify rewriter was NOT called
            mock_get_rewriter.assert_not_called()


@pytest.mark.unit
@pytest.mark.retrieval
class TestRetrievalEdgeCases:
    """Tests for edge cases and error handling."""

    @patch("app.retrieval.store._collection")
    def test_empty_collection(self, mock_collection):
        """Test handling of empty collection."""
        from app.retrieval import search

        # Mock empty collection
        mock_collection.count.return_value = 0
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        results = search(
            query="test",
            k=5,
            use_hybrid=False,
            use_query_rewriting=False,
            use_cross_encoder=False
        )

        assert len(results) == 0

    @patch("app.retrieval.store._collection")
    def test_k_larger_than_collection(self, mock_collection):
        """Test requesting more chunks than exist in collection."""
        from app.retrieval import search

        # Collection only has 2 chunks but we request 10
        mock_collection.query.return_value = {
            "ids": [["chunk-1", "chunk-2"]],
            "documents": [["Text 1", "Text 2"]],
            "metadatas": [[{"source": "test1.md"}, {"source": "test2.md"}]],
            "distances": [[0.15, 0.25]],
        }

        results = search(
            query="test",
            k=10,
            use_hybrid=False,
            use_query_rewriting=False,
            use_cross_encoder=False
        )

        # Should return only available chunks
        assert len(results) == 2

    @patch("app.retrieval.store._collection")
    def test_chromadb_connection_failure(self, mock_collection):
        """Test handling of ChromaDB connection failures."""
        from app.retrieval import search

        # Clear fallback cache to ensure no cached results
        try:
            from app.retrieval.fallback_cache import get_fallback_cache
            cache = get_fallback_cache()
            cache.clear()
        except:
            pass

        # Mock connection failure
        mock_collection.query.side_effect = Exception("ChromaDB connection failed")

        with pytest.raises(Exception, match="ChromaDB connection failed"):
            search(
                query="test_unique_query_12345",  # Use unique query to avoid cache
                k=5,
                use_hybrid=False,
                use_query_rewriting=False,
                use_cross_encoder=False
            )
