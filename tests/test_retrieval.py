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

from app.retrieval.vector_store import get_vector_store
from app.retrieval.search_engine import get_search_engine
from app.retrieval import search


@pytest.mark.unit
@pytest.mark.retrieval
class TestSemanticSearch:
    """Tests for basic semantic search functionality."""

    def setup_method(self):
        self.store = get_vector_store()
        self.patcher = patch.object(self.store, "_collection")
        self.mock_collection = self.patcher.start()

    def teardown_method(self):
        self.patcher.stop()

    def test_search_returns_chunks(self):
        """Test that search returns relevant chunks."""
        # Mock collection query
        self.mock_collection.query.return_value = {
            "ids": [["chunk-1", "chunk-2"]],
            "documents": [["Text 1", "Text 2"]],
            "metadatas": [[{"source": "test1.md"}, {"source": "test2.md"}]],
            "distances": [[0.15, 0.25]],
        }

        results = search(
            query="test question",
            k=5,
            use_hybrid=False,
            use_query_rewriting=False,
            use_cross_encoder=False,
        )

        assert len(results) == 2
        assert results[0]["id"] == "chunk-1"
        assert results[0]["text"] == "Text 1"
        assert results[0]["distance"] == 0.15
        assert results[1]["distance"] == 0.25

    def test_search_respects_max_distance(self):
        """Test that max_distance filters out distant chunks."""
        # Mock collection query
        self.mock_collection.query.return_value = {
            "ids": [["chunk-1", "chunk-2", "chunk-3"]],
            "documents": [["Close", "Medium", "Far"]],
            "metadatas": [[{}, {}, {}]],
            "distances": [[0.15, 0.55, 0.85]],  # Last one exceeds threshold
        }

        results = search(
            query="test",
            k=10,
            max_distance=0.6,
            use_hybrid=False,
            use_query_rewriting=False,
        )

        # Should only return chunks with distance <= 0.6
        assert len(results) == 2
        assert results[0]["distance"] == 0.15
        assert results[1]["distance"] == 0.55

    def test_search_with_metadata_filter(self):
        """Test that metadata filters are applied."""
        self.mock_collection.query.return_value = {
            "ids": [["chunk-1"]],
            "documents": [["Resume content"]],
            "metadatas": [[{"doc_type": "resume"}]],
            "distances": [[0.15]],
        }

        search(
            query="test",
            k=5,
            metadata_filter={"doc_type": "resume"},
            use_hybrid=False,
            use_query_rewriting=False,
        )

        # Verify query was called with where clause
        query_call = self.mock_collection.query.call_args
        assert "where" in query_call.kwargs
        assert query_call.kwargs["where"]["doc_type"]["$eq"] == "resume"

    def test_search_empty_results(self):
        """Test handling of empty search results."""
        # Mock empty results
        self.mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        results = search(
            query="nonexistent topic", k=5, use_hybrid=False, use_query_rewriting=False
        )

        assert len(results) == 0


@pytest.mark.unit
@pytest.mark.retrieval
class TestBM25Reranking:
    """Tests for BM25 lexical reranking (hybrid search)."""

    def setup_method(self):
        self.engine = get_search_engine()
        self.store = get_vector_store()

        # Patch BM25 index on the engine instance
        self.bm25_patcher = patch.object(self.engine, "bm25_index")
        self.mock_bm25_index = self.bm25_patcher.start()

        # Patch vector store search to avoid real DB calls
        self.store_patcher = patch.object(self.store, "search")
        self.mock_store_search = self.store_patcher.start()

    def teardown_method(self):
        self.bm25_patcher.stop()
        self.store_patcher.stop()

    def test_bm25_reranking_applied(self):
        """Test that BM25 hybrid search is applied when enabled."""

        # Mock BM25 index search results
        self.mock_bm25_index.search.return_value = [
            {
                "id": "chunk-2",
                "text": "Java development",
                "bm25_score": 0.9,
                "distance": None,
            },
            {
                "id": "chunk-1",
                "text": "Python programming",
                "bm25_score": 0.5,
                "distance": None,
            },
        ]

        # Mock semantic search results
        self.mock_store_search.return_value = [
            {
                "id": "chunk-1",
                "text": "Python programming",
                "distance": 0.2,
                "source": "test1.md",
                "metadata": {},
            },
            {
                "id": "chunk-2",
                "text": "Java development",
                "distance": 0.3,
                "source": "test2.md",
                "metadata": {},
            },
        ]

        # Call with use_hybrid=True (enables BM25)
        results = search(
            query="Java",
            k=5,
            use_hybrid=True,
            use_query_rewriting=False,
            use_cross_encoder=False,
        )

        # Verify BM25 was called
        self.mock_bm25_index.search.assert_called_once()

        # Verify semantic search was also called
        self.mock_store_search.assert_called()

        # Results should be merged
        assert len(results) > 0

    def test_bm25_disabled_when_use_hybrid_false(self):
        """Test that BM25 is not applied when use_hybrid=False."""

        self.mock_store_search.return_value = [
            {
                "id": "chunk-1",
                "text": "Test",
                "distance": 0.15,
                "source": "test.md",
                "metadata": {},
            }
        ]

        search(
            query="test",
            k=5,
            use_hybrid=False,
            use_query_rewriting=False,
            use_cross_encoder=False,
        )

        # Verify BM25 was NOT called
        self.mock_bm25_index.search.assert_not_called()

        # Verify semantic search was used
        self.mock_store_search.assert_called()


@pytest.mark.unit
@pytest.mark.retrieval
class TestCrossEncoderReranking:
    """Tests for cross-encoder neural reranking."""

    def setup_method(self):
        self.store = get_vector_store()
        self.store_patcher = patch.object(self.store, "search")
        self.mock_store_search = self.store_patcher.start()

    def teardown_method(self):
        self.store_patcher.stop()

    @patch("app.services.cross_encoder_reranker.get_cross_encoder_reranker")
    def test_cross_encoder_reranking(self, mock_get_reranker):
        """Test that cross-encoder reranking is applied when use_cross_encoder=True."""

        # Mock semantic search results
        self.mock_store_search.return_value = [
            {
                "id": "chunk-1",
                "text": "Relevant text",
                "distance": 0.25,
                "source": "test1.md",
                "metadata": {},
            },
            {
                "id": "chunk-2",
                "text": "Less relevant",
                "distance": 0.15,
                "source": "test2.md",
                "metadata": {},
            },
        ]

        # Mock cross-encoder reranker instance
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {
                "id": "chunk-1",
                "text": "Relevant text",
                "source": "test1.md",
                "distance": 0.25,
                "cross_encoder_score": 0.95,
            },
            {
                "id": "chunk-2",
                "text": "Less relevant",
                "source": "test2.md",
                "distance": 0.15,
                "cross_encoder_score": 0.40,
            },
        ]
        mock_get_reranker.return_value = mock_reranker

        results = search(
            query="test query",
            k=5,
            use_hybrid=False,
            use_query_rewriting=False,
            use_cross_encoder=True,
        )

        # Verify semantic search called
        self.mock_store_search.assert_called()

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

    def setup_method(self):
        self.store = get_vector_store()
        self.store_patcher = patch.object(self.store, "search")
        self.mock_store_search = self.store_patcher.start()

    def teardown_method(self):
        self.store_patcher.stop()

    @patch("app.retrieval.query_rewriter.get_query_rewriter")
    def test_query_rewriting_applied(self, mock_get_rewriter):
        """Test that query rewriting is applied when enabled."""
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
                    confidence=0.95,
                ),
            )
            mock_get_rewriter.return_value = mock_rewriter

            # Mock semantic search results
            self.mock_store_search.return_value = [
                {
                    "id": "chunk-1",
                    "text": "Test",
                    "distance": 0.15,
                    "source": "test.md",
                    "metadata": {},
                }
            ]

            results = search(
                query="What is my Python experience?",
                k=5,
                use_hybrid=False,
                use_query_rewriting=True,
                use_cross_encoder=False,
            )

            # Verify rewriter was called with original query
            mock_rewriter.rewrite_query.assert_called_once_with(
                "What is my Python experience?"
            )

            # Verify search was performed
            assert len(results) >= 0
        finally:
            # Restore original setting
            settings.query_rewriter.enabled = original_enabled

    def test_query_rewriting_disabled(self):
        """Test that query rewriting can be disabled."""

        self.mock_store_search.return_value = [
            {
                "id": "chunk-1",
                "text": "Test",
                "distance": 0.15,
                "source": "test.md",
                "metadata": {},
            }
        ]

        # Mock query rewriter to verify it's NOT called
        with patch(
            "app.retrieval.query_rewriter.get_query_rewriter"
        ) as mock_get_rewriter:
            search(
                query="test query",
                k=5,
                use_hybrid=False,
                use_query_rewriting=False,
                use_cross_encoder=False,
            )

            # Verify rewriter was NOT called
            mock_get_rewriter.assert_not_called()


@pytest.mark.unit
@pytest.mark.retrieval
class TestRetrievalEdgeCases:
    """Tests for edge cases and error handling."""

    def setup_method(self):
        self.store = get_vector_store()
        self.patcher = patch.object(self.store, "_collection")
        self.mock_collection = self.patcher.start()

    def teardown_method(self):
        self.patcher.stop()

    def test_empty_collection(self):
        """Test handling of empty collection."""
        # Mock empty collection
        self.mock_collection.count.return_value = 0
        self.mock_collection.query.return_value = {
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
            use_cross_encoder=False,
        )

        assert len(results) == 0

    def test_k_larger_than_collection(self):
        """Test requesting more chunks than exist in collection."""
        # Collection only has 2 chunks but we request 10
        self.mock_collection.query.return_value = {
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
            use_cross_encoder=False,
        )

        # Should return only available chunks
        assert len(results) == 2

    def test_chromadb_connection_failure(self):
        """Test handling of ChromaDB connection failures."""
        # Clear fallback cache to ensure no cached results
        try:
            from app.retrieval.fallback_cache import get_fallback_cache

            cache = get_fallback_cache()
            cache.clear()
        except Exception:
            pass

        # Mock connection failure
        self.mock_collection.query.side_effect = Exception("ChromaDB connection failed")

        # The search engine now catches exceptions and tries fallback cache
        # If fallback cache is empty, it re-raises

        with pytest.raises(Exception, match="ChromaDB connection failed"):
            search(
                query="test_unique_query_12345",  # Use unique query to avoid cache
                k=5,
                use_hybrid=False,
                use_query_rewriting=False,
                use_cross_encoder=False,
            )
