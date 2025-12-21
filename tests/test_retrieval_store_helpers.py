"""
Test coverage for retrieval helper functions (VectorStore and SearchEngine).

Tests add_documents, get_sample_chunks, get_collection_stats,
reset_collection, and search operations via the retrieval facade.
"""

import pytest
from unittest.mock import MagicMock, patch
from app.retrieval import (
    add_documents,
    search,
)
from app.retrieval.vector_store import VectorStore


@pytest.mark.unit
@pytest.mark.retrieval
class TestAddDocumentsFacade:
    """Tests for add_documents facade function."""

    @patch("app.retrieval.get_vector_store")
    def test_add_documents_call(self, mock_get_store):
        """Verify facade calls VectorStore.add_documents."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        docs = [{"id": "1", "text": "test"}]
        add_documents(docs)

        mock_store.add_documents.assert_called_once_with(docs)


@pytest.mark.unit
@pytest.mark.retrieval
class TestVectorStoreOperations:
    """Tests for VectorStore specific logic."""

    def setup_method(self):
        # Mock dependencies to avoid real I/O during init
        with (
            patch("app.retrieval.vector_store.chromadb.PersistentClient"),
            patch("app.retrieval.vector_store.SentenceTransformerEmbeddingFunction"),
        ):
            self.store = VectorStore()
            self.store._collection = MagicMock()
            self.store._client = MagicMock()

    def test_add_documents_empty_list(self):
        self.store.add_documents([])
        self.store._collection.upsert.assert_not_called()

    def test_add_documents_single_doc(self):
        docs = [
            {"id": "chunk-1", "text": "Test content", "metadata": {"source": "test.md"}}
        ]
        self.store.add_documents(docs)
        self.store._collection.upsert.assert_called_once_with(
            ids=["chunk-1"],
            documents=["Test content"],
            metadatas=[{"source": "test.md"}],
        )

    def test_add_documents_missing_metadata(self):
        docs = [{"id": "chunk-1", "text": "Text"}]  # No metadata key
        self.store.add_documents(docs)
        call_args = self.store._collection.upsert.call_args
        assert call_args.kwargs["metadatas"] == [{}]

    def test_get_sample_chunks_empty_collection(self):
        self.store._collection.count.return_value = 0
        result = self.store.get_sample_chunks(n=10)
        assert result == []

    @patch("random.randint")
    def test_get_sample_chunks_normal(self, mock_randint):
        self.store._collection.count.return_value = 100
        mock_randint.return_value = 10
        self.store._collection.get.return_value = {
            "ids": ["chunk-1", "chunk-2"],
            "documents": ["Text 1", "Text 2"],
            "metadatas": [{"source": "doc1.md"}, {"source": "doc2.md"}],
        }

        result = self.store.get_sample_chunks(n=2)
        assert len(result) == 2
        assert result[0]["id"] == "chunk-1"

    def test_get_sample_chunks_caps_at_100(self):
        self.store._collection.count.return_value = 200
        self.store._collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }
        self.store.get_sample_chunks(n=200)
        call_args = self.store._collection.get.call_args
        assert call_args.kwargs["limit"] == 100

    def test_get_collection_stats_empty(self):
        self.store._collection.count.return_value = 0
        stats = self.store.get_stats()
        assert stats["total_documents"] == 0

    def test_get_collection_stats_with_docs(self):
        self.store._collection.count.return_value = 150
        self.store._collection.get.return_value = {
            "metadatas": [
                {"source": "resume.md", "doc_type": "resume"},
                {"source": "cert1.md", "doc_type": "certificate"},
            ]
        }

        stats = self.store.get_stats()
        assert stats["total_documents"] == 150
        assert stats["unique_sources"] == 2

    def test_reset_collection(self):
        self.store._collection.count.return_value = 100
        self.store._client.get_or_create_collection.return_value = MagicMock()

        self.store.reset()

        self.store._client.delete_collection.assert_called_once()
        self.store._client.get_or_create_collection.assert_called_once()

    def test_semantic_search_filters_by_max_distance(self):
        # Mock query return for search
        self.store._collection.query.return_value = {
            "ids": [["chunk-1", "chunk-2"]],
            "documents": [["Content 1", "Content 2"]],
            "metadatas": [[{"source": "doc.md"}, {"source": "doc.md"}]],
            "distances": [[0.1, 0.9]],  # 0.9 > max_distance 0.8
        }

        # We need to mock embedding generation/cache
        self.store._embed = MagicMock(return_value=[[0.1, 0.1, 0.1]])

        with patch("app.services.embedding_cache.get_embedding_cache") as mock_cache:
            mock_cache.return_value.get_embedding.return_value = None

            result = self.store.search(query="test", k=5, max_distance=0.8)

            # Only chunk-1 should be returned
            assert len(result) == 1
            assert result[0]["distance"] == 0.1
            assert result[0]["id"] == "chunk-1"

    def test_semantic_search_with_metadata_filter(self):
        """Test that metadata filters are correctly passed to ChromaDB."""
        self.store._collection.query.return_value = {
            "ids": [["chunk-1"]],
            "documents": [["Content"]],
            "metadatas": [[{"doc_type": "active"}]],
            "distances": [[0.1]],
        }
        self.store._embed = MagicMock(return_value=[[0.1]])

        with patch("app.services.embedding_cache.get_embedding_cache") as mock_cache:
            mock_cache.return_value.get_embedding.return_value = None

            self.store.search(
                query="test",
                k=5,
                max_distance=1.0,
                metadata_filter={"doc_type": "active"},
            )

            # Verify _collection.query called with where clause
            call_args = self.store._collection.query.call_args
            assert call_args.kwargs["where"] == {"doc_type": {"$eq": "active"}}

    def test_get_source_helper(self):
        # Test the private helper _get_source
        assert self.store._get_source({"source": "test.md"}) == "test.md"
        assert self.store._get_source({}) == "unknown"
        assert self.store._get_source(None) == "unknown"


@pytest.mark.unit
@pytest.mark.retrieval
class TestSearchFacade:
    """Tests for search facade and orchestrator logic (SearchEngine)."""

    @patch("app.retrieval.get_search_engine")
    def test_search_empty_query_returns_empty(self, mock_get_engine):
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_engine.search.return_value = []

        search("")
        mock_engine.search.assert_called_once()


@pytest.mark.unit
@pytest.mark.retrieval
class TestSearchEngineLogic:
    """Tests for SearchEngine class logic."""

    def setup_method(self):
        # Import inside setup to avoid top-level side effects during collection
        from app.retrieval.search_engine import SearchEngine

        # Mock get_vector_store to avoid real init
        # Also need to mock BM25 check logic if it tries to open files
        with (
            patch(
                "app.retrieval.search_engine.get_vector_store", return_value=MagicMock()
            ),
            patch("app.retrieval.search_engine.BM25Index"),
            patch("pathlib.Path.exists", return_value=False),
        ):  # Force BM25 disabled to avoid load
            self.engine = SearchEngine()
            self.engine.vector_store = MagicMock()
            self.engine.bm25_index = None

    def test_search_empty_query_returns_empty(self):
        result = self.engine.search("")
        assert result == []
        self.engine.vector_store.search.assert_not_called()

    def test_search_none_query_returns_empty(self):
        result = self.engine.search(None)
        assert result == []

    def test_search_whitespace_query_returns_empty(self):
        result = self.engine.search("   ")
        assert result == []

    def test_search_query_rewriting_disabled(self):
        """Test that disabled query rewriting uses original query."""
        self.engine.vector_store.search.return_value = [{"id": "1", "text": "test"}]

        # Call with use_query_rewriting=False
        self.engine.search(
            "test query",
            use_query_rewriting=False,
            use_hybrid=False,
            use_cross_encoder=False,
        )

        # Should call search with original query
        # Arg 1 is query
        args, _ = self.engine.vector_store.search.call_args
        assert args[0] == "test query"

    def test_search_returns_results(self):
        """Test that search returns results from vector store."""
        expected = [
            {
                "id": "1",
                "text": "test",
                "distance": 0.1,
                "source": "test.md",
                "metadata": {},
            }
        ]
        self.engine.vector_store.search.return_value = expected

        result = self.engine.search("test", use_hybrid=False, use_cross_encoder=False)

        assert len(result) == 1
        assert result[0]["id"] == "1"
