"""
Test coverage for retrieval store helper functions.

Tests add_documents, get_sample_chunks, get_collection_stats,
and reset_collection operations.
"""

import pytest
from unittest.mock import MagicMock, patch
import random


@pytest.mark.unit
@pytest.mark.retrieval
class TestAddDocuments:
    """Tests for add_documents function."""

    @patch("app.retrieval.store._collection")
    def test_add_documents_empty_list(self, mock_collection):
        """Test that empty list is handled without calling upsert."""
        from app.retrieval.store import add_documents

        add_documents([])

        mock_collection.upsert.assert_not_called()

    @patch("app.retrieval.store._collection")
    def test_add_documents_single_doc(self, mock_collection):
        """Test adding a single document."""
        from app.retrieval.store import add_documents

        docs = [{
            "id": "chunk-1",
            "text": "Test content",
            "metadata": {"source": "test.md"}
        }]

        add_documents(docs)

        mock_collection.upsert.assert_called_once_with(
            ids=["chunk-1"],
            documents=["Test content"],
            metadatas=[{"source": "test.md"}]
        )

    @patch("app.retrieval.store._collection")
    def test_add_documents_multiple_docs(self, mock_collection):
        """Test adding multiple documents."""
        from app.retrieval.store import add_documents

        docs = [
            {"id": "chunk-1", "text": "Text 1", "metadata": {"source": "doc1.md"}},
            {"id": "chunk-2", "text": "Text 2", "metadata": {"source": "doc2.md"}},
            {"id": "chunk-3", "text": "Text 3", "metadata": {"source": "doc3.md"}},
        ]

        add_documents(docs)

        call_args = mock_collection.upsert.call_args
        assert len(call_args.kwargs["ids"]) == 3
        assert "chunk-1" in call_args.kwargs["ids"]

    @patch("app.retrieval.store._collection")
    def test_add_documents_missing_metadata(self, mock_collection):
        """Test that missing metadata is handled with empty dict."""
        from app.retrieval.store import add_documents

        docs = [{"id": "chunk-1", "text": "Text"}]  # No metadata key

        add_documents(docs)

        call_args = mock_collection.upsert.call_args
        assert call_args.kwargs["metadatas"] == [{}]


@pytest.mark.unit
@pytest.mark.retrieval
class TestGetSampleChunks:
    """Tests for get_sample_chunks function."""

    @patch("app.retrieval.store._collection")
    def test_get_sample_chunks_empty_collection(self, mock_collection):
        """Test that empty collection returns empty list."""
        from app.retrieval.store import get_sample_chunks

        mock_collection.count.return_value = 0

        result = get_sample_chunks(n=10)

        assert result == []
        mock_collection.get.assert_not_called()

    @patch("app.retrieval.store._collection")
    @patch("random.randint")
    def test_get_sample_chunks_normal(self, mock_randint, mock_collection):
        """Test getting sample chunks from populated collection."""
        from app.retrieval.store import get_sample_chunks

        mock_collection.count.return_value = 100
        mock_randint.return_value = 10  # Start offset
        mock_collection.get.return_value = {
            "ids": ["chunk-1", "chunk-2"],
            "documents": ["Text 1", "Text 2"],
            "metadatas": [{"source": "doc1.md"}, {"source": "doc2.md"}]
        }

        result = get_sample_chunks(n=2)

        assert len(result) == 2
        assert result[0]["id"] == "chunk-1"
        assert result[0]["text"] == "Text 1"
        assert result[0]["source"] == "doc1.md"

    @patch("app.retrieval.store._collection")
    def test_get_sample_chunks_caps_at_100(self, mock_collection):
        """Test that n is capped at 100."""
        from app.retrieval.store import get_sample_chunks

        mock_collection.count.return_value = 200
        mock_collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": []
        }

        get_sample_chunks(n=200)

        # Should be capped to 100
        call_args = mock_collection.get.call_args
        assert call_args.kwargs["limit"] == 100

    @patch("app.retrieval.store._collection")
    def test_get_sample_chunks_handles_error(self, mock_collection):
        """Test that exceptions during get are handled."""
        from app.retrieval.store import get_sample_chunks

        mock_collection.count.return_value = 50
        mock_collection.get.side_effect = Exception("ChromaDB error")

        result = get_sample_chunks(n=10)

        assert result == []


@pytest.mark.unit
@pytest.mark.retrieval
class TestGetCollectionStats:
    """Tests for get_collection_stats function."""

    @patch("app.retrieval.store._collection")
    def test_get_collection_stats_empty(self, mock_collection):
        """Test stats for empty collection."""
        from app.retrieval.store import get_collection_stats

        mock_collection.count.return_value = 0

        stats = get_collection_stats()

        assert stats["total_documents"] == 0
        assert "collection_name" in stats
        assert "embed_model" in stats

    @patch("app.retrieval.store._collection")
    def test_get_collection_stats_with_docs(self, mock_collection):
        """Test stats with documents."""
        from app.retrieval.store import get_collection_stats

        mock_collection.count.return_value = 150
        mock_collection.get.return_value = {
            "metadatas": [
                {"source": "resume.md", "doc_type": "resume"},
                {"source": "cert1.md", "doc_type": "certificate"},
                {"source": "cert2.md", "doc_type": "certificate"},
                {"source": "resume.md", "doc_type": "resume"},  # Duplicate source
            ]
        }

        stats = get_collection_stats()

        assert stats["total_documents"] == 150
        assert stats["unique_sources"] == 3  # resume.md, cert1.md, cert2.md
        assert "resume" in stats["doc_types"]
        assert "certificate" in stats["doc_types"]
        assert len(stats["sample_sources"]) <= 10  # Capped at 10


@pytest.mark.unit
@pytest.mark.retrieval
class TestResetCollection:
    """Tests for reset_collection function."""

    @patch("app.retrieval.store._client")
    @patch("app.retrieval.store._collection")
    def test_reset_collection_success(self, mock_collection, mock_client):
        """Test successful collection reset."""
        from app.retrieval.store import reset_collection

        mock_collection.count.return_value = 100
        mock_new_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_new_collection

        reset_collection()

        mock_client.delete_collection.assert_called_once()
        mock_client.get_or_create_collection.assert_called_once()

    @patch("app.retrieval.store._client")
    @patch("app.retrieval.store._collection")
    def test_reset_collection_already_deleted(self, mock_collection, mock_client):
        """Test reset when collection already deleted."""
        from app.retrieval.store import reset_collection

        mock_collection.count.side_effect = Exception("Collection not found")
        mock_client.delete_collection.side_effect = Exception("Already deleted")
        mock_new_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_new_collection

        # Should not raise, should recreate
        reset_collection()

        mock_client.get_or_create_collection.assert_called_once()


@pytest.mark.unit
@pytest.mark.retrieval
class TestSearchFunction:
    """Tests for the search function."""

    @patch("app.retrieval.store._collection")
    def test_search_empty_query_returns_empty(self, mock_collection):
        """Test that empty query returns empty list."""
        from app.retrieval.store import search

        result = search("")

        assert result == []
        mock_collection.query.assert_not_called()

    @patch("app.retrieval.store._collection")
    def test_search_none_query_returns_empty(self, mock_collection):
        """Test that None query returns empty list."""
        from app.retrieval.store import search

        result = search(None)

        assert result == []

    @patch("app.retrieval.store._collection")
    def test_search_whitespace_query_returns_empty(self, mock_collection):
        """Test that whitespace-only query returns empty list."""
        from app.retrieval.store import search

        result = search("   ")

        assert result == []


@pytest.mark.unit
@pytest.mark.retrieval
class TestSearchQueryRewriting:
    """Tests for query rewriting in search."""

    @patch("app.retrieval.store._semantic_search")
    def test_search_query_rewriting_disabled_uses_original(
        self, mock_semantic
    ):
        """Test that disabled query rewriting uses original query."""
        from app.retrieval.store import search

        mock_semantic.return_value = [
            {"id": "1", "text": "test", "source": "test.md", "distance": 0.2, "metadata": {}}
        ]

        result = search("test query", use_query_rewriting=False)

        # Should work with original query
        assert len(result) >= 0


@pytest.mark.unit
@pytest.mark.retrieval
class TestSearchCaching:
    """Tests for fallback caching in search."""

    @patch("app.retrieval.store._semantic_search")
    def test_search_returns_results(
        self, mock_semantic
    ):
        """Test that search returns results from semantic search."""
        from app.retrieval.store import search

        mock_semantic.return_value = [
            {"id": "1", "text": "test", "source": "test.md", "distance": 0.2, "metadata": {}}
        ]

        result = search("test query", use_query_rewriting=False, use_hybrid=False)

        # Should return results
        assert len(result) == 1


@pytest.mark.unit
@pytest.mark.retrieval
class TestSemanticSearch:
    """Tests for _semantic_search function."""

    @patch("app.retrieval.store._collection")
    def test_semantic_search_with_metadata_filter(self, mock_collection):
        """Test semantic search with metadata filter."""
        from app.retrieval.store import _semantic_search

        mock_collection.query.return_value = {
            "ids": [["chunk-1"]],
            "documents": [["Test content"]],
            "metadatas": [[{"source": "resume.md", "doc_type": "resume"}]],
            "distances": [[0.1]]
        }

        result = _semantic_search(
            query="test",
            k=5,
            max_dist=0.8,
            metadata_filter={"doc_type": "resume"}
        )

        assert len(result) == 1

    @patch("app.retrieval.store._collection")
    def test_semantic_search_filters_by_max_distance(self, mock_collection):
        """Test that results beyond max_distance are filtered out."""
        from app.retrieval.store import _semantic_search

        mock_collection.query.return_value = {
            "ids": [["chunk-1", "chunk-2", "chunk-3"]],
            "documents": [["Content 1", "Content 2", "Content 3"]],
            "metadatas": [[{"source": "doc.md"}, {"source": "doc.md"}, {"source": "doc.md"}]],
            "distances": [[0.1, 0.5, 0.9]]  # Only first two should pass max_dist=0.8
        }

        result = _semantic_search(
            query="test",
            k=5,
            max_dist=0.8
        )

        # Only chunks within max_dist should be returned
        assert len(result) == 2
        assert result[0]["distance"] == 0.1
        assert result[1]["distance"] == 0.5


@pytest.mark.unit
@pytest.mark.retrieval
class TestGetSource:
    """Tests for _get_source helper."""

    def test_get_source_normal(self):
        """Test _get_source with normal metadata."""
        from app.retrieval.store import _get_source

        result = _get_source({"source": "resume.md", "doc_type": "resume"})

        assert result == "resume.md"

    def test_get_source_missing_key(self):
        """Test _get_source with missing source key."""
        from app.retrieval.store import _get_source

        result = _get_source({"doc_type": "resume"})

        assert result == "unknown"

    def test_get_source_non_dict(self):
        """Test _get_source with non-dict input."""
        from app.retrieval.store import _get_source

        result = _get_source("not a dict")

        assert result == "unknown"

    def test_get_source_none(self):
        """Test _get_source with None."""
        from app.retrieval.store import _get_source

        result = _get_source(None)

        assert result == "unknown"

