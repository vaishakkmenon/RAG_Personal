"""
Tests for document ingestion pipeline.

Tests end-to-end ingestion pipeline: Loader -> Chunker -> VectorStore.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.ingest.pipeline import IngestionPipeline
from app.ingest.loader import Loader, IngestDocument
from app.ingest.chunker import Chunker


@pytest.mark.unit
class TestLoader:
    """Tests for Loader component."""

    @patch("app.ingest.loader.read_text")
    @patch("app.ingest.loader.extract_frontmatter")
    @patch("app.ingest.loader.extract_doc_id")
    @patch("app.ingest.loader.generate_version_identifier")
    @patch("os.path.getsize")
    def test_load_file_success(
        self, mock_size, mock_version, mock_doc_id, mock_frontmatter, mock_read
    ):
        """Test successful file loading."""
        mock_size.return_value = 100
        mock_read.return_value = "---\ndoc_type: resume\n---\n# Experience"
        mock_frontmatter.return_value = ({"doc_type": "resume"}, "# Experience")
        mock_doc_id.return_value = ("resume", "resume")
        mock_version.return_value = "v1"

        loader = Loader()
        doc = loader.load_file("/path/to/doc.md")

        assert doc is not None
        assert doc.text_content == "# Experience"
        assert doc.doc_id == "resume"
        assert doc.metadata["doc_type"] == "resume"

    @patch("os.path.getsize")
    def test_load_file_too_large(self, mock_size):
        """Test picking up large files returns None."""
        mock_size.return_value = 999999999
        loader = Loader()
        loader.max_file_size = 1000

        doc = loader.load_file("/path/to/large.md")
        assert doc is None


@pytest.mark.unit
class TestChunker:
    """Tests for Chunker component."""

    def test_chunker_routing(self):
        """Test that Chunker routes to correct strategy."""
        chunker = Chunker()

        strategy_default = chunker.get_strategy("resume")
        assert strategy_default.__class__.__name__ == "HeaderChunkingStrategy"

        strategy_transcript = chunker.get_strategy("transcript_analysis")
        assert strategy_transcript.__class__.__name__ == "TermChunkingStrategy"

    @patch("app.ingest.chunker.chunk_by_headers")
    def test_chunker_process(self, mock_chunk_func):
        """Test processing a document."""
        mock_chunk_func.return_value = [{"id": "1", "text": "foo"}]

        chunker = Chunker()
        doc = IngestDocument(
            source_path="/path/test.md",
            text_content="foo",
            metadata={"doc_type": "resume"},
            doc_id="test",
            doc_type="resume",
        )

        chunks = chunker.process(doc)
        assert len(chunks) == 1
        assert chunks[0]["id"] == "1"


@pytest.mark.unit
class TestIngestionPipeline:
    """Tests for the main pipeline."""

    @patch("app.ingest.pipeline.Loader")
    @patch("app.ingest.pipeline.Chunker")
    @patch("app.retrieval.vector_store.get_vector_store")
    def test_pipeline_run(self, mock_get_store, MockChunker, MockLoader):
        """Test full pipeline run."""
        # Setup mocks
        mock_loader = MockLoader.return_value
        mock_chunker = MockChunker.return_value
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        # Mock discovery
        mock_loader.discover.return_value = ["/docs/file1.md"]

        # Mock loading
        mock_doc = IngestDocument(
            source_path="/docs/file1.md",
            text_content="content",
            doc_id="file1",
            doc_type="resume",
        )
        mock_loader.load_file.return_value = mock_doc

        # Mock chunking
        mock_chunker.process.return_value = [
            {"id": "c1", "text": "content", "metadata": {}}
        ]

        # Run pipeline
        pipeline = IngestionPipeline(batch_size=10)
        stats = pipeline.run(["/docs"])

        assert stats.files_processed == 1
        assert stats.chunks_added == 1
        mock_store.add_documents.assert_called_once()

    @patch("app.ingest.pipeline.Loader")
    def test_pipeline_deduplication(self, MockLoader):
        """Test pipeline deduplication."""
        mock_loader = MockLoader.return_value
        mock_loader.discover.return_value = ["/docs/file1.md"]

        # Initialize pipeline without mocking internal methods
        # But we need to manually trigger dedup check
        pipeline = IngestionPipeline()

        chunk1 = {"id": "1", "text": "same", "metadata": {}}
        chunk2 = {"id": "2", "text": "same", "metadata": {}}

        assert pipeline._is_duplicate(chunk1) is False  # First time
        assert pipeline._is_duplicate(chunk2) is True  # Duplicate
