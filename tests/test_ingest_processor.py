"""
Tests for document ingestion processor.

Tests end-to-end ingestion pipeline: file discovery → metadata extraction →
chunking → deduplication → ChromaDB storage.
"""

import pytest
from unittest.mock import patch
import tempfile
import os


@pytest.mark.unit
class TestChunkIdGeneration:
    """Tests for chunk ID generation."""

    def test_generate_chunk_id_format(self):
        """Test that chunk IDs follow expected format."""
        from app.ingest.processor import _generate_chunk_id

        chunk_id = _generate_chunk_id(
            doc_id="resume",
            version="2025-01-15",
            section_slug="experience",
            chunk_idx=0,
        )

        assert chunk_id == "resume@2025-01-15#experience:0"

    def test_generate_chunk_id_with_multipart(self):
        """Test chunk ID with higher index."""
        from app.ingest.processor import _generate_chunk_id

        chunk_id = _generate_chunk_id(
            doc_id="certificate-cka",
            version="2024-06-26",
            section_slug="skills",
            chunk_idx=3,
        )

        assert chunk_id == "certificate-cka@2024-06-26#skills:3"


@pytest.mark.unit
class TestProcessFile:
    """Tests for single file processing."""

    @patch("app.ingest.processor.read_text")
    @patch("app.ingest.processor.extract_frontmatter")
    @patch("app.ingest.processor.smart_chunk")
    @patch("app.ingest.processor.extract_doc_id")
    @patch("app.ingest.processor.generate_version_identifier")
    def test_process_file_success(
        self, mock_version, mock_doc_id, mock_smart_chunk, mock_frontmatter, mock_read
    ):
        """Test successful file processing."""
        from app.ingest.processor import _process_file

        # Set up mocks
        mock_read.return_value = (
            "---\ndoc_type: resume\n---\n# Experience\nTest content"
        )
        mock_frontmatter.return_value = (
            {"doc_type": "resume"},
            "# Experience\nTest content",
        )
        mock_doc_id.return_value = ("resume", "resume")
        mock_version.return_value = "2025-01-15"
        mock_smart_chunk.return_value = [
            {
                "id": "resume@2025-01-15#experience:0",
                "text": "Test content",
                "metadata": {},
            }
        ]

        # Create temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            chunks = _process_file(temp_path)
            assert len(chunks) == 1
            assert chunks[0]["id"] == "resume@2025-01-15#experience:0"
        finally:
            os.unlink(temp_path)

    def test_process_file_too_large(self):
        """Test that oversized files are skipped."""
        from app.ingest.processor import _process_file, MAX_FILE_SIZE

        # Create a temp file larger than MAX_FILE_SIZE
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".md", delete=False) as f:
            # Write more than max size
            f.write(b"x" * (MAX_FILE_SIZE + 1000))
            temp_path = f.name

        try:
            chunks = _process_file(temp_path)
            assert chunks == []  # Should return empty for oversized files
        finally:
            os.unlink(temp_path)

    def test_process_file_error_handling(self):
        """Test that errors return empty list."""
        from app.ingest.processor import _process_file

        # Non-existent file
        chunks = _process_file("/nonexistent/path/file.md")
        assert chunks == []


@pytest.mark.unit
class TestIngestPaths:
    """Tests for main ingestion function."""

    @patch("app.ingest.processor.find_files")
    @patch("app.ingest.processor._process_file")
    @patch("app.retrieval.vector_store.VectorStore.add_documents")
    def test_ingest_paths_processes_files(self, mock_add_docs, mock_process, mock_find):
        """Test that ingest_paths processes all discovered files."""
        from app.ingest.processor import ingest_paths

        # Mock file discovery
        mock_find.return_value = ["/docs/file1.md", "/docs/file2.md"]

        # Mock processing to return chunks
        mock_process.side_effect = [
            [{"id": "chunk-1", "text": "Content 1", "metadata": {"source": "f1"}}],
            [{"id": "chunk-2", "text": "Content 2", "metadata": {"source": "f2"}}],
        ]

        result = ingest_paths(["/docs"])

        assert result == 2  # Two chunks added
        assert mock_process.call_count == 2
        mock_add_docs.assert_called_once()

    @patch("app.ingest.processor.find_files")
    @patch("app.ingest.processor._process_file")
    @patch("app.retrieval.vector_store.VectorStore.add_documents")
    def test_ingest_paths_deduplication(self, mock_add_docs, mock_process, mock_find):
        """Test that duplicate chunks are skipped."""
        from app.ingest.processor import ingest_paths

        mock_find.return_value = ["/docs/file1.md"]

        # Return duplicate chunks (same text content)
        mock_process.return_value = [
            {"id": "chunk-1", "text": "Same content", "metadata": {"source": "f1"}},
            {
                "id": "chunk-2",
                "text": "Same content",
                "metadata": {"source": "f2"},
            },  # Duplicate
            {
                "id": "chunk-3",
                "text": "Different content",
                "metadata": {"source": "f3"},
            },
        ]

        result = ingest_paths(["/docs"])

        # Should only add 2 unique chunks
        assert result == 2

    @patch("app.ingest.processor.find_files")
    @patch("app.ingest.processor._process_file")
    def test_ingest_paths_empty_file(self, mock_process, mock_find):
        """Test handling of files that produce no chunks."""
        from app.ingest.processor import ingest_paths

        mock_find.return_value = ["/docs/empty.md"]
        mock_process.return_value = []  # No chunks

        result = ingest_paths(["/docs"])

        assert result == 0
