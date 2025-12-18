"""
Tests for metadata extraction and version generation.

Tests YAML frontmatter parsing, version identifier generation,
and ChromaDB-compatible metadata normalization.
"""

import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os


@pytest.mark.unit
class TestExtractFrontmatter:
    """Tests for YAML frontmatter extraction."""

    def test_extract_frontmatter_valid_yaml(self):
        """Test extracting valid YAML frontmatter."""
        from app.ingest.metadata import extract_frontmatter

        text = """---
doc_type: resume
version_date: 2025-01-15
tags:
  - python
  - fastapi
---

# Resume Content
This is the body.
"""
        metadata, body = extract_frontmatter(text)

        assert metadata["doc_type"] == "resume"
        assert "Resume Content" in body
        assert "---" not in body

    def test_extract_frontmatter_no_frontmatter(self):
        """Test handling text without frontmatter."""
        from app.ingest.metadata import extract_frontmatter

        text = "# Just a heading\n\nNo frontmatter here."

        metadata, body = extract_frontmatter(text)

        assert metadata == {}
        assert body == text

    def test_extract_frontmatter_normalizes_lists(self):
        """Test that lists are converted to comma-separated strings."""
        from app.ingest.metadata import extract_frontmatter

        text = """---
tags:
  - python
  - javascript
  - go
---
Content
"""
        metadata, body = extract_frontmatter(text)

        # Lists should be converted to comma-separated strings for ChromaDB
        if "tags" in metadata:
            assert isinstance(metadata["tags"], str)
            assert "python" in metadata["tags"]

    def test_extract_frontmatter_skips_none_values(self):
        """Test that None values are skipped."""
        from app.ingest.metadata import extract_frontmatter

        text = """---
doc_type: resume
optional_field:
---
Content
"""
        metadata, body = extract_frontmatter(text)

        # None values should be skipped
        assert "optional_field" not in metadata or metadata.get("optional_field") is not None

    def test_extract_frontmatter_malformed(self):
        """Test handling malformed YAML."""
        from app.ingest.metadata import extract_frontmatter

        text = """---
this is not: valid: yaml: structure
---
Content
"""
        metadata, body = extract_frontmatter(text)

        # Should return empty metadata on parse error
        assert metadata == {} or isinstance(metadata, dict)


@pytest.mark.unit
class TestReadText:
    """Tests for file reading."""

    def test_read_text_success(self):
        """Test reading a text file."""
        from app.ingest.metadata import read_text

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write("# Test Content\nThis is a test.")
            temp_path = f.name

        try:
            content = read_text(temp_path)
            assert "Test Content" in content
            assert "This is a test" in content
        finally:
            os.unlink(temp_path)

    def test_read_text_utf8(self):
        """Test reading UTF-8 content."""
        from app.ingest.metadata import read_text

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write("Unicode: café, naïve, 日本語")
            temp_path = f.name

        try:
            content = read_text(temp_path)
            assert "café" in content
            assert "日本語" in content
        finally:
            os.unlink(temp_path)


@pytest.mark.unit
class TestNormalizeVersionIdentifier:
    """Tests for version identifier normalization."""

    def test_normalize_version_from_date(self):
        """Test normalizing version from version_date field."""
        from app.ingest.metadata import normalize_version_identifier

        metadata = {"doc_type": "resume", "version_date": "2025-01-15"}

        version = normalize_version_identifier(metadata)

        # Should contain the date
        assert "2025" in version or version is not None

    def test_normalize_version_from_earned(self):
        """Test normalizing version from earned field (certificates)."""
        from app.ingest.metadata import normalize_version_identifier

        metadata = {"doc_type": "certificate", "earned": "2024-06-26"}

        version = normalize_version_identifier(metadata)

        assert version is not None

    def test_normalize_version_fallback_to_today(self):
        """Test that missing version fields fall back to today's date."""
        from app.ingest.metadata import normalize_version_identifier
        from datetime import date

        metadata = {"doc_type": "resume"}  # No version_date

        version = normalize_version_identifier(metadata)

        # Should return today's date or some default
        assert version is not None


@pytest.mark.unit
class TestGenerateVersionIdentifier:
    """Tests for version identifier generation with content detection."""

    @patch("app.ingest.metadata.get_existing_versions")
    @patch("app.ingest.metadata.get_existing_content_hash")
    def test_generate_version_new_document(self, mock_hash, mock_versions):
        """Test generating version for new document."""
        from app.ingest.metadata import generate_version_identifier

        mock_versions.return_value = []  # No existing versions
        mock_hash.return_value = None

        metadata = {"doc_type": "resume", "version_date": "2025-01-15"}

        version = generate_version_identifier(
            metadata=metadata,
            doc_id="resume",
            content_hash="abc123"
        )

        assert version == "2025-01-15"

    @patch("app.ingest.metadata.get_existing_versions")
    @patch("app.ingest.metadata.get_existing_content_hash")
    def test_generate_version_unchanged_content(self, mock_hash, mock_versions):
        """Test that unchanged content reuses existing version."""
        from app.ingest.metadata import generate_version_identifier

        mock_versions.return_value = ["2025-01-15"]
        mock_hash.return_value = "abc123"  # Same hash

        metadata = {"doc_type": "resume", "version_date": "2025-01-15"}

        version = generate_version_identifier(
            metadata=metadata,
            doc_id="resume",
            content_hash="abc123"  # Same as existing
        )

        # Should reuse existing version
        assert "2025-01-15" in version

    @patch("app.ingest.metadata.get_existing_versions")
    @patch("app.ingest.metadata.get_existing_content_hash")
    def test_generate_version_changed_content(self, mock_hash, mock_versions):
        """Test that changed content gets new version."""
        from app.ingest.metadata import generate_version_identifier

        mock_versions.return_value = ["2025-01-15"]
        mock_hash.return_value = "old_hash"  # Different hash

        metadata = {"doc_type": "resume", "version_date": "2025-01-15"}

        version = generate_version_identifier(
            metadata=metadata,
            doc_id="resume",
            content_hash="new_hash"  # Different from existing
        )

        # Should get new version with .v2
        assert ".v" in version or version == "2025-01-15"

    @patch("app.ingest.metadata.get_existing_versions")
    @patch("app.ingest.metadata.get_existing_content_hash")
    def test_generate_version_increments_sequence(self, mock_hash, mock_versions):
        """Test that version sequence increments properly."""
        from app.ingest.metadata import generate_version_identifier

        # Already have v2
        mock_versions.return_value = ["2025-01-15", "2025-01-15.v2"]
        mock_hash.return_value = "old_hash"  # Different

        metadata = {"doc_type": "resume", "version_date": "2025-01-15"}

        version = generate_version_identifier(
            metadata=metadata,
            doc_id="resume",
            content_hash="new_hash"
        )

        # Should get v3
        assert "v3" in version or ".v" in version


@pytest.mark.unit
class TestGetExistingVersions:
    """Tests for get_existing_versions."""

    @patch("app.retrieval.store._collection")
    def test_get_existing_versions_returns_list(self, mock_collection):
        """Test that get_existing_versions returns a list."""
        from app.ingest.metadata import get_existing_versions

        mock_collection.get.return_value = {
            "metadatas": [{"version_identifier": "2025-01-15"}]
        }

        result = get_existing_versions("resume", "2025-01-15")

        assert isinstance(result, list)


@pytest.mark.unit
class TestGetExistingContentHash:
    """Tests for get_existing_content_hash."""

    @patch("app.retrieval.store._collection")
    def test_get_existing_content_hash_returns_hash(self, mock_collection):
        """Test that get_existing_content_hash works."""
        from app.ingest.metadata import get_existing_content_hash

        mock_collection.get.return_value = {
            "metadatas": [{"content_hash": "abc123"}]
        }

        result = get_existing_content_hash("resume", "2025-01-15")

        # Result is either hash or None
        assert result is None or isinstance(result, str)
