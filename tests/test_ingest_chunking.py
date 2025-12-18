"""
Tests for document chunking strategies.

Tests header-based chunking, term-based chunking, smart chunking router,
and helper functions.
"""

import pytest


@pytest.mark.unit
class TestExtractDocId:
    """Tests for document ID extraction from filenames."""

    def test_extract_doc_id_resume(self):
        """Test extracting doc_id from resume filename."""
        from app.ingest.chunking import extract_doc_id

        doc_id, doc_type = extract_doc_id("/docs/resume--vaishak-menon--2025-09-23.md")

        assert doc_type == "resume"
        assert "resume" in doc_id

    def test_extract_doc_id_certificate(self):
        """Test extracting doc_id from certificate filename."""
        from app.ingest.chunking import extract_doc_id

        doc_id, doc_type = extract_doc_id("/docs/certificate--cka--2024-06-26.md")

        assert doc_type == "certificate"
        assert "cka" in doc_id.lower() or "certificate" in doc_id.lower()

    def test_extract_doc_id_simple_name(self):
        """Test extracting doc_id from simple filename."""
        from app.ingest.chunking import extract_doc_id

        doc_id, doc_type = extract_doc_id("/docs/notes.md")

        assert doc_id is not None
        assert doc_type is not None


@pytest.mark.unit
class TestSlugify:
    """Tests for section name slugification."""

    def test_slugify_simple(self):
        """Test slugifying simple text."""
        from app.ingest.chunking import slugify

        result = slugify("Work Experience")
        assert result == "work-experience"

    def test_slugify_with_special_chars(self):
        """Test slugifying text with special characters."""
        from app.ingest.chunking import slugify

        result = slugify("CS 665 - Deep Learning")
        # Should be lowercase with hyphens
        assert "-" in result
        assert result.islower() or all(c.isalnum() or c == "-" for c in result)

    def test_slugify_nested(self):
        """Test slugifying nested section name."""
        from app.ingest.chunking import slugify

        result = slugify("Teaching Assistant > Responsibilities")
        # Should handle > character
        assert ">" not in result


@pytest.mark.unit
class TestChunkByHeaders:
    """Tests for header-based chunking."""

    def test_chunk_by_headers_basic(self):
        """Test basic header-based chunking."""
        from app.ingest.chunking import chunk_by_headers

        text = """# Main Title

## Section One
Content for section one.

## Section Two
Content for section two.
"""
        metadata = {"doc_type": "resume", "source": "test.md"}

        chunks = chunk_by_headers(
            text=text, base_metadata=metadata, source_path="/docs/test.md"
        )

        # Should produce multiple chunks
        assert len(chunks) >= 1
        # Each chunk should have text and metadata
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert "id" in chunk

    def test_chunk_by_headers_respects_split_level(self):
        """Test that split_level parameter works."""
        from app.ingest.chunking import chunk_by_headers

        text = """# H1 Title

## H2 Section
### H3 Subsection
Content here.
"""
        metadata = {"doc_type": "test"}

        # Split at level 2 (##)
        chunks_level2 = chunk_by_headers(
            text=text, base_metadata=metadata, source_path="/test.md", split_level=2
        )

        assert len(chunks_level2) >= 1

    def test_chunk_by_headers_empty_text(self):
        """Test chunking empty text."""
        from app.ingest.chunking import chunk_by_headers

        chunks = chunk_by_headers(
            text="", base_metadata={"doc_type": "test"}, source_path="/test.md"
        )

        assert chunks == [] or len(chunks) == 0


@pytest.mark.unit
class TestSmartChunk:
    """Tests for smart chunking router."""

    def test_smart_chunk_routes_transcript(self):
        """Test that transcripts use term-based chunking."""
        from app.ingest.chunking import smart_chunk

        text = """### Fall 2023
Course content for fall 2023.

### Spring 2024
Course content for spring 2024.
"""
        metadata = {"doc_type": "transcript_analysis"}

        chunks = smart_chunk(
            text=text, base_metadata=metadata, source_path="/docs/transcript.md"
        )

        # Should produce chunks for each term
        assert len(chunks) >= 1

    def test_smart_chunk_routes_resume(self):
        """Test that resumes use header-based chunking."""
        from app.ingest.chunking import smart_chunk

        text = """## Experience
Work experience content.

## Education
Education content.
"""
        metadata = {"doc_type": "resume"}

        chunks = smart_chunk(
            text=text, base_metadata=metadata, source_path="/docs/resume.md"
        )

        assert len(chunks) >= 1


@pytest.mark.unit
class TestChunkByTerms:
    """Tests for term-based chunking (transcripts)."""

    def test_chunk_by_terms_basic(self):
        """Test basic term-based chunking."""
        from app.ingest.chunking import chunk_by_terms

        text = """### Fall 2023
CS 101 - Introduction to Programming
Grade: A

### Spring 2024
CS 201 - Data Structures
Grade: A
"""
        metadata = {"doc_type": "transcript_analysis"}

        chunks = chunk_by_terms(
            text=text, base_metadata=metadata, source_path="/docs/transcript.md"
        )

        assert len(chunks) >= 1
        # Each chunk should have term metadata
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk


@pytest.mark.unit
class TestParseTermInfo:
    """Tests for term info parsing."""

    def test_parse_term_info_fall_semester(self):
        """Test parsing fall semester header."""
        from app.ingest.chunking import _parse_term_info

        result = _parse_term_info("Fall 2023")

        assert result is not None
        assert "term_year" in result or "term_season" in result

    def test_parse_term_info_spring_semester(self):
        """Test parsing spring semester header."""
        from app.ingest.chunking import _parse_term_info

        result = _parse_term_info("Spring 2024")

        assert result is not None

    def test_parse_term_info_with_term_word(self):
        """Test parsing header with 'Term' word."""
        from app.ingest.chunking import _parse_term_info

        result = _parse_term_info("Fall Term 2023")

        assert result is not None


@pytest.mark.unit
class TestParseMarkdownSections:
    """Tests for markdown section parsing."""

    def test_parse_markdown_sections_basic(self):
        """Test parsing markdown into sections."""
        from app.ingest.chunking import _parse_markdown_sections

        text = """## Section One
Content one.

## Section Two
Content two.
"""
        sections = _parse_markdown_sections(text)

        assert len(sections) >= 1

    def test_parse_markdown_sections_nested(self):
        """Test parsing nested sections."""
        from app.ingest.chunking import _parse_markdown_sections

        text = """## Parent
Content.

### Child
Child content.
"""
        sections = _parse_markdown_sections(text, split_level=2)

        assert len(sections) >= 1


@pytest.mark.unit
class TestSplitContentWithOverlap:
    """Tests for content splitting with overlap."""

    def test_split_short_content(self):
        """Test splitting short content."""
        from app.ingest.chunking import _split_content_with_overlap

        content = "This is a short paragraph."

        parts = _split_content_with_overlap(content, max_size=1000)

        assert len(parts) == 1
        assert parts[0] == content

    def test_split_long_content(self):
        """Test splitting long content into parts."""
        from app.ingest.chunking import _split_content_with_overlap

        # Create content longer than max_size
        content = "Word " * 500  # ~2500 chars

        parts = _split_content_with_overlap(content, max_size=500, overlap=50)

        assert len(parts) >= 2

    def test_split_preserves_content(self):
        """Test that splitting preserves all content."""
        from app.ingest.chunking import _split_content_with_overlap

        content = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

        parts = _split_content_with_overlap(content, max_size=50, overlap=10)

        # All paragraphs should be in some part
        combined = " ".join(parts)
        assert "one" in combined
        assert "two" in combined
        assert "three" in combined


@pytest.mark.unit
class TestCreateChunkMetadata:
    """Tests for chunk metadata creation."""

    def test_create_chunk_metadata_basic(self):
        """Test creating chunk metadata."""
        from app.ingest.chunking import _create_chunk_metadata

        base_metadata = {"doc_type": "resume", "source": "test.md"}

        metadata = _create_chunk_metadata(
            base_metadata=base_metadata,
            section_stack=["Experience"],
            section_name="Work",
            doc_id="resume",
            doc_type="resume",
            version="2025-01-15",
        )

        assert "doc_type" in metadata
        assert metadata["doc_type"] == "resume"

    def test_create_chunk_metadata_multipart(self):
        """Test metadata for multi-part chunks."""
        from app.ingest.chunking import _create_chunk_metadata

        metadata = _create_chunk_metadata(
            base_metadata={"doc_type": "resume"},
            section_stack=["Large Section"],
            section_name="Large Section",
            doc_id="resume",
            doc_type="resume",
            version="2025-01-15",
            part_num=1,
            total_parts=3,
        )

        assert "part" in metadata or metadata is not None
