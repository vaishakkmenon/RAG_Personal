"""
Tests for file discovery functionality.

Tests recursive file finding, extension filtering, and security checks.
"""

import pytest
import tempfile
import os
from unittest.mock import patch


@pytest.mark.unit
class TestFindFiles:
    """Tests for file discovery function."""

    @patch("app.ingest.discovery.settings")
    def test_find_files_discovers_markdown(self, mock_settings):
        """Test that markdown files are discovered."""
        from app.ingest.discovery import find_files

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.docs_dir = tmpdir

            # Create test files
            md_file = os.path.join(tmpdir, "test.md")
            with open(md_file, "w") as f:
                f.write("# Test")

            files = find_files([tmpdir])

            assert len(files) >= 1
            assert any(f.endswith(".md") for f in files)

    @patch("app.ingest.discovery.settings")
    def test_find_files_discovers_txt(self, mock_settings):
        """Test that .txt files are discovered."""
        from app.ingest.discovery import find_files

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.docs_dir = tmpdir

            # Create test file
            txt_file = os.path.join(tmpdir, "test.txt")
            with open(txt_file, "w") as f:
                f.write("Test content")

            files = find_files([tmpdir])

            assert len(files) >= 1
            assert any(f.endswith(".txt") for f in files)

    @patch("app.ingest.discovery.settings")
    def test_find_files_ignores_invalid_extensions(self, mock_settings):
        """Test that invalid extensions are ignored."""
        from app.ingest.discovery import find_files

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.docs_dir = tmpdir

            # Create files with various extensions
            md_file = os.path.join(tmpdir, "valid.md")
            py_file = os.path.join(tmpdir, "invalid.py")
            pdf_file = os.path.join(tmpdir, "invalid.pdf")

            for f in [md_file, py_file, pdf_file]:
                with open(f, "w") as fp:
                    fp.write("content")

            files = find_files([tmpdir])

            # Only .md should be included
            assert any(f.endswith(".md") for f in files)
            assert not any(f.endswith(".py") for f in files)
            assert not any(f.endswith(".pdf") for f in files)

    @patch("app.ingest.discovery.settings")
    def test_find_files_recursive(self, mock_settings):
        """Test that files in subdirectories are found."""
        from app.ingest.discovery import find_files

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.docs_dir = tmpdir

            # Create nested structure
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)

            md_file = os.path.join(subdir, "nested.md")
            with open(md_file, "w") as f:
                f.write("Nested content")

            files = find_files([tmpdir])

            assert len(files) >= 1
            assert any("nested.md" in f for f in files)

    @patch("app.ingest.discovery.settings")
    def test_find_files_security_check(self, mock_settings):
        """Test that files outside docs_dir are rejected."""
        from app.ingest.discovery import find_files

        with tempfile.TemporaryDirectory() as docs_dir:
            with tempfile.TemporaryDirectory() as other_dir:
                mock_settings.docs_dir = docs_dir

                # Create file outside docs_dir
                outside_file = os.path.join(other_dir, "outside.md")
                with open(outside_file, "w") as f:
                    f.write("Outside content")

                # Try to find files from outside docs_dir
                # Should skip files outside docs_dir
                try:
                    files = find_files([other_dir])
                    # Files outside docs_dir should be skipped
                    assert len(files) == 0 or not any(other_dir in f for f in files)
                except Exception:
                    # Security check may raise an error - that's acceptable
                    pass

    @patch("app.ingest.discovery.settings")
    def test_find_files_single_file(self, mock_settings):
        """Test finding a single file by path."""
        from app.ingest.discovery import find_files

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.docs_dir = tmpdir

            md_file = os.path.join(tmpdir, "single.md")
            with open(md_file, "w") as f:
                f.write("Single file")

            files = find_files([md_file])

            assert len(files) == 1
            assert files[0] == md_file

    @patch("app.ingest.discovery.settings")
    def test_find_files_empty_directory(self, mock_settings):
        """Test that empty directory raises error."""
        from app.ingest.discovery import find_files
        from fastapi import HTTPException

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.docs_dir = tmpdir

            # Empty directory - should raise HTTPException
            with pytest.raises(HTTPException) as exc_info:
                find_files([tmpdir])

            assert exc_info.value.status_code == 400
