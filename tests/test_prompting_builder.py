"""
Tests for prompt builder functionality.

Tests prompt construction, context formatting, conversation history,
and answer validation methods.
"""

import pytest


@pytest.mark.unit
class TestFormatContext:
    """Tests for context formatting."""

    def test_format_context_basic(self):
        """Test basic context formatting with XML format."""
        from app.prompting.builder import _format_context_xml

        chunks = [
            {
                "source": "resume.md",
                "text": "I have Python experience.",
                "metadata": {},
            },
            {"source": "certs.md", "text": "CKA certification.", "metadata": {}},
        ]

        result = _format_context_xml(chunks)

        # XML format uses [1], [2] numbered citations without filenames
        assert "[1]" in result
        assert "[2]" in result
        assert "Python experience" in result
        assert "CKA certification" in result

    def test_format_context_skips_empty(self):
        """Test that empty text chunks are skipped."""
        from app.prompting.builder import _format_context_xml

        chunks = [
            {"source": "file1.md", "text": "Valid content", "metadata": {}},
            {"source": "file2.md", "text": "", "metadata": {}},  # Empty
            {"source": "file3.md", "text": "   ", "metadata": {}},  # Whitespace only
        ]

        result = _format_context_xml(chunks)

        assert "Valid content" in result
        # Empty chunks should not get citation numbers
        assert result.count("[") == 1  # Only one citation marker


@pytest.mark.unit
class TestFormatConversationHistory:
    """Tests for conversation history formatting."""

    def test_format_conversation_history_basic(self):
        """Test formatting conversation history with XML tags."""
        from app.prompting.builder import _format_history_xml

        history = [
            {"role": "user", "content": "What is my GPA?"},
            {"role": "assistant", "content": "Your GPA is 4.00."},
        ]

        result = _format_history_xml(history)

        # XML format uses <user>...</user> and <assistant>...</assistant> tags
        assert "<user>" in result or "GPA" in result
        assert "GPA" in result

    def test_format_conversation_history_empty(self):
        """Test formatting empty history."""
        from app.prompting.builder import _format_history_xml

        result = _format_history_xml([])

        assert result == ""

    def test_format_conversation_history_none(self):
        """Test formatting None history."""
        from app.prompting.builder import _format_history_xml

        result = _format_history_xml(None)

        assert result == ""


@pytest.mark.unit
class TestPromptBuilder:
    """Tests for PromptBuilder class."""

    def test_prompt_builder_initialization(self):
        """Test PromptBuilder initialization."""
        from app.prompting.builder import PromptBuilder

        builder = PromptBuilder()

        assert builder.config is not None

    def test_build_prompt_success(self):
        """Test building a prompt successfully."""
        from app.prompting.builder import PromptBuilder

        builder = PromptBuilder()

        chunks = [
            {"source": "resume.md", "text": "5 years Python experience", "metadata": {}}
        ]

        result = builder.build_prompt(
            question="What is my Python experience?", context_chunks=chunks
        )

        assert result.status == "success"
        assert result.prompt is not None
        assert "Python" in result.prompt

    def test_build_prompt_with_keywords(self):
        """Test building prompt with keywords (ignored in XML mode)."""
        from app.prompting.builder import PromptBuilder

        builder = PromptBuilder()

        result = builder.build_prompt(
            question="Tell me about skills",
            context_chunks=[
                {"source": "test.md", "text": "Test content", "metadata": {}}
            ],
            keywords=["Python", "AWS", "Docker"],
        )

        assert result.status == "success"
        # In XML mode, keywords are ignored but the prompt should still be built
        assert result.prompt is not None
        # The question should be in the prompt
        assert "skills" in result.prompt

    def test_build_prompt_with_history(self):
        """Test building prompt with conversation history."""
        from app.prompting.builder import PromptBuilder

        builder = PromptBuilder()

        history = [
            {"role": "user", "content": "What's my GPA?"},
            {"role": "assistant", "content": "Your GPA is 4.00."},
        ]

        result = builder.build_prompt(
            question="What about my coursework?",
            context_chunks=[{"source": "test.md", "text": "Test", "metadata": {}}],
            conversation_history=history,
        )

        assert result.status == "success"
        # History should be included in prompt
        assert "GPA" in result.prompt or "PREVIOUS" in result.prompt


@pytest.mark.unit
class TestPromptBuilderValidation:
    """Tests for answer validation methods."""

    def test_is_refusal_empty_answer(self):
        """Test that empty answers are detected as refusals."""
        from app.prompting.builder import PromptBuilder

        builder = PromptBuilder()

        # Empty answers should be refusals
        assert builder.is_refusal("") is True
        assert builder.is_refusal("   ") is True

    def test_is_refusal_accepts_valid(self):
        """Test that valid answers are not marked as refusals."""
        from app.prompting.builder import PromptBuilder

        builder = PromptBuilder()

        valid_answers = [
            "I have 5 years of Python experience.",
            "Yes, I earned the CKA certification in 2024.",
            "My GPA is 4.00.",
        ]

        for answer in valid_answers:
            assert builder.is_refusal(answer) is False

    def test_is_refusal_with_refusal_cues(self):
        """Test detection based on config refusal cues."""
        from app.prompting.builder import PromptBuilder

        builder = PromptBuilder()

        # Check what cues exist in config
        if hasattr(builder.config, "refusal_cues") and builder.config.refusal_cues:
            # Test with a known refusal cue
            for cue in builder.config.refusal_cues[:1]:
                test_answer = f"Sorry, {cue}."
                assert builder.is_refusal(test_answer) is True
