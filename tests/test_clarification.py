"""
Tests for clarification prompt generation.

Tests the clarification module that generates helpful messages
when user questions are ambiguous or vague.
"""

import pytest
from unittest.mock import MagicMock

from app.prompting.clarification import (
    _clarification_examples,
    build_clarification_message,
)


@pytest.mark.unit
class TestClarificationExamples:
    """Tests for clarification example generation."""

    def test_gpa_question(self):
        """Test clarification for GPA-related questions."""
        topic, domains = _clarification_examples("What is your GPA?")

        assert topic == "your academic performance"
        assert "your undergraduate GPA" in domains
        assert "your graduate GPA" in domains
        assert "your overall GPA" in domains

    def test_experience_question(self):
        """Test clarification for experience questions."""
        topic, domains = _clarification_examples("Tell me about your experience")

        assert topic == "your experience"
        assert "your professional roles" in domains
        assert "projects you've completed" in domains
        assert "your technical skills" in domains

    def test_background_question(self):
        """Test clarification for background questions."""
        topic, domains = _clarification_examples("What is your background?")

        assert topic == "your background"
        assert "your education (degrees, GPA, coursework)" in domains
        assert "your work experience (roles, companies, responsibilities)" in domains
        assert "your certifications (CKA, AWS certifications)" in domains

    def test_qualifications_question(self):
        """Test clarification for qualifications questions."""
        topic, domains = _clarification_examples("Tell me your qualifications")

        assert topic == "your background"
        assert len(domains) >= 3  # Should have multiple options

    def test_history_question(self):
        """Test clarification for history questions."""
        topic, domains = _clarification_examples("What is your history?")

        assert topic == "your history"
        assert "your education history" in domains
        assert "your work history" in domains
        assert "key milestones" in domains

    def test_kubernetes_question(self):
        """Test clarification for Kubernetes-specific questions."""
        topic, domains = _clarification_examples("Tell me about kubernetes")

        assert topic == "Kubernetes"
        assert "projects that used Kubernetes" in domains
        assert "certifications like the CKA" in domains
        assert "work experience involving Kubernetes" in domains

    def test_default_fallback(self):
        """Test default clarification for non-specific questions."""
        topic, domains = _clarification_examples("random question")

        assert topic == "your profile"
        assert "your education" in domains
        assert "your work experience" in domains
        assert "your certifications" in domains
        assert "your skills" in domains

    def test_empty_question(self):
        """Test clarification with empty question."""
        topic, domains = _clarification_examples("")

        assert topic == "your profile"
        assert len(domains) == 4  # Default domains

    def test_none_question(self):
        """Test clarification with None question."""
        topic, domains = _clarification_examples(None)

        assert topic == "your profile"
        assert len(domains) == 4  # Default domains

    def test_case_insensitive(self):
        """Test that question matching is case-insensitive."""
        topic1, _ = _clarification_examples("KUBERNETES")
        topic2, _ = _clarification_examples("kubernetes")
        topic3, _ = _clarification_examples("KuBeRnEtEs")

        assert topic1 == topic2 == topic3 == "Kubernetes"


@pytest.mark.unit
class TestBuildClarificationMessage:
    """Tests for building complete clarification messages."""

    def test_single_domain(self):
        """Test message with a single domain option."""
        # Create a minimal mock config (not used in the function but required by signature)
        mock_config = MagicMock()

        # This should produce a single domain
        message = build_clarification_message("very specific question", mock_config)

        assert "Could you clarify" in message
        assert "specific details" in message
        # Single domain uses no commas/ors
        assert message.count(",") + message.count(" or ") >= 0

    def test_two_domains(self):
        """Test message with two domain options."""
        mock_config = MagicMock()

        message = build_clarification_message("Tell me your GPA", mock_config)

        assert "Could you clarify" in message
        assert "your academic performance" in message
        # Two domains should use "or" between them
        assert " or " in message

    def test_multiple_domains(self):
        """Test message with multiple domain options."""
        mock_config = MagicMock()

        message = build_clarification_message("What is your background?", mock_config)

        assert "Could you clarify" in message
        assert "your background" in message
        # Multiple domains should use commas and "or" for the last one
        assert "," in message
        assert ", or " in message

    def test_experience_clarification_message(self):
        """Test complete clarification message for experience question."""
        mock_config = MagicMock()

        message = build_clarification_message("experience", mock_config)

        assert "your experience" in message
        assert "your professional roles" in message
        assert "projects you've completed" in message
        assert "your technical skills" in message

    def test_default_clarification_message(self):
        """Test complete clarification message for generic question."""
        mock_config = MagicMock()

        message = build_clarification_message("random", mock_config)

        assert "your profile" in message
        assert "your education" in message
        assert "your work experience" in message
        assert "your certifications" in message
        assert "your skills" in message

    def test_message_structure(self):
        """Test that message has all expected components."""
        mock_config = MagicMock()

        message = build_clarification_message("test question", mock_config)

        # Should contain all these structural elements
        assert "Your question seems a bit broad" in message
        assert "Could you clarify which specific details" in message
        assert "For example, I can share information about" in message
        assert "Typical areas I can cover include" in message
        assert "Please let me know which detail to focus on" in message

    def test_kubernetes_clarification_complete(self):
        """Test complete Kubernetes-specific clarification."""
        mock_config = MagicMock()

        message = build_clarification_message("kubernetes", mock_config)

        assert "Kubernetes" in message
        assert "projects that used Kubernetes" in message
        assert "certifications like the CKA" in message
        assert "work experience involving Kubernetes" in message

    def test_history_clarification_complete(self):
        """Test complete history clarification."""
        mock_config = MagicMock()

        message = build_clarification_message("history", mock_config)

        assert "your history" in message
        assert "your education history" in message
        assert "your work history" in message
        assert "key milestones" in message
