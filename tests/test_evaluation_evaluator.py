"""
Tests for evaluation framework.

Tests RetrievalEvaluator and AnswerEvaluator classes.
"""

import pytest
from unittest.mock import MagicMock


@pytest.mark.unit
class TestRetrievalEvaluator:
    """Tests for RetrievalEvaluator."""

    def test_evaluate_single_query_perfect(self):
        """Test evaluating single query with perfect retrieval."""
        from app.evaluation.evaluator import RetrievalEvaluator

        evaluator = RetrievalEvaluator(k_values=[1, 3, 5])

        result = evaluator.evaluate_single_query(
            query="Test question",
            retrieved_ids=["chunk-1", "chunk-2", "chunk-3"],
            relevant_ids=["chunk-1", "chunk-2"]
        )

        # Results are under "metrics" key
        assert result["metrics"]["recall@3"] == 1.0
        assert result["metrics"]["precision@3"] == pytest.approx(2/3)
        assert result["metrics"]["mrr"] == 1.0  # First result is relevant

    def test_evaluate_single_query_no_relevant(self):
        """Test evaluating query with no relevant docs retrieved."""
        from app.evaluation.evaluator import RetrievalEvaluator

        evaluator = RetrievalEvaluator(k_values=[3])

        result = evaluator.evaluate_single_query(
            query="Test",
            retrieved_ids=["chunk-1", "chunk-2", "chunk-3"],
            relevant_ids=["chunk-99"]  # Not in retrieved
        )

        assert result["metrics"]["recall@3"] == 0.0
        assert result["metrics"]["precision@3"] == 0.0
        assert result["metrics"]["mrr"] == 0.0

    def test_evaluate_batch(self):
        """Test batch evaluation of multiple queries."""
        from app.evaluation.evaluator import RetrievalEvaluator

        evaluator = RetrievalEvaluator(k_values=[3])

        test_cases = [
            {
                "query": "Query 1",
                "expected_chunks": ["chunk-1", "chunk-2"]
            },
            {
                "query": "Query 2",
                "expected_chunks": ["chunk-3"]
            }
        ]

        retrieval_results = [
            {"query": "Query 1", "chunk_ids": ["chunk-1", "chunk-2", "chunk-4"]},  # 2/2 relevant
            {"query": "Query 2", "chunk_ids": ["chunk-3", "chunk-5", "chunk-6"]}   # 1/1 relevant
        ]

        report = evaluator.evaluate_batch(test_cases, retrieval_results)

        assert "aggregate" in report
        assert "individual" in report

    def test_evaluate_single_query_by_content(self):
        """Test content-based evaluation."""
        from app.evaluation.evaluator import RetrievalEvaluator

        evaluator = RetrievalEvaluator(k_values=[3])

        retrieved_chunks = [
            {"id": "chunk-1", "content": "Python experience: 5 years"},
            {"id": "chunk-2", "content": "JavaScript skills"},
            {"id": "chunk-3", "content": "Worked with FastAPI"}
        ]

        required_content = ["Python", "5 years"]

        result = evaluator.evaluate_single_query_by_content(
            query="Python experience",
            retrieved_chunks=retrieved_chunks,
            required_content=required_content,
            k=3
        )

        # Uses actual return key from implementation
        assert result["content_pass"] is True


@pytest.mark.unit
class TestAnswerEvaluator:
    """Tests for AnswerEvaluator."""

    def test_evaluate_single_answer_all_facts_present(self):
        """Test answer containing all required facts."""
        from app.evaluation.evaluator import AnswerEvaluator

        evaluator = AnswerEvaluator()

        answer = "I have 5 years of Python experience and worked at Google."
        expected_facts = ["5 years", "Python", "Google"]

        result = evaluator.evaluate_single_answer(
            query="What is my experience?",
            answer=answer,
            expected_facts=expected_facts,
            must_not_contain=[]
        )

        # Uses actual return keys from the implementation
        assert result["fact_coverage"] == 1.0
        assert len(result["missing_facts"]) == 0
        assert result["passed"] is True

    def test_evaluate_single_answer_missing_facts(self):
        """Test answer missing some required facts."""
        from app.evaluation.evaluator import AnswerEvaluator

        evaluator = AnswerEvaluator()

        answer = "I have Python experience."
        expected_facts = ["Python", "5 years", "Google"]

        result = evaluator.evaluate_single_answer(
            query="Test",
            answer=answer,
            expected_facts=expected_facts,
            must_not_contain=[]
        )

        assert len(result["found_facts"]) == 1  # Only "Python"
        assert len(result["missing_facts"]) == 2  # Missing "5 years" and "Google"
        assert result["passed"] is False

    def test_evaluate_single_answer_prohibited_content(self):
        """Test detection of prohibited content."""
        from app.evaluation.evaluator import AnswerEvaluator

        evaluator = AnswerEvaluator()

        answer = "Based on the provided context, I have Python experience."
        must_not_contain = ["provided context", "I don't have"]

        result = evaluator.evaluate_single_answer(
            query="Test",
            answer=answer,
            expected_facts=["Python"],
            must_not_contain=must_not_contain
        )

        assert result["has_prohibited"] is True
        assert "provided context" in result["found_prohibited"]

    def test_evaluate_single_answer_alternative_facts(self):
        """Test alternative fact matching with pipes."""
        from app.evaluation.evaluator import AnswerEvaluator

        evaluator = AnswerEvaluator()

        answer = "I worked at Google for 3 years."
        # Accept either "Google" or "Alphabet"
        expected_facts = ["Google|Alphabet", "3 years"]

        result = evaluator.evaluate_single_answer(
            query="Test",
            answer=answer,
            expected_facts=expected_facts,
            must_not_contain=[]
        )

        assert result["passed"] is True

    def test_evaluate_batch_answers(self):
        """Test batch answer evaluation."""
        from app.evaluation.evaluator import AnswerEvaluator

        evaluator = AnswerEvaluator()

        test_cases = [
            {
                "query": "Q1",
                "expected_facts": ["Python", "5 years"],
                "must_not_contain": []
            },
            {
                "query": "Q2",
                "expected_facts": ["Google"],
                "must_not_contain": ["I don't know"]
            }
        ]

        answers = [
            "I have 5 years of Python experience.",  # All facts found
            "I don't know about that."                # Prohibited content
        ]

        report = evaluator.evaluate_batch(test_cases, answers)

        assert "aggregate" in report
        assert "individual" in report
        assert report["individual"][0]["passed"] is True
        assert report["individual"][1]["has_prohibited"] is True


@pytest.mark.unit
class TestEvaluationSummary:
    """Tests for evaluation summary generation."""

    def test_generate_summary_all_pass(self):
        """Test summary when all queries pass."""
        from app.evaluation.evaluator import RetrievalEvaluator

        evaluator = RetrievalEvaluator(k_values=[5])

        aggregate = {"recall@5": 1.0, "precision@5": 1.0, "mrr": 1.0}
        individual = [
            {"test_id": "test-1", "query": "Q1", "difficulty": "easy", "found_any": True,
             "metrics": {"recall@5": 1.0, "precision@5": 1.0, "mrr": 1.0}},
            {"test_id": "test-2", "query": "Q2", "difficulty": "easy", "found_any": True,
             "metrics": {"recall@5": 1.0, "precision@5": 1.0, "mrr": 1.0}}
        ]

        summary = evaluator._generate_summary(aggregate, individual)

        # Returns a dict, not a string
        assert isinstance(summary, dict)
        assert "total_queries" in summary

    def test_generate_summary_some_failures(self):
        """Test summary when some queries fail."""
        from app.evaluation.evaluator import RetrievalEvaluator

        evaluator = RetrievalEvaluator(k_values=[5])

        aggregate = {"recall@5": 0.5, "precision@5": 0.6, "mrr": 0.4}
        individual = [
            {"test_id": "test-1", "query": "Q1", "difficulty": "easy", "found_any": True,
             "metrics": {"recall@5": 1.0, "precision@5": 1.0, "mrr": 1.0}},
            {"test_id": "test-2", "query": "Q2", "difficulty": "hard", "found_any": False,
             "metrics": {"recall@5": 0.0, "precision@5": 0.0, "mrr": 0.0}}  # Failed
        ]

        summary = evaluator._generate_summary(aggregate, individual)

        assert isinstance(summary, dict)
        assert summary["failures"] == 1
