"""
Evaluation runners for retrieval and answer quality.
"""

import logging
from typing import Any, Dict, List, Optional

from .metrics import (
    calculate_recall_at_k,
    calculate_mrr,
    calculate_precision_at_k,
    calculate_ndcg_at_k,
    calculate_f1_score,
    _has_any_match,
)

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Evaluates retrieval quality using IR metrics."""

    def __init__(self, k_values: List[int] = None):
        """Initialize evaluator.

        Args:
            k_values: List of K values for Recall@K, Precision@K, etc.
                     Defaults to [1, 3, 5, 10]
        """
        self.k_values = k_values or [1, 3, 5, 10]

    def evaluate_single_query(
        self,
        query: str,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> Dict[str, Any]:
        """Evaluate retrieval for a single query.

        Args:
            query: The query string
            retrieved_ids: List of retrieved chunk IDs (in rank order)
            relevant_ids: List of relevant chunk IDs (ground truth)

        Returns:
            Dictionary with metrics for this query
        """
        result = {
            "query": query,
            "num_retrieved": len(retrieved_ids),
            "num_relevant": len(relevant_ids),
            "metrics": {}
        }

        # Calculate metrics for each K value
        for k in self.k_values:
            if k <= len(retrieved_ids):
                recall = calculate_recall_at_k(retrieved_ids, relevant_ids, k)
                precision = calculate_precision_at_k(retrieved_ids, relevant_ids, k)
                ndcg = calculate_ndcg_at_k(retrieved_ids, relevant_ids, k)
                f1 = calculate_f1_score(precision, recall)

                result["metrics"][f"recall@{k}"] = recall
                result["metrics"][f"precision@{k}"] = precision
                result["metrics"][f"ndcg@{k}"] = ndcg
                result["metrics"][f"f1@{k}"] = f1

        # MRR (not K-dependent)
        result["metrics"]["mrr"] = calculate_mrr(retrieved_ids, relevant_ids)

        # Check if any relevant document was retrieved (using prefix matching)
        result["found_any"] = _has_any_match(retrieved_ids, relevant_ids)

        return result

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        retrieval_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate retrieval for multiple queries.

        Args:
            test_cases: List of test case dicts with 'query' and 'expected_chunks'
            retrieval_results: List of retrieval result dicts with 'query' and 'chunk_ids'

        Returns:
            Aggregate metrics across all queries
        """
        individual_results = []

        for test_case, retrieval_result in zip(test_cases, retrieval_results):
            result = self.evaluate_single_query(
                query=test_case["query"],
                retrieved_ids=retrieval_result["chunk_ids"],
                relevant_ids=test_case["expected_chunks"]
            )
            result["test_id"] = test_case.get("id", "unknown")
            result["difficulty"] = test_case.get("difficulty", "unknown")
            result["category"] = test_case.get("category", "unknown")
            individual_results.append(result)

        # Calculate aggregate metrics
        aggregate = self._calculate_aggregate_metrics(individual_results)

        return {
            "aggregate": aggregate,
            "individual": individual_results,
            "summary": self._generate_summary(aggregate, individual_results)
        }

    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate average metrics across all queries."""
        if not results:
            return {}

        aggregate = {}

        # Get all metric names from first result
        if results[0]["metrics"]:
            metric_names = results[0]["metrics"].keys()

            for metric_name in metric_names:
                values = [r["metrics"][metric_name] for r in results]
                aggregate[metric_name] = sum(values) / len(values)

        # Additional aggregate metrics
        aggregate["found_any_rate"] = sum(r["found_any"] for r in results) / len(results)

        return aggregate

    def _generate_summary(self, aggregate: Dict, individual: List[Dict]) -> Dict[str, Any]:
        """Generate human-readable summary."""
        # Find failures (recall@5 < 0.8)
        failures = [
            r for r in individual
            if r["metrics"].get("recall@5", 0) < 0.8
        ]

        # Group by difficulty
        by_difficulty = {}
        for result in individual:
            diff = result["difficulty"]
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(result)

        difficulty_metrics = {}
        for diff, results in by_difficulty.items():
            difficulty_metrics[diff] = {
                "count": len(results),
                "avg_recall@5": sum(r["metrics"].get("recall@5", 0) for r in results) / len(results),
                "avg_mrr": sum(r["metrics"].get("mrr", 0) for r in results) / len(results)
            }

        return {
            "total_queries": len(individual),
            "failures": len(failures),
            "pass_rate": (len(individual) - len(failures)) / len(individual) if individual else 0,
            "failure_details": [
                {
                    "test_id": f["test_id"],
                    "query": f["query"],
                    "recall@5": f["metrics"].get("recall@5", 0),
                    "found_any": f["found_any"]
                }
                for f in failures
            ],
            "by_difficulty": difficulty_metrics
        }


class AnswerEvaluator:
    """Evaluates answer quality."""

    def evaluate_single_answer(
        self,
        query: str,
        answer: str,
        expected_facts: List[str],
        must_not_contain: List[str] = None
    ) -> Dict[str, Any]:
        """Evaluate a single answer.

        Args:
            query: The query string
            answer: The generated answer
            expected_facts: List of facts that should be in the answer
            must_not_contain: List of facts that should NOT be in the answer

        Returns:
            Dictionary with evaluation results
        """
        must_not_contain = must_not_contain or []

        # Check expected facts (case-insensitive)
        answer_lower = answer.lower()
        found_facts = [fact for fact in expected_facts if fact.lower() in answer_lower]
        missing_facts = [fact for fact in expected_facts if fact.lower() not in answer_lower]

        # Check prohibited content
        found_prohibited = [fact for fact in must_not_contain if fact.lower() in answer_lower]

        # Calculate score
        fact_coverage = len(found_facts) / len(expected_facts) if expected_facts else 1.0
        has_prohibited = len(found_prohibited) > 0

        # Pass if all facts present and no prohibited content
        passed = fact_coverage == 1.0 and not has_prohibited

        return {
            "query": query,
            "answer": answer,
            "passed": passed,
            "fact_coverage": fact_coverage,
            "found_facts": found_facts,
            "missing_facts": missing_facts,
            "found_prohibited": found_prohibited,
            "has_prohibited": has_prohibited
        }

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        answers: List[str]
    ) -> Dict[str, Any]:
        """Evaluate multiple answers.

        Args:
            test_cases: List of test case dicts with query, expected_facts, must_not_contain
            answers: List of generated answers

        Returns:
            Aggregate metrics and individual results
        """
        individual_results = []

        for test_case, answer in zip(test_cases, answers):
            result = self.evaluate_single_answer(
                query=test_case["query"],
                answer=answer,
                expected_facts=test_case.get("expected_facts", []),
                must_not_contain=test_case.get("must_not_contain", [])
            )
            result["test_id"] = test_case.get("id", "unknown")
            result["difficulty"] = test_case.get("difficulty", "unknown")
            individual_results.append(result)

        # Calculate aggregate metrics
        pass_rate = sum(r["passed"] for r in individual_results) / len(individual_results) if individual_results else 0
        avg_fact_coverage = sum(r["fact_coverage"] for r in individual_results) / len(individual_results) if individual_results else 0
        prohibited_rate = sum(r["has_prohibited"] for r in individual_results) / len(individual_results) if individual_results else 0

        failures = [r for r in individual_results if not r["passed"]]

        return {
            "aggregate": {
                "pass_rate": pass_rate,
                "avg_fact_coverage": avg_fact_coverage,
                "prohibited_content_rate": prohibited_rate
            },
            "individual": individual_results,
            "summary": {
                "total_queries": len(individual_results),
                "passed": len(individual_results) - len(failures),
                "failed": len(failures),
                "failure_details": [
                    {
                        "test_id": f["test_id"],
                        "query": f["query"],
                        "fact_coverage": f["fact_coverage"],
                        "missing_facts": f["missing_facts"],
                        "found_prohibited": f["found_prohibited"]
                    }
                    for f in failures
                ]
            }
        }
