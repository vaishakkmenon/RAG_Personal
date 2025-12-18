"""
Evaluation framework for RAG system.

Provides metrics and evaluation tools for:
- Retrieval quality (Recall@K, MRR, NDCG)
- Answer quality (fact checking, correctness)
"""

from app.evaluation.metrics import (
    calculate_recall_at_k,
    calculate_mrr,
    calculate_precision_at_k,
    calculate_ndcg_at_k,
)

from app.evaluation.evaluator import RetrievalEvaluator, AnswerEvaluator

__all__ = [
    "calculate_recall_at_k",
    "calculate_mrr",
    "calculate_precision_at_k",
    "calculate_ndcg_at_k",
    "RetrievalEvaluator",
    "AnswerEvaluator",
]
