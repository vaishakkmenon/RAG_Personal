"""
Run comprehensive evaluation of the RAG system.

Usage:
    python scripts/run_evaluation.py --baseline
    python scripts/run_evaluation.py --compare-to baseline_results.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.chat_service import ChatService
from app.evaluation import RetrievalEvaluator, AnswerEvaluator
from app.models import ChatRequest
from app.retrieval import search

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ChromaDB connection for content-based evaluation
_chroma_client = None
_chroma_collection = None


def get_chroma_collection():
    """Get ChromaDB collection (lazy initialization)."""
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        _chroma_client = chromadb.PersistentClient(
            path="./data/chroma",
            settings=Settings(allow_reset=False)
        )
        _chroma_collection = _chroma_client.get_collection("personal_rag")
    return _chroma_collection


def get_chunk_content(chunk_ids: List[str]) -> Dict[str, str]:
    """Fetch chunk content from ChromaDB."""
    if not chunk_ids:
        return {}
    collection = get_chroma_collection()
    result = collection.get(ids=chunk_ids, include=["documents"])
    content_map = {}
    for i, chunk_id in enumerate(result["ids"]):
        content_map[chunk_id] = result["documents"][i] if result["documents"] else ""
    return content_map


def load_test_queries(filepath: str) -> Dict[str, List[Dict]]:
    """Load test queries from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def warmup_models():
    """Pre-load models to avoid delays during evaluation."""
    logger.info("Warming up models...")
    
    # Run a quick dummy search to load cross-encoder and embedding model
    try:
        from app.retrieval import search
        _ = search("warmup query", k=1)
        logger.info("Models warmed up successfully")
    except Exception as e:
        logger.warning(f"Warmup failed (non-critical): {e}")


def run_retrieval_evaluation(test_cases: List[Dict], k: int = 5) -> Dict[str, Any]:
    """Run retrieval evaluation on test cases.

    Args:
        test_cases: List of retrieval test cases
        k: Number of chunks to retrieve

    Returns:
        Evaluation results with both chunk ID-based and content-based metrics
    """
    logger.info(f"Running retrieval evaluation on {len(test_cases)} test cases...")

    retrieval_results = []
    content_results = []

    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        logger.info(f"[{i}/{len(test_cases)}] Retrieving for: {query}")

        try:
            # Run retrieval
            chunks = search(query=query, k=k)
            chunk_ids = [c["id"] for c in chunks]

            retrieval_results.append({
                "test_id": test_case["id"],
                "query": query,
                "chunk_ids": chunk_ids,
                "chunks": chunks  # Keep for analysis
            })

            # Content-based evaluation (if required_content specified)
            if "required_content" in test_case:
                content_map = get_chunk_content(chunk_ids)
                combined_content = " ".join(content_map.values()).lower()
                
                required = test_case["required_content"]
                found = [item for item in required if item.lower() in combined_content]
                missing = [item for item in required if item.lower() not in combined_content]
                coverage = len(found) / len(required) if required else 1.0
                
                content_results.append({
                    "test_id": test_case["id"],
                    "query": query,
                    "content_coverage": coverage,
                    "content_pass": coverage == 1.0,
                    "found_content": found,
                    "missing_content": missing
                })

        except Exception as e:
            logger.error(f"Error retrieving for '{query}': {e}")
            retrieval_results.append({
                "test_id": test_case["id"],
                "query": query,
                "chunk_ids": [],
                "error": str(e)
            })

    # Evaluate using RetrievalEvaluator (chunk ID-based)
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5])
    evaluation = evaluator.evaluate_batch(test_cases, retrieval_results)

    # Add raw retrieval results for debugging
    evaluation["raw_retrievals"] = retrieval_results

    # Add content-based metrics if available
    if content_results:
        content_pass_rate = sum(1 for r in content_results if r["content_pass"]) / len(content_results)
        avg_coverage = sum(r["content_coverage"] for r in content_results) / len(content_results)
        evaluation["content_evaluation"] = {
            "total_tests": len(content_results),
            "content_pass_rate": content_pass_rate,
            "avg_content_coverage": avg_coverage,
            "individual": content_results
        }

    return evaluation


def run_answer_evaluation(test_cases: List[Dict], delay_seconds: float = 15.0, max_retries: int = 3) -> Dict[str, Any]:
    """Run answer quality evaluation on test cases.

    Args:
        test_cases: List of answer quality test cases
        delay_seconds: Delay between LLM calls to avoid rate limiting
        max_retries: Maximum number of retries for failed LLM calls

    Returns:
        Evaluation results
    """
    import time
    
    logger.info(f"Running answer evaluation on {len(test_cases)} test cases...")
    logger.info(f"Using {delay_seconds}s delay between calls, max {max_retries} retries per call")

    chat_service = ChatService()
    answers = []

    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        logger.info(f"[{i}/{len(test_cases)}] Generating answer for: {query}")

        answer = ""
        for attempt in range(max_retries):
            try:
                # Generate answer
                request = ChatRequest(question=query)
                response = chat_service.handle_chat(request)
                answer = response.answer
                
                # Success - break retry loop
                if answer:
                    break
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed for '{query}': {e}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff: 5s, 10s, 20s
                    backoff = 5 * (2 ** attempt)
                    logger.info(f"Retrying in {backoff}s...")
                    time.sleep(backoff)
        
        if not answer:
            logger.warning(f"All {max_retries} attempts failed for: {query}")
        
        answers.append(answer)
        
        # Rate limiting delay (skip after last request)
        if i < len(test_cases):
            time.sleep(delay_seconds)

    # Evaluate using AnswerEvaluator
    evaluator = AnswerEvaluator()
    evaluation = evaluator.evaluate_batch(test_cases, answers)

    return evaluation


def print_retrieval_report(results: Dict[str, Any]):
    """Print human-readable retrieval evaluation report."""
    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION REPORT")
    print("=" * 80)

    aggregate = results["aggregate"]
    summary = results["summary"]

    print(f"\nOverall Metrics:")
    print(f"  Total Queries: {summary['total_queries']}")
    print(f"  Pass Rate: {summary['pass_rate']:.1%} ({summary['total_queries'] - summary['failures']}/{summary['total_queries']})")
    print(f"  Found Any: {aggregate['found_any_rate']:.1%}")
    print(f"\n  Recall@1:  {aggregate.get('recall@1', 0):.3f}")
    print(f"  Recall@3:  {aggregate.get('recall@3', 0):.3f}")
    print(f"  Recall@5:  {aggregate.get('recall@5', 0):.3f}")
    print(f"  MRR:       {aggregate.get('mrr', 0):.3f}")
    print(f"  NDCG@5:    {aggregate.get('ndcg@5', 0):.3f}")

    # By difficulty
    print(f"\nBy Difficulty:")
    for diff, metrics in summary["by_difficulty"].items():
        print(f"  {diff.capitalize()}:")
        print(f"    Count: {metrics['count']}")
        print(f"    Recall@5: {metrics['avg_recall@5']:.3f}")
        print(f"    MRR: {metrics['avg_mrr']:.3f}")

    # Failures
    if summary["failures"] > 0:
        print(f"\nChunk ID Failures ({summary['failures']}):")
        for failure in summary["failure_details"][:10]:  # Show first 10
            print(f"  [{failure['test_id']}] {failure['query']}")
            print(f"    Recall@5: {failure['recall@5']:.2f}, Found Any: {failure['found_any']}")

        if summary["failures"] > 10:
            print(f"  ... and {summary['failures'] - 10} more")

    # Content-based evaluation (if available)
    if "content_evaluation" in results:
        content_eval = results["content_evaluation"]
        print(f"\nContent-Based Evaluation:")
        print(f"  Tests with required_content: {content_eval['total_tests']}")
        print(f"  Content Pass Rate: {content_eval['content_pass_rate']:.1%}")
        print(f"  Avg Content Coverage: {content_eval['avg_content_coverage']:.1%}")
        
        content_failures = [r for r in content_eval["individual"] if not r["content_pass"]]
        if content_failures:
            print(f"\n  Content Failures ({len(content_failures)}):")
            for f in content_failures[:5]:
                print(f"    [{f['test_id']}] Missing: {f['missing_content']}")


def print_answer_report(results: Dict[str, Any]):
    """Print human-readable answer evaluation report."""
    print("\n" + "=" * 80)
    print("ANSWER QUALITY EVALUATION REPORT")
    print("=" * 80)

    aggregate = results["aggregate"]
    summary = results["summary"]

    print(f"\nOverall Metrics:")
    print(f"  Total Queries: {summary['total_queries']}")
    print(f"  Pass Rate: {aggregate['pass_rate']:.1%} ({summary['passed']}/{summary['total_queries']})")
    print(f"  Avg Fact Coverage: {aggregate['avg_fact_coverage']:.1%}")
    print(f"  Prohibited Content Rate: {aggregate['prohibited_content_rate']:.1%}")

    # Failures
    if summary["failed"] > 0:
        print(f"\nFailures ({summary['failed']}):")
        for failure in summary["failure_details"][:10]:
            print(f"  [{failure['test_id']}] {failure['query']}")
            print(f"    Fact Coverage: {failure['fact_coverage']:.1%}")
            if failure["missing_facts"]:
                print(f"    Missing: {', '.join(failure['missing_facts'])}")
            if failure["found_prohibited"]:
                print(f"    Prohibited: {', '.join(failure['found_prohibited'])}")

        if summary["failed"] > 10:
            print(f"  ... and {summary['failed'] - 10} more")


def save_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")


def compare_results(current: Dict, baseline: Dict):
    """Compare current results to baseline."""
    print("\n" + "=" * 80)
    print("COMPARISON TO BASELINE")
    print("=" * 80)

    current_agg = current["retrieval"]["aggregate"]
    baseline_agg = baseline["retrieval"]["aggregate"]

    print(f"\nRetrieval Metrics:")
    metrics = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"]
    for metric in metrics:
        current_val = current_agg.get(metric, 0)
        baseline_val = baseline_agg.get(metric, 0)
        diff = current_val - baseline_val
        emoji = "[OK]" if diff >= 0 else "[WARNING]"
        print(f"  {metric:12s}: {current_val:.3f} (baseline: {baseline_val:.3f}) {emoji} {diff:+.3f}")

    current_summary = current["retrieval"]["summary"]
    baseline_summary = baseline["retrieval"]["summary"]

    pass_rate_diff = current_summary["pass_rate"] - baseline_summary["pass_rate"]
    emoji = "[OK]" if pass_rate_diff >= 0 else "[WARNING]"
    print(f"\n  Pass Rate:     {current_summary['pass_rate']:.1%} (baseline: {baseline_summary['pass_rate']:.1%}) {emoji} {pass_rate_diff:+.1%}")


def main():
    parser = argparse.ArgumentParser(description="Run RAG system evaluation")
    parser.add_argument("--baseline", action="store_true", help="Save results as baseline")
    parser.add_argument("--compare-to", type=str, help="Compare to baseline results file")
    parser.add_argument("--retrieval-only", action="store_true", help="Run only retrieval evaluation")
    parser.add_argument("--answer-only", action="store_true", help="Run only answer evaluation")
    parser.add_argument("--test-file", type=str, default="data/eval/test_queries.json", help="Test queries file")
    parser.add_argument("--test-id", type=str, help="Run only a specific test by ID (e.g., answer-002)")
    parser.add_argument("--output", type=str, help="Output file for results")

    args = parser.parse_args()

    # Load test queries
    logger.info(f"Loading test queries from {args.test_file}")
    test_data = load_test_queries(args.test_file)

    results = {
        "timestamp": datetime.now().isoformat(),
        "test_file": args.test_file
    }

    # Warmup models to avoid delays during evaluation
    warmup_models()

    # Run retrieval evaluation
    if not args.answer_only:
        retrieval_test_cases = test_data.get("retrieval_tests", [])
        if retrieval_test_cases:
            results["retrieval"] = run_retrieval_evaluation(retrieval_test_cases)
            print_retrieval_report(results["retrieval"])

    # Run answer evaluation
    if not args.retrieval_only:
        answer_test_cases = test_data.get("answer_quality_tests", [])
        # Filter by test ID if specified
        if args.test_id:
            answer_test_cases = [t for t in answer_test_cases if t.get("id") == args.test_id]
            if not answer_test_cases:
                logger.warning(f"No answer test found with ID: {args.test_id}")
        if answer_test_cases:
            results["answer"] = run_answer_evaluation(answer_test_cases)
            print_answer_report(results["answer"])

    # Save results
    if args.baseline:
        output_path = args.output or f"data/eval/baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(results, output_path)
        print(f"\n✅ Baseline saved to {output_path}")

    elif args.output:
        save_results(results, args.output)

    # Compare to baseline if requested
    if args.compare_to:
        with open(args.compare_to, 'r') as f:
            baseline = json.load(f)
        compare_results(results, baseline)

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
