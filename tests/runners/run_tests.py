#!/usr/bin/env python3
"""
Two-Phase Test Runner

Phase 1: Collect all RAG answers (uses 1B model only)
Phase 2: Validate with batch validator (uses 3B model once)

Usage:
    # Run all tests
    python tests/run_tests_batch.py --phase both

    # Run only specific category
    python tests/run_tests_batch.py --phase both --category skills

    # Run specific test IDs
    python tests/run_tests_batch.py --phase both --test-id skills_001 --test-id skills_002

    # Collect answers only (no validation)
    python tests/run_tests_batch.py --phase collect --category projects

    # Validate existing answers
    python tests/run_tests_batch.py --phase validate

Examples:
    # Test only skills category
    python tests/run_tests_batch.py --category skills

    # Test projects and ambiguity categories
    python tests/run_tests_batch.py --test-id projects_001 --test-id ambiguity_control_001
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.validators.batch_validator import BatchValidator


class TwoPhaseTestRunner:
    """Test runner that collects answers then validates in batch"""

    def __init__(self, api_url: str, api_key: str, verbose: bool = False):
        if not api_url.endswith("/"):
            api_url += "/"

        self.base_url = api_url
        self.api_key = api_key
        self.verbose = verbose
        self.headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
        self.provider = self._detect_provider()

    def _detect_provider(self) -> str:
        """Detect which LLM provider the API is using"""
        try:
            response = requests.get(
                f"{self.base_url}health", headers=self.headers, timeout=5
            )
            response.raise_for_status()
            data = response.json()
            return data.get("provider", "unknown")
        except Exception as e:
            print(f"Warning: Could not detect provider from /health endpoint: {e}")
            return "unknown"

    def load_tests(self, filepath: str) -> Dict[str, Any]:
        """Load test suite from JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def make_request(self, question: str, max_retries: int = 2) -> Dict[str, Any]:
        """Make API request to RAG system with retry logic

        Args:
            question: Question to ask
            max_retries: Maximum number of retry attempts (default 2)

        Returns:
            Dict with success status, data/error, and response_time
        """
        url = f"{self.base_url}chat"
        payload = {"question": question}

        for attempt in range(max_retries + 1):
            start_time = time.time()
            try:
                response = requests.post(
                    url, json=payload, headers=self.headers, timeout=30
                )
                response_time = time.time() - start_time
                response.raise_for_status()

                return {
                    "success": True,
                    "data": response.json(),
                    "response_time": response_time,
                }
            except requests.exceptions.RequestException as e:
                response_time = time.time() - start_time

                # If this was the last attempt, return error
                if attempt == max_retries:
                    return {
                        "success": False,
                        "error": str(e),
                        "response_time": response_time,
                    }

                # Otherwise, wait a bit and retry
                if self.verbose:
                    print(
                        f"\n   Retry {attempt + 1}/{max_retries} after error: {str(e)[:50]}..."
                    )
                time.sleep(2)  # Wait 2 seconds before retry

    def collect_answers(
        self,
        test_suite: Dict[str, Any],
        output_file: str = "test_answers.json",
        delay: float = 5.0,
        filter_category: str = None,
        filter_test_ids: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """Phase 1: Collect all answers from RAG system

        Args:
            test_suite: Dictionary containing test cases
            output_file: File to save results
            delay: Delay between requests
            filter_category: Optional category to filter tests (e.g., 'skills', 'projects')
            filter_test_ids: Optional list of specific test IDs to run
        """

        test_cases = test_suite.get("test_cases") or test_suite.get("tests", [])

        # Apply filters if specified
        if filter_category:
            test_cases = [t for t in test_cases if t.get("category") == filter_category]
            print(f"Filtering by category: {filter_category}")

        if filter_test_ids:
            test_cases = [t for t in test_cases if t.get("test_id") in filter_test_ids]
            print(f"Filtering by test IDs: {', '.join(filter_test_ids)}")

        if not test_cases:
            print("No tests match the specified filters!")
            return []

        # Apply delay for all providers to prevent resource contention
        effective_delay = delay
        if self.provider == "ollama":
            delay_reason = "(Ollama local - delay helps prevent GPU/CPU overload)"
        elif self.provider == "groq":
            delay_reason = "(Groq cloud - delay for rate limiting)"
        else:
            delay_reason = "(Unknown provider - applying delay to be safe)"

        print(f"{'='*80}")
        print("PHASE 1: COLLECTING ANSWERS")
        print(f"{'='*80}")
        print(f"Running {len(test_cases)} tests...")
        print(f"Provider: {self.provider}")
        print(f"Delay between requests: {effective_delay}s {delay_reason}")
        estimated_time = (
            len(test_cases) * effective_delay / 60
            if effective_delay > 0
            else len(test_cases) * 2 / 60
        )
        print(f"Estimated time: {estimated_time:.1f} minutes\n")

        results = []

        for i, test_case in enumerate(test_cases, 1):
            test_id = test_case["test_id"]
            question = test_case["question"]

            if self.verbose:
                print(f"[{i}/{len(test_cases)}] {test_id}: {question}")
            else:
                print(f"[{i}/{len(test_cases)}] {test_id}", end=" ")

            # Make request
            response = self.make_request(question)

            if not response["success"]:
                print("X ERROR")
                result = {
                    "test_id": test_id,
                    "category": test_case["category"],
                    "question": question,
                    "error": response["error"],
                    "response_time": response["response_time"],
                    "expected_keywords": test_case.get("expected_keywords", []),
                    "expected_answer": test_case.get("expected_answer", ""),
                    "is_impossible": test_case.get("is_impossible", False),
                }
            else:
                print("OK" if not self.verbose else "OK Answered")
                result = {
                    "test_id": test_id,
                    "category": test_case["category"],
                    "question": question,
                    "answer": response["data"].get("answer", ""),
                    "grounded": response["data"].get("grounded", False),
                    "num_sources": len(response["data"].get("sources", [])),
                    "response_time": response["response_time"],
                    "expected_keywords": test_case.get("expected_keywords", []),
                    "expected_answer": test_case.get("expected_answer", ""),
                    "is_impossible": test_case.get("is_impossible", False),
                    "sources": response["data"].get("sources", [])[
                        :3
                    ],  # First 3 sources
                }

            results.append(result)

            # Delay between requests to avoid rate limits (skip for Ollama)
            if (
                i < len(test_cases) and effective_delay > 0
            ):  # Don't delay after last test
                time.sleep(effective_delay)

        # Retry failed tests
        errors = [r for r in results if "error" in r]
        if errors:
            print(f"\n{'='*80}")
            print(f"RETRYING {len(errors)} FAILED TESTS")
            print(f"{'='*80}")

            for i, error_result in enumerate(errors, 1):
                test_id = error_result["test_id"]
                question = error_result["question"]
                print(f"[{i}/{len(errors)}] Retrying {test_id}...", end=" ")

                # Retry the request
                response = self.make_request(question, max_retries=2)

                if response["success"]:
                    print("SUCCESS")
                    # Update the result in the results list
                    for idx, r in enumerate(results):
                        if r["test_id"] == test_id:
                            results[idx] = {
                                "test_id": test_id,
                                "category": error_result["category"],
                                "question": question,
                                "answer": response["data"].get("answer", ""),
                                "grounded": response["data"].get("grounded", False),
                                "num_sources": len(response["data"].get("sources", [])),
                                "response_time": response["response_time"],
                                "expected_keywords": error_result.get(
                                    "expected_keywords", []
                                ),
                                "expected_answer": error_result.get(
                                    "expected_answer", ""
                                ),
                                "is_impossible": error_result.get(
                                    "is_impossible", False
                                ),
                                "sources": response["data"].get("sources", [])[:3],
                            }
                            break
                else:
                    print(f"STILL FAILED: {response['error'][:50]}")

                # Small delay between retries
                if i < len(errors):
                    time.sleep(2)

            # Count final errors
            final_errors = [r for r in results if "error" in r]
            print(
                f"\nRetry complete: {len(errors) - len(final_errors)} recovered, {len(final_errors)} still failed"
            )

        # Save collected answers
        output = {
            "timestamp": datetime.now().isoformat(),
            "phase": "collection",
            "total_tests": len(results),
            "successful_tests": len([r for r in results if "answer" in r]),
            "failed_tests": len([r for r in results if "error" in r]),
            "results": results,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print("PHASE 1 COMPLETE")
        print(f"{'='*80}")
        print(f"Collected {len(results)} answers")
        print(f"Successful: {output['successful_tests']}")
        print(f"Failed: {output['failed_tests']}")
        print(f"Saved to: {output_file}\n")

        return results


def main():
    parser = argparse.ArgumentParser(description="Two-Phase RAG Test Runner")
    parser.add_argument(
        "--phase",
        choices=["collect", "validate", "both"],
        default="both",
        help="Which phase to run",
    )
    parser.add_argument(
        "--api-url", default="http://localhost:8000", help="API base URL"
    )
    parser.add_argument(
        "--api-key", default="dev-key-1", help="API key for authentication"
    )
    parser.add_argument(
        "--tests",
        default="tests/fixtures/test_suite.json",
        help="Path to test cases JSON file (e.g., tests/fixtures/test_suite.json, tests/fixtures/negative_inference_test.json)",
    )
    parser.add_argument(
        "--answers-file", default="test_answers.json", help="File to save/load answers"
    )
    parser.add_argument(
        "--report-file",
        default="test_validation_report.json",
        help="File to save validation report",
    )
    parser.add_argument(
        "--validation-provider",
        choices=["ollama", "groq"],
        default="ollama",
        help="LLM provider for validation (default: ollama to save Groq quota)",
    )
    parser.add_argument(
        "--validation-model",
        help="Model for validation (optional, uses provider defaults)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Delay between requests in seconds (applies to both collection and validation phases)",
    )
    parser.add_argument(
        "--category",
        help="Filter tests by category (e.g., 'skills', 'projects', 'transcript_gpa')",
    )
    parser.add_argument(
        "--test-id",
        action="append",
        dest="test_ids",
        help="Filter by specific test ID(s). Can be used multiple times (e.g., --test-id skills_001 --test-id skills_002)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear Redis cache before running tests to ensure fresh responses (no cache hits)",
    )

    args = parser.parse_args()

    # Clear Redis cache if requested
    if args.clear_cache:
        print(f"{'='*80}")
        print("CLEARING REDIS CACHE")
        print(f"{'='*80}")
        try:
            # Import Redis connection settings
            import os

            redis_url = os.getenv(
                "SESSION_REDIS_URL", "redis://:devpassword123@localhost:6379/0"
            )

            # Import required modules
            from app.storage.primary.redis_store import (
                RedisSessionStore,
                REDIS_AVAILABLE,
            )

            if not REDIS_AVAILABLE:
                print("⚠️  Redis not available - skipping cache clear")
            else:
                store = RedisSessionStore(redis_url)
                if store.clear_cache():
                    print("✅ Redis cache cleared successfully")
                else:
                    print("❌ Failed to clear Redis cache")
        except Exception as e:
            print(f"⚠️  Could not clear cache: {e}")
            print("Continuing with tests...")
        print()

    # Phase 1: Collect answers
    if args.phase in ["collect", "both"]:
        runner = TwoPhaseTestRunner(
            api_url=args.api_url, api_key=args.api_key, verbose=args.verbose
        )

        try:
            test_suite = runner.load_tests(args.tests)
        except FileNotFoundError:
            print(f"Error: Test file not found: {args.tests}")
            sys.exit(1)

        runner.collect_answers(
            test_suite,
            args.answers_file,
            delay=args.delay,
            filter_category=args.category,
            filter_test_ids=args.test_ids,
        )

    # Phase 2: Validate
    if args.phase in ["validate", "both"]:
        if args.phase == "validate":
            # Check if answers file exists
            try:
                with open(args.answers_file, "r") as f:
                    json.load(f)
            except FileNotFoundError:
                print(f"Error: Answers file not found: {args.answers_file}")
                print("Run with --phase collect first")
                sys.exit(1)

        print(f"\n{'='*80}")
        print("PHASE 2: VALIDATING ANSWERS")
        print(f"{'='*80}")
        print(f"Using validation provider: {args.validation_provider}")
        if args.validation_model:
            print(f"Using validation model: {args.validation_model}")
        else:
            default_model = (
                "llama-3.1-8b-instant"
                if args.validation_provider == "groq"
                else "llama3.1:8b"
            )
            print(f"Using default model: {default_model}")
        print("This will validate all answers using a single model instance\n")

        validator = BatchValidator(
            provider=args.validation_provider,
            validation_model=args.validation_model,
            delay=args.delay,
        )
        validator.validate_batch(args.answers_file, args.report_file)


if __name__ == "__main__":
    main()
