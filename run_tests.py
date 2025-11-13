#!/usr/bin/env python3
"""
Personal RAG System Test Runner (Refactored Version)
====================================================

Comprehensive test suite for the modular Personal RAG system.

Usage:
    python run_tests.py                           # Run all tests
    python run_tests.py --category transcript_gpa  # Run specific category
    python run_tests.py --test-id work_001         # Run specific test
    python run_tests.py --impossible-only          # Run only impossible questions
    python run_tests.py --verbose                  # Show detailed output
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests


@dataclass
class TestResult:
    """Results from a single test execution"""

    test_id: str
    category: str
    question: str
    passed: bool
    answer: str
    grounded: bool
    num_sources: int
    response_time: float
    failure_reason: Optional[str] = None
    expected_keywords_found: List[str] = field(default_factory=list)
    expected_keywords_missing: List[str] = field(default_factory=list)
    sources: List[Dict] = field(default_factory=list)
    ambiguity_detected: Optional[bool] = None
    ambiguity_score: Optional[float] = None
    clarification_requested: Optional[bool] = None


class RAGTestRunner:
    """Test runner for Personal RAG system"""

    def __init__(self, api_url: str, api_key: str, verbose: bool = False):
        # Normalize API URL
        if not api_url.endswith("/"):
            api_url += "/"

        self.base_url = api_url
        self.api_key = api_key
        self.verbose = verbose
        self.headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
        self.results: List[TestResult] = []

    def load_tests(self, filepath: str) -> Dict[str, Any]:
        """Load test suite from JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def make_request(
        self, question: str, doc_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make API request to RAG system

        Args:
            question: The question to ask
            doc_type: Optional document type filter

        Returns:
            Response dict with success status and data/error
        """
        # Build URL - the /chat endpoint is at root level
        url = f"{self.base_url}chat"

        # Build query params
        params = {}
        if doc_type:
            params["doc_type"] = doc_type

        # Build request payload
        payload = {
            "question": question,
        }

        start_time = time.time()
        try:
            response = requests.post(
                url, json=payload, params=params, headers=self.headers, timeout=30
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
            return {"success": False, "error": str(e), "response_time": response_time}

    def check_keywords(
        self, answer: str, expected_keywords: List[str]
    ) -> tuple[List[str], List[str]]:
        """Check which expected keywords are present in the answer"""
        answer_lower = answer.lower()
        found = []
        missing = []

        for keyword in expected_keywords:
            if keyword.lower() in answer_lower:
                found.append(keyword)
            else:
                missing.append(keyword)

        return found, missing

    def validate_test(
        self, test_case: Dict[str, Any], response: Dict[str, Any]
    ) -> TestResult:
        """Validate a single test case against its response"""
        test_id = test_case["test_id"]
        category = test_case["category"]
        question = test_case["question"]
        is_impossible = test_case["is_impossible"]
        expected_keywords = test_case["expected_keywords"]
        min_sources = test_case["min_sources"]
        grounded_expected = test_case["grounded_expected"]

        # Extract response data
        answer = response["data"].get("answer", "")
        grounded = response["data"].get("grounded", False)
        sources = response["data"].get("sources", [])
        ambiguity_meta = response["data"].get("ambiguity", {})
        ambiguity_detected = ambiguity_meta.get("is_ambiguous")
        ambiguity_score = ambiguity_meta.get("score")
        clarification_requested = ambiguity_meta.get("clarification_requested", False)
        num_sources = len(sources)
        response_time = response["response_time"]

        # Check keywords
        found_keywords, missing_keywords = self.check_keywords(
            answer, expected_keywords
        )

        # Determine pass/fail
        passed = True
        failure_reasons = []

        # Rule 1: Impossible questions should not be grounded
        if is_impossible and grounded:
            passed = False
            failure_reasons.append(
                "Expected ungrounded response for impossible question, got grounded=True"
            )

        # Rule 2: Possible questions should be grounded (unless they're negative assertions)
        if not is_impossible and grounded_expected and not grounded:
            passed = False
            failure_reasons.append("Expected grounded response, got grounded=False")

        # Rule 3: Check minimum sources requirement
        if num_sources < min_sources:
            passed = False
            failure_reasons.append(
                f"Expected at least {min_sources} sources, got {num_sources}"
            )

        # Rule 4: Check for expected keywords (at least 60% should be present)
        if expected_keywords:
            keyword_match_rate = len(found_keywords) / len(expected_keywords)
            if keyword_match_rate < 0.6:
                passed = False
                failure_reasons.append(
                    f"Only {len(found_keywords)}/{len(expected_keywords)} expected keywords found. "
                    f"Missing: {', '.join(missing_keywords)}"
                )

        # Rule 5: Impossible questions should indicate uncertainty
        if is_impossible:
            uncertainty_phrases = [
                "don't know",
                "not mentioned",
                "no information",
                "cannot find",
                "not available",
                "not sure",
            ]
            if not any(phrase in answer.lower() for phrase in uncertainty_phrases):
                passed = False
                failure_reasons.append(
                    "Impossible question did not express uncertainty"
                )

        # Validate ambiguity expectations
        expected_behavior = test_case.get("expected_behavior")
        if expected_behavior:
            should_detect = expected_behavior.get("should_detect_ambiguity")
            min_score = expected_behavior.get("min_ambiguity_score")
            expected_resp = expected_behavior.get("expected_response", {})
            expected_clarification = expected_resp.get("should_ask_clarification")
            max_sources = expected_resp.get("max_sources")
            allowed_clarification_keywords = expected_resp.get("allowed_keywords", [])

            if should_detect and not ambiguity_detected:
                passed = False
                failure_reasons.append("Expected ambiguity to be detected")
            if should_detect and min_score is not None and ambiguity_score is not None:
                if ambiguity_score < min_score:
                    passed = False
                    failure_reasons.append(
                        f"Ambiguity score {ambiguity_score:.2f} below expected minimum {min_score:.2f}"
                    )
            if should_detect is False and ambiguity_detected:
                passed = False
                failure_reasons.append("Did not expect ambiguity detection")
            if expected_clarification and not clarification_requested:
                passed = False
                failure_reasons.append(
                    "Expected clarification prompt for ambiguous query"
                )
            if expected_clarification is False and clarification_requested:
                passed = False
                failure_reasons.append("Did not expect clarification prompt")
            if max_sources is not None and num_sources > max_sources:
                passed = False
                failure_reasons.append(
                    f"Expected at most {max_sources} sources for ambiguous query, got {num_sources}"
                )
            if allowed_clarification_keywords:
                answer_lower = answer.lower()
                if not any(
                    keyword.lower() in answer_lower
                    for keyword in allowed_clarification_keywords
                ):
                    passed = False
                    failure_reasons.append(
                        "Clarification response missing expected guidance keywords"
                    )

        failure_reason = "; ".join(failure_reasons) if failure_reasons else None

        return TestResult(
            test_id=test_id,
            category=category,
            question=question,
            passed=passed,
            answer=answer,
            grounded=grounded,
            num_sources=num_sources,
            response_time=response_time,
            failure_reason=failure_reason,
            expected_keywords_found=found_keywords,
            expected_keywords_missing=missing_keywords,
            sources=sources,
            ambiguity_detected=ambiguity_detected,
            ambiguity_score=ambiguity_score,
            clarification_requested=clarification_requested,
        )

    def run_test(self, test_case: Dict[str, Any]) -> TestResult:
        """Run a single test case"""
        question = test_case["question"]

        # Let the router intelligently decide doc_type
        # Don't force it unless explicitly needed for testing

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Running Test: {test_case['test_id']}")
            print(f"Category: {test_case['category']}")
            print(f"Question: {question}")
            print(f"{'='*80}")

        # Make request (let router decide doc_type)
        response = self.make_request(question, doc_type=None)

        if not response["success"]:
            return TestResult(
                test_id=test_case["test_id"],
                category=test_case["category"],
                question=question,
                passed=False,
                answer="",
                grounded=False,
                num_sources=0,
                response_time=response["response_time"],
                failure_reason=f"API request failed: {response['error']}",
            )

        # Validate response
        result = self.validate_test(test_case, response)

        if self.verbose:
            print(f"\nAnswer: {result.answer}")
            print(f"Grounded: {result.grounded}")
            print(f"Sources: {result.num_sources}")
            print(f"Response time: {result.response_time:.2f}s")
            if result.passed:
                print("✅ PASSED")
            else:
                print(f"❌ FAILED: {result.failure_reason}")

        return result

    def run_all_tests(
        self,
        test_suite: Dict[str, Any],
        category_filter: Optional[str] = None,
        test_id_filter: Optional[str] = None,
        impossible_only: bool = False,
    ) -> List[TestResult]:
        """Run all tests with optional filters"""
        test_cases = test_suite["test_cases"]

        # Apply filters
        if category_filter:
            test_cases = [t for t in test_cases if t["category"] == category_filter]
        if test_id_filter:
            test_cases = [t for t in test_cases if t["test_id"] == test_id_filter]
        if impossible_only:
            test_cases = [t for t in test_cases if t["is_impossible"]]

        print(f"\nRunning {len(test_cases)} tests...\n")

        for i, test_case in enumerate(test_cases, 1):
            print(
                f"[{i}/{len(test_cases)}] {test_case['test_id']}: ", end="", flush=True
            )
            result = self.run_test(test_case)
            self.results.append(result)

            if not self.verbose:
                if result.passed:
                    print("✅ PASSED")
                else:
                    print("❌ FAILED")

        return self.results

    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        if not self.results:
            return "No tests run"

        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        pass_rate = (passed / len(self.results)) * 100
        avg_response_time = sum(r.response_time for r in self.results) / len(
            self.results
        )

        # Group by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = {"passed": 0, "total": 0}
            categories[result.category]["total"] += 1
            if result.passed:
                categories[result.category]["passed"] += 1

        # Build report
        report = []
        report.append("=" * 80)
        report.append("PERSONAL RAG SYSTEM - TEST REPORT (REFACTORED)")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Tests: {len(self.results)}")
        report.append(f"Passed: {passed} ({pass_rate:.1f}%)")
        report.append(f"Failed: {failed}")
        report.append(f"Average Response Time: {avg_response_time:.2f}s")
        report.append("")
        report.append("RESULTS BY CATEGORY")
        report.append("-" * 80)

        for category, stats in sorted(categories.items()):
            cat_pass_rate = (stats["passed"] / stats["total"]) * 100
            report.append(
                f"{category:30} | {stats['passed']:2}/{stats['total']:2} passed "
                f"({cat_pass_rate:5.1f}%)"
            )

        # Failed tests detail
        failed_tests = [r for r in self.results if not r.passed]
        if failed_tests:
            report.append("")
            report.append("FAILED TESTS")
            report.append("-" * 80)
            for result in failed_tests:
                report.append("")
                report.append(f"Test ID: {result.test_id}")
                report.append(f"Category: {result.category}")
                report.append(f"Question: {result.question}")
                report.append(f"Answer: {result.answer[:200]}...")
                report.append(f"Failure Reason: {result.failure_reason}")
                report.append(
                    f"Grounded: {result.grounded} | Sources: {result.num_sources}"
                )
                if result.expected_keywords_missing:
                    report.append(
                        f"Missing Keywords: {', '.join(result.expected_keywords_missing)}"
                    )
                if result.ambiguity_detected is not None:
                    report.append(
                        f"Ambiguity: detected={result.ambiguity_detected} score={result.ambiguity_score} "
                        f"clarification={result.clarification_requested}"
                    )

        # Ambiguity summary
        ambiguous_results = [r for r in self.results if r.ambiguity_detected]
        if ambiguous_results:
            report.append("")
            report.append("AMBIGUITY SUMMARY")
            report.append("-" * 80)
            avg_score = sum(r.ambiguity_score or 0 for r in ambiguous_results) / len(
                ambiguous_results
            )
            clarifications = sum(
                1 for r in ambiguous_results if r.clarification_requested
            )
            report.append(f"Ambiguous queries detected: {len(ambiguous_results)}")
            report.append(f"Average ambiguity score: {avg_score:.2f}")
            report.append(f"Clarification prompts issued: {clarifications}")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def save_results(self, filepath: str):
        """Save results to JSON file"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
                "pass_rate": (
                    sum(1 for r in self.results if r.passed) / len(self.results)
                )
                * 100,
            },
            "results": [
                {
                    "test_id": r.test_id,
                    "category": r.category,
                    "question": r.question,
                    "passed": r.passed,
                    "answer": r.answer,
                    "grounded": r.grounded,
                    "num_sources": r.num_sources,
                    "response_time": r.response_time,
                    "failure_reason": r.failure_reason,
                    "keywords_found": r.expected_keywords_found,
                    "keywords_missing": r.expected_keywords_missing,
                    "ambiguity_detected": r.ambiguity_detected,
                    "ambiguity_score": r.ambiguity_score,
                    "clarification_requested": r.clarification_requested,
                }
                for r in self.results
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Run tests against Personal RAG system"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the RAG API (e.g., http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key", type=str, required=True, help="API key for authentication"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="tests/test_suite.json",
        help="Path to test suite JSON file",
    )
    parser.add_argument(
        "--category", type=str, default=None, help="Run only tests in this category"
    )
    parser.add_argument(
        "--test-id", type=str, default=None, help="Run only test with this ID"
    )
    parser.add_argument(
        "--impossible-only", action="store_true", help="Run only impossible questions"
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument(
        "--output",
        type=str,
        default="test_results.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    # Initialize test runner
    test_runner = RAGTestRunner(args.api_url, args.api_key, args.verbose)

    if args.verbose:
        print(f"Using API endpoint: {test_runner.base_url}")
        print(f"Test file: {args.test_file}")

    # Load test suite
    try:
        test_suite = test_runner.load_tests(args.test_file)
        print(f"Loaded test suite: {test_suite['metadata']['description']}")
        print(f"Version: {test_suite['metadata']['version']}")
        print(f"Total tests available: {test_suite['metadata']['total_tests']}")
    except FileNotFoundError:
        print(f"Error: Test suite file '{args.test_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Invalid JSON in test suite file")
        sys.exit(1)

    # Run tests
    test_runner.run_all_tests(
        test_suite,
        category_filter=args.category,
        test_id_filter=args.test_id,
        impossible_only=args.impossible_only,
    )

    # Generate and print report
    report = test_runner.generate_report()
    print("\n" + report)

    # Save results
    test_runner.save_results(args.output)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
