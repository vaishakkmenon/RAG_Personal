#!/usr/bin/env python3
"""
Two-Phase Test Runner

Phase 1: Collect all RAG answers (uses 1B model only)
Phase 2: Validate with batch validator (uses 3B model once)

Usage:
    # Phase 1: Collect answers
    python tests/run_tests_batch.py --phase collect

    # Phase 2: Validate answers
    python tests/run_tests_batch.py --phase validate

    # Or do both in sequence:
    python tests/run_tests_batch.py --phase both
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.batch_validator import BatchValidator


class TwoPhaseTestRunner:
    """Test runner that collects answers then validates in batch"""

    def __init__(self, api_url: str, api_key: str, verbose: bool = False):
        if not api_url.endswith("/"):
            api_url += "/"

        self.base_url = api_url
        self.api_key = api_key
        self.verbose = verbose
        self.headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    def load_tests(self, filepath: str) -> Dict[str, Any]:
        """Load test suite from JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def make_request(self, question: str) -> Dict[str, Any]:
        """Make API request to RAG system"""
        url = f"{self.base_url}chat"
        payload = {"question": question}

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
            return {
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time,
            }

    def collect_answers(
        self,
        test_suite: Dict[str, Any],
        output_file: str = "test_answers.json"
    ) -> List[Dict[str, Any]]:
        """Phase 1: Collect all answers from RAG system"""

        test_cases = test_suite.get('test_cases') or test_suite.get('tests', [])

        print(f"{'='*80}")
        print(f"PHASE 1: COLLECTING ANSWERS")
        print(f"{'='*80}")
        print(f"Running {len(test_cases)} tests...")
        print(f"This will use the 1B model for all queries\n")

        results = []

        for i, test_case in enumerate(test_cases, 1):
            test_id = test_case['test_id']
            question = test_case['question']

            if self.verbose:
                print(f"[{i}/{len(test_cases)}] {test_id}: {question}")
            else:
                print(f"[{i}/{len(test_cases)}] {test_id}", end=' ')

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
                    "sources": response["data"].get("sources", [])[:3]  # First 3 sources
                }

            results.append(result)
            time.sleep(0.1)  # Small delay between requests

        # Save collected answers
        output = {
            "timestamp": datetime.now().isoformat(),
            "phase": "collection",
            "total_tests": len(results),
            "results": results
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"PHASE 1 COMPLETE")
        print(f"{'='*80}")
        print(f"Collected {len(results)} answers")
        print(f"Saved to: {output_file}\n")

        return results


def main():
    parser = argparse.ArgumentParser(description="Two-Phase RAG Test Runner")
    parser.add_argument(
        "--phase",
        choices=["collect", "validate", "both"],
        default="both",
        help="Which phase to run"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API base URL"
    )
    parser.add_argument(
        "--api-key",
        default="dev-key-1",
        help="API key for authentication"
    )
    parser.add_argument(
        "--tests",
        default="tests/test_suite.json",
        help="Path to test cases JSON file"
    )
    parser.add_argument(
        "--answers-file",
        default="test_answers.json",
        help="File to save/load answers"
    )
    parser.add_argument(
        "--report-file",
        default="test_validation_report.json",
        help="File to save validation report"
    )
    parser.add_argument(
        "--validation-model",
        default="llama3.2:3b-instruct-q4_K_M",
        help="Model for validation (3B recommended)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    # Phase 1: Collect answers
    if args.phase in ["collect", "both"]:
        runner = TwoPhaseTestRunner(
            api_url=args.api_url,
            api_key=args.api_key,
            verbose=args.verbose
        )

        try:
            test_suite = runner.load_tests(args.tests)
        except FileNotFoundError:
            print(f"Error: Test file not found: {args.tests}")
            sys.exit(1)

        runner.collect_answers(test_suite, args.answers_file)

    # Phase 2: Validate
    if args.phase in ["validate", "both"]:
        if args.phase == "validate":
            # Check if answers file exists
            try:
                with open(args.answers_file, 'r') as f:
                    json.load(f)
            except FileNotFoundError:
                print(f"Error: Answers file not found: {args.answers_file}")
                print(f"Run with --phase collect first")
                sys.exit(1)

        print(f"\n{'='*80}")
        print(f"PHASE 2: VALIDATING ANSWERS")
        print(f"{'='*80}")
        print(f"Using validation model: {args.validation_model}")
        print(f"This will load the 3B model ONCE for all validations\n")

        validator = BatchValidator(validation_model=args.validation_model)
        validator.validate_batch(args.answers_file, args.report_file)


if __name__ == "__main__":
    main()
