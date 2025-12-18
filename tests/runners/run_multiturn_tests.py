#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Turn Conversation Test Runner

Tests conversation continuity, context-awareness, and session management.

Usage:
    # Run all multi-turn tests
    python tests/runners/run_multiturn_tests.py

    # Run specific conversation
    python tests/runners/run_multiturn_tests.py --conversation-id multiturn_001

    # Run specific category
    python tests/runners/run_multiturn_tests.py --category short_followup

    # Verbose output
    python tests/runners/run_multiturn_tests.py --verbose

Examples:
    python tests/runners/run_multiturn_tests.py --category pronoun_reference
    python tests/runners/run_multiturn_tests.py --conversation-id multiturn_002 --verbose
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class MultiTurnTestRunner:
    """Test runner for multi-turn conversations"""

    def __init__(self, api_url: str, api_key: str, verbose: bool = False):
        if not api_url.endswith("/"):
            api_url += "/"

        self.base_url = api_url
        self.api_key = api_key
        self.verbose = verbose
        self.headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    def load_conversations(self, filepath: str) -> Dict[str, Any]:
        """Load multi-turn test suite from JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def make_request(
        self, question: str, session_id: Optional[str] = None, max_retries: int = 2
    ) -> Dict[str, Any]:
        """Make API request to RAG system"""
        url = f"{self.base_url}chat"
        payload = {"question": question}
        if session_id:
            payload["session_id"] = session_id

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
                if attempt == max_retries:
                    return {
                        "success": False,
                        "error": str(e),
                        "response_time": response_time,
                    }
                if self.verbose:
                    print(f"  [WARN] Retry {attempt + 1}/{max_retries}...")
                time.sleep(1)

    def validate_turn(
        self, turn_result: Dict[str, Any], expected: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a single turn's response"""
        issues = []
        checks = {"grounded": False, "keywords": False}

        # Check grounding
        if "expected_grounded" in expected:
            actual = turn_result.get("grounded", False)
            expected_val = expected["expected_grounded"]
            checks["grounded"] = actual == expected_val
            if not checks["grounded"]:
                issues.append(f"Grounding: expected {expected_val}, got {actual}")

        # Check keywords
        if "expected_keywords" in expected:
            answer = turn_result.get("answer", "").lower()
            keywords = [kw.lower() for kw in expected["expected_keywords"]]
            found = [kw for kw in keywords if kw in answer]
            checks["keywords"] = len(found) > 0
            if not checks["keywords"]:
                issues.append(f"No expected keywords found: {keywords}")

        return {"passed": all(checks.values()), "checks": checks, "issues": issues}

    def run_conversation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Run a multi-turn conversation test"""
        conv_id = conversation["conversation_id"]
        category = conversation.get("category", "unknown")
        description = conversation.get("description", "")

        print(f"\n{'='*80}")
        print(f"Conversation: {conv_id}")
        print(f"Category: {category}")
        print(f"Description: {description}")
        print(f"{'='*80}")

        session_id = None
        turn_results = []
        all_passed = True

        for turn in conversation["turns"]:
            turn_num = turn["turn_number"]
            question = turn["question"]
            context_dep = turn.get("context_dependent", False)

            print(f"\n[Turn {turn_num}] User: {question}")
            if context_dep:
                print("  (context-dependent)")

            result = self.make_request(question, session_id=session_id)

            if not result["success"]:
                print(f"  [FAIL] {result['error']}")
                turn_results.append(
                    {
                        "turn_number": turn_num,
                        "success": False,
                        "error": result["error"],
                    }
                )
                all_passed = False
                continue

            data = result["data"]

            if session_id is None:
                session_id = data.get("session_id")
                print(f"  Session ID: {session_id}")

            validation = self.validate_turn(data, turn)

            answer = data.get("answer", "")
            answer_preview = answer[:150] + "..." if len(answer) > 150 else answer
            print(f"  Assistant: {answer_preview}")
            print(f"  Grounded: {data.get('grounded', False)}")
            print(f"  Response time: {result['response_time']:.2f}s")

            if validation["passed"]:
                print("  [PASS]")
            else:
                print("  [FAIL]")
                for issue in validation["issues"]:
                    print(f"    - {issue}")
                all_passed = False

            if self.verbose:
                print(f"  Full answer: {answer}")

            turn_results.append(
                {
                    "turn_number": turn_num,
                    "question": question,
                    "answer": answer,
                    "grounded": data.get("grounded", False),
                    "validation": validation,
                    "success": validation["passed"],
                }
            )

            time.sleep(0.5)

        passed = sum(1 for tr in turn_results if tr.get("success", False))
        total = len(turn_results)

        print(f"\n{'='*80}")
        print(f"Summary: {conv_id}")
        print(f"Turns: {passed}/{total} passed")
        print(f"Overall: {'[PASS]' if all_passed else '[FAIL]'}")
        print(f"{'='*80}")

        return {
            "conversation_id": conv_id,
            "category": category,
            "passed": all_passed,
            "turns": turn_results,
        }

    def run_tests(
        self,
        test_suite: Dict[str, Any],
        conversation_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run multi-turn conversation tests"""
        conversations = test_suite["test_conversations"]

        if conversation_id:
            conversations = [
                c for c in conversations if c["conversation_id"] == conversation_id
            ]
            if not conversations:
                print(f"Error: Conversation ID '{conversation_id}' not found")
                return {"error": "Conversation not found"}

        if category:
            conversations = [c for c in conversations if c.get("category") == category]
            if not conversations:
                print(f"Error: No conversations found for category '{category}'")
                return {"error": "Category not found"}

        print(f"\n{'='*80}")
        print("Multi-Turn Conversation Test Runner")
        print(f"{'='*80}")
        print(f"Total conversations: {len(conversations)}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        results = []
        for conversation in conversations:
            result = self.run_conversation(conversation)
            results.append(result)

        passed = sum(1 for r in results if r.get("passed", False))
        total = len(results)

        print(f"\n{'='*80}")
        print("OVERALL RESULTS")
        print(f"{'='*80}")
        print(f"Conversations: {passed}/{total} passed")
        print(f"Success rate: {passed/total*100:.1f}%")

        # Category breakdown
        categories = {}
        for r in results:
            cat = r.get("category", "unknown")
            if cat not in categories:
                categories[cat] = {"passed": 0, "total": 0}
            categories[cat]["total"] += 1
            if r.get("passed", False):
                categories[cat]["passed"] += 1

        print("\nCategory Breakdown:")
        for cat, stats in sorted(categories.items()):
            rate = stats["passed"] / stats["total"] * 100
            print(f"  {cat}: {stats['passed']}/{stats['total']} ({rate:.0f}%)")

        return {
            "total_conversations": total,
            "passed_conversations": passed,
            "success_rate": passed / total,
            "results": results,
        }


def main():
    parser = argparse.ArgumentParser(description="Run multi-turn conversation tests")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/",
        help="Base URL for the RAG API",
    )
    parser.add_argument(
        "--api-key", default="dev-key-1", help="API key for authentication"
    )
    parser.add_argument(
        "--test-file",
        default="tests/fixtures/multiturn_test_suite.json",
        help="Path to multi-turn test suite JSON file",
    )
    parser.add_argument("--conversation-id", help="Run only a specific conversation ID")
    parser.add_argument(
        "--category", help="Run only conversations in a specific category"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    test_file_path = Path(args.test_file)
    if not test_file_path.exists():
        print(f"Error: Test file not found: {test_file_path}")
        sys.exit(1)

    runner = MultiTurnTestRunner(
        api_url=args.api_url, api_key=args.api_key, verbose=args.verbose
    )

    test_suite = runner.load_conversations(str(test_file_path))

    results = runner.run_tests(
        test_suite, conversation_id=args.conversation_id, category=args.category
    )

    if "error" in results:
        sys.exit(1)
    elif results["success_rate"] < 1.0:
        print("\n[WARN] Some tests failed")
        sys.exit(1)
    else:
        print("\n[OK] All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
