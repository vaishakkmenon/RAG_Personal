#!/usr/bin/env python3
"""
Batch Validator - Two-phase testing approach

Phase 1: Collect all RAG answers using 1B model (fast)
Phase 2: Validate all answers at once using 3B model (one model load)

This avoids model swapping overhead while getting accurate validation.
"""

import json
import ollama
from typing import Dict, Any, List
from datetime import datetime


class BatchValidator:
    """Validates test results in batch using a larger model"""

    def __init__(self, validation_model: str = "llama3.2:3b-instruct-q4_K_M"):
        """Initialize batch validator

        Args:
            validation_model: Model to use for batch validation (should be larger/better than RAG model)
        """
        self.validation_model = validation_model
        self.client = ollama.Client()

    def validate_batch(
        self,
        results_file: str,
        output_file: str = "test_validation_report.json"
    ) -> Dict[str, Any]:
        """Validate a batch of test results

        Args:
            results_file: Path to JSON file with test results (from RAG system)
            output_file: Where to save validation report

        Returns:
            Validation report with pass/fail for each test
        """

        print("Loading test results...")
        with open(results_file, 'r') as f:
            test_data = json.load(f)

        results = test_data.get('results', [])
        print(f"Found {len(results)} tests to validate\n")

        print(f"Loading validation model: {self.validation_model}")
        print("This may take a moment for the first validation...")

        # Validate each test
        validated_results = []
        passed = 0
        failed = 0

        for i, result in enumerate(results, 1):
            print(f"[{i}/{len(results)}] Validating {result['test_id']}...", end=' ')

            validation = self._validate_single(result)

            # Merge validation into result
            result['validation'] = validation
            result['passed'] = validation['passed']
            result['validation_score'] = validation['score']
            result['validation_issues'] = validation['issues']
            result['validation_explanation'] = validation['explanation']

            if validation['passed']:
                passed += 1
                print("PASS")
            else:
                failed += 1
                print("FAIL")

            validated_results.append(result)

        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "validation_model": self.validation_model,
            "summary": {
                "total_tests": len(results),
                "passed": passed,
                "failed": failed,
                "pass_rate": (passed / len(results) * 100) if results else 0,
                "average_score": sum(r['validation']['score'] for r in validated_results) / len(validated_results) if validated_results else 0
            },
            "results": validated_results
        }

        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*80}")
        print(f"VALIDATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total tests: {len(results)}")
        print(f"Passed: {passed} ({passed/len(results)*100:.1f}%)")
        print(f"Failed: {failed} ({failed/len(results)*100:.1f}%)")
        print(f"Average score: {report['summary']['average_score']:.2f}")
        print(f"\nReport saved to: {output_file}")

        return report

    def _validate_single(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single test result

        Args:
            result: Test result dict with question, answer, expected_keywords, etc.

        Returns:
            Validation dict with passed, score, issues, explanation
        """

        question = result['question']
        answer = result['answer']
        expected_keywords = result.get('expected_keywords', [])
        expected_answer = result.get('expected_answer', '')
        category = result.get('category', '')
        is_impossible = result.get('is_impossible', False)

        # Build validation prompt
        if is_impossible:
            validation_prompt = self._build_impossible_validation_prompt(question, answer)
        else:
            validation_prompt = self._build_validation_prompt(
                question, answer, expected_keywords, expected_answer, category
            )

        # Get validation from LLM
        try:
            response = self.client.generate(
                model=self.validation_model,
                prompt=validation_prompt,
                options={"temperature": 0.0},
                keep_alive="30m"  # Keep model loaded for batch processing
            )

            validation_text = response['response'].strip()
            return self._parse_validation(validation_text)

        except Exception as e:
            return {
                "passed": False,
                "score": 0.0,
                "issues": [f"Validation error: {str(e)}"],
                "explanation": "Failed to validate"
            }

    def _build_validation_prompt(
        self,
        question: str,
        answer: str,
        expected_keywords: List[str],
        expected_answer: str,
        category: str
    ) -> str:
        """Build validation prompt for normal questions"""

        # Build expected content section
        expected_content_section = ""
        if expected_answer:
            expected_content_section = f"""EXPECTED ANSWER (primary criteria):
{expected_answer}

The answer should convey this information, even if worded differently."""

        if expected_keywords:
            if expected_content_section:
                expected_content_section += f"\n\nKEY ELEMENTS (supporting check):\n"
            else:
                expected_content_section = "KEY ELEMENTS (should be present):\n"
            expected_content_section += chr(10).join(f"- {kw}" for kw in expected_keywords)

        if not expected_content_section:
            expected_content_section = "(No specific expectations - evaluate if answer is reasonable and complete)"

        return f"""You are evaluating if an AI answer is correct and complete.

QUESTION: {question}
CATEGORY: {category}

ANSWER TO EVALUATE:
{answer}

{expected_content_section}

EVALUATION CRITERIA:
1. SEMANTIC CORRECTNESS: Does answer convey the expected meaning/information?
2. COMPLETENESS: Are all key elements present?
3. ACCURACY: Is the information factually correct?
4. PRECISION: Does it directly answer without irrelevant info?
5. NO REFUSAL: Answer shouldn't refuse if information is available

PRIMARY FOCUS: Judge based on whether the answer is SEMANTICALLY CORRECT, not just keyword matching.
The answer may use different wording, format, or structure - that's fine if the meaning is correct.

IMPORTANT - SEMANTIC EQUIVALENCE:
When checking for expected keywords, treat these as EQUIVALENT:
- Dates: "2024-06-26" = "June 26, 2024" = "June 26th, 2024" = "6/26/2024"
- Numbers: "4.0" = "4.00" = "4" (for GPAs, scores)
- Versions: "3.2" = "3.2.0" = "v3.2"
- Names: "CKA" = "Certified Kubernetes Administrator"
- Abbreviations: "MS" = "Master of Science", "BS" = "Bachelor of Science"

If the answer contains the SEMANTIC MEANING of expected keywords (even in different format),
consider it present. Do NOT penalize for format differences.

SCORING:
- 1.0: Perfect - all expected content, accurate, direct
- 0.9: Excellent - all expected content with minor format issues
- 0.7: Good - most expected content, accurate
- 0.5: Partial - missing significant content OR some inaccuracies
- 0.3: Poor - missing most content OR major inaccuracies
- 0.0: Wrong - completely incorrect OR refuses when it shouldn't

Provide evaluation in EXACTLY this format:
SCORE: [0.0-1.0]
PASSED: [YES if score >= 0.8, else NO]
ISSUES: [list specific problems, or "none"]
EXPLANATION: [1-2 sentences explaining the score]

Evaluation:"""

    def _build_impossible_validation_prompt(
        self,
        question: str,
        answer: str
    ) -> str:
        """Build validation prompt for impossible questions"""

        return f"""You are evaluating if an AI correctly handled an IMPOSSIBLE question.

QUESTION: {question}

This question CANNOT be answered from the provided documents.
The correct response should express uncertainty.

ANSWER TO EVALUATE:
{answer}

CRITERIA FOR CORRECT HANDLING:
+ Expresses uncertainty ("I don't know", "not mentioned", etc.)
+ Doesn't make up information
- Providing specific facts/dates (likely made up)
- Confidently answering with details

SCORING:
- 1.0: Perfect - clearly expresses uncertainty, no false info
- 0.8: Good - expresses uncertainty, minor issues
- 0.5: Partial - some uncertainty but also provides some info
- 0.0: Wrong - confidently provides made-up information

Provide evaluation in EXACTLY this format:
SCORE: [0.0-1.0]
PASSED: [YES if score >= 0.8, else NO]
ISSUES: [list problems, or "none"]
EXPLANATION: [1-2 sentences]

Evaluation:"""

    def _parse_validation(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM validation response"""

        lines = response_text.strip().split('\n')

        validation = {
            "passed": False,
            "score": 0.0,
            "issues": [],
            "explanation": ""
        }

        for line in lines:
            line = line.strip()

            if line.startswith("SCORE:"):
                try:
                    score_str = line.split(":", 1)[1].strip()
                    validation["score"] = float(score_str)
                except ValueError:
                    validation["score"] = 0.0

            elif line.startswith("PASSED:"):
                # Read LLM's opinion but don't trust it
                value = line.split(":", 1)[1].strip().upper()
                llm_passed = value == "YES"
                # We'll override this based on score below

            elif line.startswith("ISSUES:"):
                issues_str = line.split(":", 1)[1].strip()
                if issues_str.lower() != "none":
                    validation["issues"] = [i.strip() for i in issues_str.split(",")]

            elif line.startswith("EXPLANATION:"):
                validation["explanation"] = line.split(":", 1)[1].strip()

        # IMPORTANT: Don't trust the LLM's PASSED decision
        # Use strict threshold: score must be >= 0.8 to pass
        validation["passed"] = validation["score"] >= 0.8

        return validation


def main():
    """Main entry point for batch validation"""
    import argparse

    parser = argparse.ArgumentParser(description="Batch validate RAG test results")
    parser.add_argument(
        "--results",
        default="test_results.json",
        help="Test results file from RAG system"
    )
    parser.add_argument(
        "--output",
        default="test_validation_report.json",
        help="Output file for validation report"
    )
    parser.add_argument(
        "--model",
        default="llama3.2:3b-instruct-q4_K_M",
        help="Model to use for validation"
    )

    args = parser.parse_args()

    validator = BatchValidator(validation_model=args.model)
    validator.validate_batch(args.results, args.output)


if __name__ == "__main__":
    main()
