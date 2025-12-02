"""
Test Quality Analyzer - Main orchestration for quality analysis.

Loads test validation reports and test definitions, runs quality checks,
and produces analysis results.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from tests.analyzers.quality_checks import ALL_CHECKS


class TestQualityAnalyzer:
    """
    Analyzes test validation reports for quality issues beyond pass/fail.
    """

    def __init__(self, validation_report_path: str, test_suite_path: str):
        """
        Initialize analyzer with report and test suite paths.

        Args:
            validation_report_path: Path to test_validation_report.json
            test_suite_path: Path to test_suite.json
        """
        self.validation_report_path = validation_report_path
        self.test_suite_path = test_suite_path

        # Load data
        self.validation_data = self._load_json(validation_report_path)
        self.test_suite_data = self._load_json(test_suite_path)

        # Index test definitions by test_id for fast lookup
        self.test_definitions = self._index_test_definitions()

    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _index_test_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Index test definitions by test_id."""
        test_cases = self.test_suite_data.get('test_cases', [])
        return {test['test_id']: test for test in test_cases}

    def analyze(
        self,
        severity_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        passed_only: bool = False,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Run comprehensive quality analysis.

        Args:
            severity_filter: Only include issues at or above this severity
            category_filter: Only analyze tests in this category
            passed_only: Only analyze tests that passed (find false positives)
            verbose: Print detailed progress

        Returns:
            List of analysis results, one per test
        """
        results = self.validation_data.get('results', [])

        if verbose:
            print(f"Analyzing {len(results)} tests...")

        analysis_results = []

        for i, test_result in enumerate(results, 1):
            test_id = test_result.get('test_id')
            category = test_result.get('category')
            passed = test_result.get('passed', False)

            # Apply filters
            if category_filter and category != category_filter:
                continue

            if passed_only and not passed:
                continue

            if verbose:
                print(f"[{i}/{len(results)}] Analyzing {test_id}...", end=' ')

            # Get test definition
            test_definition = self.test_definitions.get(test_id, {})

            # Run analysis
            analysis = self._analyze_single_test(test_result, test_definition)

            # Apply severity filter
            if severity_filter:
                analysis['issues'] = self._filter_by_severity(
                    analysis['issues'],
                    severity_filter
                )

            analysis_results.append(analysis)

            if verbose:
                issue_count = len(analysis['issues'])
                if issue_count > 0:
                    print(f"{issue_count} issue(s) found")
                else:
                    print("OK")

        if verbose:
            print(f"\nAnalysis complete: {len(analysis_results)} tests analyzed")

        return analysis_results

    def _analyze_single_test(
        self,
        test_result: Dict[str, Any],
        test_definition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a single test for quality issues.

        Args:
            test_result: Test result from validation report
            test_definition: Test definition from test suite

        Returns:
            Analysis dict with test info and issues found
        """
        # Collect issues from all checks
        all_issues = []

        for check_func in ALL_CHECKS:
            issues = check_func(test_result, test_definition)
            all_issues.extend(issues)

        return {
            'test_id': test_result.get('test_id'),
            'category': test_result.get('category'),
            'question': test_result.get('question'),
            'answer': test_result.get('answer'),
            'test_result': test_result,
            'test_definition': test_definition,
            'issues': all_issues,
            'issue_count': len(all_issues)
        }

    def _filter_by_severity(
        self,
        issues: List[Dict[str, Any]],
        min_severity: str
    ) -> List[Dict[str, Any]]:
        """
        Filter issues by minimum severity level.

        Args:
            issues: List of issue dicts
            min_severity: Minimum severity (CRITICAL, HIGH, MEDIUM, LOW)

        Returns:
            Filtered list of issues
        """
        severity_levels = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        min_level = severity_levels.get(min_severity, 1)

        return [
            issue for issue in issues
            if severity_levels.get(issue.get('severity', 'LOW'), 1) >= min_level
        ]
