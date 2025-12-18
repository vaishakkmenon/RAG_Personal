#!/usr/bin/env python3
"""
Test Quality Analyzer - CLI Entry Point

Analyzes test validation reports to identify quality issues beyond pass/fail.

Usage:
    python tests/run_quality_analysis.py
    python tests/run_quality_analysis.py --report custom_report.json
    python tests/run_quality_analysis.py --output detailed_analysis.json
    python tests/run_quality_analysis.py --format markdown --output analysis.md
    python tests/run_quality_analysis.py --show-passed-only --severity-filter HIGH
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.analyzers.quality_analyzer import TestQualityAnalyzer
from tests.analyzers.quality_report import QualityReportGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Analyze test validation reports for quality issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python tests/run_quality_analysis.py

  # Analyze specific category
  python tests/run_quality_analysis.py --category impossible

  # Find false positives only
  python tests/run_quality_analysis.py --show-passed-only

  # High severity issues only
  python tests/run_quality_analysis.py --severity-filter HIGH

  # Generate markdown report
  python tests/run_quality_analysis.py --format markdown --output report.md
        """,
    )

    parser.add_argument(
        "--report",
        default="test_validation_report.json",
        help="Path to validation report file (default: test_validation_report.json)",
    )
    parser.add_argument(
        "--test-suite",
        default="tests/fixtures/test_suite.json",
        help="Path to test suite definition file (default: tests/fixtures/test_suite.json)",
    )
    parser.add_argument(
        "--output",
        help="Output file for quality analysis (default: test_quality_analysis.json or .md based on format)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "console"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--severity-filter",
        choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        help="Only show issues at or above this severity",
    )
    parser.add_argument(
        "--category",
        help="Filter analysis to specific category (e.g., impossible, ambiguity_detection)",
    )
    parser.add_argument(
        "--show-passed-only",
        action="store_true",
        help="Only analyze tests that passed (to find false positives)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed output during analysis"
    )

    args = parser.parse_args()

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        if args.format == "markdown":
            output_file = "test_quality_analysis.md"
        elif args.format == "json":
            output_file = "test_quality_analysis.json"
        else:
            output_file = None

    try:
        # Initialize analyzer
        if args.verbose or args.format == "console":
            print("Initializing quality analyzer...")
            print(f"Report: {args.report}")
            print(f"Test suite: {args.test_suite}")
            if args.category:
                print(f"Category filter: {args.category}")
            if args.severity_filter:
                print(f"Severity filter: {args.severity_filter}")
            if args.show_passed_only:
                print("Mode: False positives only (passed tests with issues)")
            print()

        analyzer = TestQualityAnalyzer(args.report, args.test_suite)

        # Run analysis
        if args.verbose or args.format == "console":
            print("Running comprehensive quality analysis...")

        analysis_results = analyzer.analyze(
            severity_filter=args.severity_filter,
            category_filter=args.category,
            passed_only=args.show_passed_only,
            verbose=args.verbose,
        )

        # Generate report
        if args.verbose or args.format == "console":
            print("\nGenerating quality report...")

        generator = QualityReportGenerator()
        report = generator.generate_report(analysis_results)

        # Output report
        if args.format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"\nQuality analysis saved to: {output_file}")

        elif args.format == "markdown":
            markdown = generator.to_markdown(report)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown)
            print(f"\nQuality analysis saved to: {output_file}")

        elif args.format == "console":
            console_output = generator.to_console(report)
            print("\n" + console_output)
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(console_output)
                print(f"\nConsole output also saved to: {output_file}")

        # Always print summary to console (unless format is already console)
        if args.format != "console":
            print("\n" + "=" * 80)
            print("QUALITY ANALYSIS SUMMARY")
            print("=" * 80)

            summary = report["summary"]
            print(f"Total tests analyzed: {summary['total_tests']}")
            print(f"Tests with quality issues: {summary['tests_with_quality_issues']}")
            print(f"Total quality issues found: {summary['total_quality_issues']}")
            print(f"False positives (passed with issues): {summary['false_positives']}")
            print(f"False positive rate: {summary['false_positive_rate']:.1f}%")

            # Show issue type counts
            if summary["issue_type_counts"]:
                print("\nIssue Type Distribution:")
                for issue_type, count in sorted(
                    summary["issue_type_counts"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    print(f"  {issue_type}: {count}")

            # Show critical findings
            critical = report["critical_findings"]
            if critical:
                print(f"\n{len(critical)} CRITICAL ISSUE(S) FOUND:")
                for finding in critical[:5]:
                    print(
                        f"  [{finding['test_id']}] {finding['issue_type']}: {finding['description']}"
                    )
                if len(critical) > 5:
                    print(f"  ... and {len(critical) - 5} more (see full report)")

            # Show top false positives
            false_positives = report["false_positives"]
            if false_positives:
                print("\nTop False Positives (passed with issues):")
                for fp in false_positives[:5]:
                    print(
                        f"  [{fp['test_id']}] Score: {fp['score']:.2f}, Issues: {fp['issue_count']}"
                    )
                    for issue in fp["issues"][:2]:
                        print(
                            f"    - [{issue['severity']}] {issue['type']}: {issue['description']}"
                        )
                if len(false_positives) > 5:
                    print(
                        f"  ... and {len(false_positives) - 5} more (see full report)"
                    )

            # Show recommendations
            recommendations = report.get("recommendations", [])
            if recommendations:
                print("\nRECOMMENDATIONS:")
                for rec in recommendations[:3]:
                    print(
                        f"  [{rec['priority']}] {rec['category']}: {rec['recommendation']}"
                    )
                if len(recommendations) > 3:
                    print(
                        f"  ... and {len(recommendations) - 3} more (see full report)"
                    )

            print("=" * 80)

        # Exit with error code if critical issues found
        if report["critical_findings"]:
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file - {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
