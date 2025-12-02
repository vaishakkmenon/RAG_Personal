#!/usr/bin/env python3
"""
Convenience script to run negative inference tests

This is a wrapper around run_tests.py specifically for negative inference testing.

Usage Examples:
    # Run all negative inference tests
    python tests/run_negative_inference_tests.py

    # Run specific category
    python tests/run_negative_inference_tests.py --category anti_memorization

    # Run specific test
    python tests/run_negative_inference_tests.py --test-id neg_001_phd

    # Just collect answers (no validation)
    python tests/run_negative_inference_tests.py --phase collect

    # Validate existing answers
    python tests/run_negative_inference_tests.py --phase validate

    # Use custom output files
    python tests/run_negative_inference_tests.py --answers-file neg_answers.json --report-file neg_report.json
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Base command
    cmd = [
        sys.executable,
        "tests/runners/run_tests.py",
        "--tests", "tests/fixtures/negative_inference_test.json",
        "--answers-file", "test_negative_inference_answers.json",
        "--report-file", "test_negative_inference_report.json"
    ]

    # Forward all command line arguments
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
