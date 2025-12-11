"""
Pattern Configuration Validator

Validates query_patterns.yaml before deployment:
- YAML syntax correctness
- Required fields present
- Regex patterns compile
- Test cases complete
- No duplicate pattern names
"""

import yaml
import re
import sys
from pathlib import Path
from typing import Dict, List, Set


def validate_patterns(config_path: str = "config/query_patterns.yaml") -> bool:
    """
    Validate pattern configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        True if valid, False otherwise
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Load YAML file
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"❌ Config file not found: {config_path}")
        return False

    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"❌ YAML syntax error: {e}")
        return False

    if not config or 'patterns' not in config:
        errors.append("Config must contain 'patterns' key")
        _print_results(errors, warnings)
        return False

    patterns = config['patterns']
    if not isinstance(patterns, list):
        errors.append("'patterns' must be a list")
        _print_results(errors, warnings)
        return False

    # Track pattern names for duplicates
    pattern_names: Set[str] = set()

    # Validate each pattern
    for idx, pattern in enumerate(patterns):
        pattern_id = f"Pattern {idx + 1}"

        # Check required fields
        required_fields = ['name', 'enabled', 'priority', 'matching', 'rewrite_strategy']
        for field in required_fields:
            if field not in pattern:
                errors.append(f"{pattern_id}: missing required field '{field}'")

        # Get pattern name for better error messages
        if 'name' in pattern:
            pattern_name = pattern['name']
            pattern_id = f"Pattern '{pattern_name}'"

            # Check for duplicate names
            if pattern_name in pattern_names:
                errors.append(f"{pattern_id}: duplicate pattern name")
            pattern_names.add(pattern_name)

        # Validate priority is a number
        if 'priority' in pattern:
            if not isinstance(pattern['priority'], (int, float)):
                errors.append(f"{pattern_id}: priority must be a number")
            elif pattern['priority'] < 0 or pattern['priority'] > 100:
                warnings.append(f"{pattern_id}: priority {pattern['priority']} outside typical range (0-100)")

        # Validate matching section
        if 'matching' in pattern:
            matching = pattern['matching']

            if 'type' not in matching:
                errors.append(f"{pattern_id}: matching must have 'type' field")
            else:
                match_type = matching['type']

                if match_type == 'regex_list':
                    # Validate regex patterns
                    if 'rules' not in matching:
                        errors.append(f"{pattern_id}: regex_list matching requires 'rules'")
                    else:
                        rules = matching['rules']
                        if not isinstance(rules, list):
                            errors.append(f"{pattern_id}: 'rules' must be a list")
                        else:
                            for rule_idx, rule in enumerate(rules):
                                if not isinstance(rule, dict):
                                    errors.append(f"{pattern_id}, rule {rule_idx + 1}: must be a dict")
                                    continue

                                if 'pattern' not in rule:
                                    errors.append(f"{pattern_id}, rule {rule_idx + 1}: missing 'pattern'")
                                    continue

                                # Try to compile regex
                                try:
                                    re.compile(rule['pattern'])
                                except re.error as e:
                                    errors.append(f"{pattern_id}, rule {rule_idx + 1}: invalid regex - {e}")

                elif match_type == 'keyword_presence':
                    # Validate keywords
                    if 'keywords' not in matching:
                        errors.append(f"{pattern_id}: keyword_presence matching requires 'keywords'")
                    else:
                        keywords = matching['keywords']
                        if not isinstance(keywords, list):
                            errors.append(f"{pattern_id}: 'keywords' must be a list")
                        elif len(keywords) == 0:
                            warnings.append(f"{pattern_id}: 'keywords' list is empty")

                else:
                    warnings.append(f"{pattern_id}: unknown matching type '{match_type}'")

        # Validate rewrite_strategy section
        if 'rewrite_strategy' in pattern:
            rewrite = pattern['rewrite_strategy']

            if 'type' not in rewrite:
                errors.append(f"{pattern_id}: rewrite_strategy must have 'type' field")
            else:
                strategy_type = rewrite['type']
                valid_types = ['category_expansion', 'list_expansion', 'synonym_expansion', 'passthrough']

                if strategy_type not in valid_types:
                    warnings.append(f"{pattern_id}: unknown rewrite strategy '{strategy_type}'")

        # Validate test cases
        if 'test_cases' in pattern:
            test_cases = pattern['test_cases']
            if not isinstance(test_cases, list):
                errors.append(f"{pattern_id}: 'test_cases' must be a list")
            else:
                for test_idx, test in enumerate(test_cases):
                    if not isinstance(test, dict):
                        errors.append(f"{pattern_id}, test case {test_idx + 1}: must be a dict")
                        continue

                    if 'input' not in test:
                        errors.append(f"{pattern_id}, test case {test_idx + 1}: missing 'input'")

                    if 'expected_output' not in test:
                        errors.append(f"{pattern_id}, test case {test_idx + 1}: missing 'expected_output'")
        else:
            warnings.append(f"{pattern_id}: no test cases defined")

    # Print results
    _print_results(errors, warnings, len(patterns))

    return len(errors) == 0


def _print_results(errors: List[str], warnings: List[str], pattern_count: int = 0):
    """Print validation results."""
    print()
    print("=" * 80)
    print("PATTERN CONFIGURATION VALIDATION")
    print("=" * 80)

    if pattern_count > 0:
        print(f"\nPatterns found: {pattern_count}")

    if errors:
        print(f"\n[FAIL] VALIDATION FAILED ({len(errors)} errors)")
        print()
        for error in errors:
            print(f"  ERROR: {error}")
    else:
        print("\n[OK] All required validations passed")

    if warnings:
        print(f"\n[WARN] Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  WARNING: {warning}")

    print()
    print("=" * 80)


def main():
    """Main entry point."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/query_patterns.yaml"

    print(f"Validating pattern configuration: {config_path}")

    success = validate_patterns(config_path)

    if success:
        print("\n[OK] Configuration is valid and ready for deployment")
        sys.exit(0)
    else:
        print("\n[FAIL] Configuration has errors - fix before deploying")
        sys.exit(1)


if __name__ == "__main__":
    main()
