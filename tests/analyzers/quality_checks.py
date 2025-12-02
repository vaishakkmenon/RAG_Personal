"""
Quality check functions for test validation analysis.

Each function takes test_result and test_definition dicts and returns
a list of quality issues found.
"""

from typing import Dict, Any, List


def is_uncertainty_response(answer: str) -> bool:
    """Check if answer expresses uncertainty (e.g., 'I don't know')"""
    uncertainty_phrases = [
        "don't know", "do not know", "not mentioned", "not available",
        "can't find", "cannot find", "isn't mentioned", "no information",
        "not in the", "not provided", "unable to find"
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in uncertainty_phrases)


def check_grounding_compliance(test_result: Dict[str, Any], test_definition: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Check if grounding behavior matches expectations.

    CRITICAL priority - user requires factual answers to be grounded.
    """
    issues = []

    grounded = test_result.get('grounded', False)
    num_sources = test_result.get('num_sources', 0)
    grounded_expected = test_definition.get('grounded_expected')
    is_impossible = test_definition.get('is_impossible', False)
    passed = test_result.get('passed', False)
    answer = test_result.get('answer', '')

    # CRITICAL: Grounding mismatch
    if grounded_expected is not None and grounded != grounded_expected:
        severity = 'CRITICAL' if passed else 'HIGH'
        issues.append({
            'type': 'GROUNDING_MISMATCH',
            'severity': severity,
            'description': f'Expected grounded={grounded_expected}, got grounded={grounded}',
            'expected': grounded_expected,
            'actual': grounded,
            'impact': 'Factual answer without source grounding' if grounded_expected else 'Answer grounded when it should not be'
        })

    # CRITICAL: Ungrounded factual answer (user requirement)
    if grounded_expected and not grounded and num_sources == 0:
        # Only flag if answer provides facts (not uncertainty)
        if not is_uncertainty_response(answer):
            issues.append({
                'type': 'UNGROUNDED_FACTUAL_ANSWER',
                'severity': 'CRITICAL',
                'description': 'Provides factual information without grounding',
                'expected': 'grounded answer with sources',
                'actual': f'grounded={grounded}, num_sources={num_sources}',
                'impact': 'Violates grounding requirement for factual answers'
            })

    # CRITICAL: Impossible question should not be grounded
    if is_impossible and grounded:
        issues.append({
            'type': 'IMPOSSIBLE_GROUNDED',
            'severity': 'CRITICAL',
            'description': 'Impossible question should not be grounded',
            'expected': 'grounded=False',
            'actual': 'grounded=True',
            'impact': 'System is likely making up information'
        })

    # HIGH: False grounding (claims grounded but no sources)
    if grounded and num_sources == 0 and not is_impossible:
        issues.append({
            'type': 'FALSE_GROUNDING',
            'severity': 'HIGH',
            'description': 'Marked as grounded but has 0 sources',
            'expected': 'num_sources > 0 if grounded',
            'actual': {'grounded': True, 'num_sources': 0},
            'impact': 'System claims factual basis without evidence'
        })

    return issues


def check_source_requirements(test_result: Dict[str, Any], test_definition: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Check if source count meets minimum requirements.
    """
    issues = []

    num_sources = test_result.get('num_sources', 0)
    min_sources = test_definition.get('min_sources', 0)
    passed = test_result.get('passed', False)

    # Insufficient sources
    if min_sources > 0 and num_sources < min_sources:
        severity = 'HIGH' if passed else 'MEDIUM'
        issues.append({
            'type': 'INSUFFICIENT_SOURCES',
            'severity': severity,
            'description': f'Has {num_sources} sources, needs {min_sources}',
            'expected': min_sources,
            'actual': num_sources,
            'impact': 'Answer may lack sufficient evidence'
        })

    return issues


def check_impossible_behavior(test_result: Dict[str, Any], test_definition: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Check if impossible questions are handled correctly.

    Context-dependent approach:
    - Truly impossible (favorite color, birthplace) -> direct uncertainty
    - Ambiguous impossible questions -> menu OK
    """
    is_impossible = test_definition.get('is_impossible', False)
    if not is_impossible:
        return []

    issues = []
    answer = test_result.get('answer', '')
    answer_lower = answer.lower()
    question = test_result.get('question', '')
    question_lower = question.lower()
    passed = test_result.get('passed', False)

    # Detect response type
    has_uncertainty = is_uncertainty_response(answer)

    menu_indicators = [
        "i can help with:", "what would you like to know",
        "academic history", "work experience", "certifications",
        "technical skills", "options:", "choose from",
        "such as", "including", "like:"
    ]
    has_menu = any(indicator in answer_lower for indicator in menu_indicators)

    # Determine if truly impossible (vs ambiguous)
    truly_impossible_indicators = [
        "favorite", "birthplace", "born", "breakfast", "salary",
        "what did", "where was", "high school", "what is my",
        "where did i", "lunch", "dinner", "spouse", "married",
        "children", "pets", "hobbies", "weekend"
    ]
    is_truly_impossible = any(ind in question_lower for ind in truly_impossible_indicators)

    # Issue 1: Truly impossible provides menu instead of uncertainty
    if is_truly_impossible and has_menu and not has_uncertainty:
        severity = 'HIGH' if passed else 'MEDIUM'
        issues.append({
            'type': 'IMPOSSIBLE_PROVIDES_MENU',
            'severity': severity,
            'description': 'Provides menu for truly impossible question',
            'expected': 'Direct uncertainty expression (I don\'t know)',
            'actual': 'Capability menu',
            'impact': 'Should say "I don\'t know" not offer menus',
            'example': answer[:150] + '...' if len(answer) > 150 else answer
        })

    # Issue 2: Doesn't express uncertainty at all
    if not has_uncertainty and not has_menu:
        # Check if providing specific information (bad for impossible)
        provides_facts = any(word in answer_lower for word in
                           ['earned', 'graduated', 'worked', 'studied', 'located', 'born in', 'is'])
        if provides_facts:
            issues.append({
                'type': 'IMPOSSIBLE_PROVIDES_FACTS',
                'severity': 'CRITICAL',
                'description': 'Provides specific facts for impossible question',
                'expected': 'Uncertainty expression',
                'actual': 'Factual information',
                'impact': 'System is making up information',
                'example': answer[:150] + '...' if len(answer) > 150 else answer
            })

    return issues


def check_ambiguity_behavior(test_result: Dict[str, Any], test_definition: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Check if ambiguity detection and clarification work as expected.
    """
    category = test_result.get('category', '')
    if category not in ['ambiguity_detection', 'clarification_handling', 'edge_case', 'edge_cases']:
        return []

    issues = []
    expected_behavior = test_definition.get('expected_behavior', {})
    should_ask_clarification = expected_behavior.get('expected_response', {}).get('should_ask_clarification')

    # Only check if expected_behavior explicitly requires clarification
    if should_ask_clarification is None:
        return []

    answer = test_result.get('answer', '')
    answer_lower = answer.lower()

    # Check for clarification indicators
    clarification_phrases = [
        'clarify', 'which', 'specific', 'what would you like',
        'can you specify', 'need more detail', 'which type',
        'what aspect', 'more specific', 'narrow down'
    ]
    has_clarification = any(phrase in answer_lower for phrase in clarification_phrases)

    if should_ask_clarification and not has_clarification:
        issues.append({
            'type': 'MISSING_CLARIFICATION',
            'severity': 'MEDIUM',
            'description': 'Should ask for clarification but doesn\'t',
            'expected': 'Clarification request',
            'actual': 'Direct answer or other',
            'impact': 'Ambiguous query not properly handled'
        })
    elif not should_ask_clarification and has_clarification:
        # Over-clarifying when not needed (lower severity)
        issues.append({
            'type': 'UNNECESSARY_CLARIFICATION',
            'severity': 'LOW',
            'description': 'Asks for clarification when question is clear',
            'expected': 'Direct answer',
            'actual': 'Clarification request',
            'impact': 'May frustrate users with clear questions'
        })

    return issues


def check_completeness(test_result: Dict[str, Any], test_definition: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Check if answers are complete despite passing validation.
    """
    passed = test_result.get('passed', False)
    if not passed:
        return []  # Only check passed tests for completeness

    issues = []
    validation_explanation = test_result.get('validation_explanation', '')
    validation_issues = test_result.get('validation_issues', [])

    # Issue 1: Validation explanation mentions incompleteness
    explanation_lower = validation_explanation.lower()
    incompleteness_words = [
        'missing', 'incomplete', 'lacks', 'does not mention',
        'doesn\'t include', 'not provide', 'absent', 'omitted',
        'does not include', 'fails to mention', 'without'
    ]

    if any(word in explanation_lower for word in incompleteness_words):
        issues.append({
            'type': 'INCOMPLETE_CONTENT',
            'severity': 'MEDIUM',
            'description': 'Validator noted missing content in explanation',
            'explanation': validation_explanation,
            'impact': 'Passed but incomplete'
        })

    # Issue 2: Has validation issues despite passing
    if validation_issues and any(issue.strip() for issue in validation_issues):
        # Filter out empty strings
        real_issues = [issue for issue in validation_issues if issue.strip()]
        if real_issues:
            issues.append({
                'type': 'PASSED_WITH_ISSUES',
                'severity': 'MEDIUM',
                'description': f'Passed but has {len(real_issues)} validation issue(s)',
                'validation_issues': real_issues,
                'impact': 'Quality concerns despite passing score'
            })

    return issues


def check_score_quality(test_result: Dict[str, Any], test_definition: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Check for marginal scores and score-pass consistency.
    """
    issues = []

    score = test_result.get('validation_score', 0.0)
    passed = test_result.get('passed', False)

    # Issue 1: Score-pass inconsistency
    expected_pass = score >= 0.8
    if passed != expected_pass:
        issues.append({
            'type': 'SCORE_PASS_INCONSISTENCY',
            'severity': 'CRITICAL',
            'description': f'Pass status ({passed}) inconsistent with score ({score})',
            'expected': {'passed': expected_pass},
            'actual': {'passed': passed, 'score': score},
            'impact': 'Data integrity issue'
        })

    # Issue 2: Marginal pass (0.8-0.89)
    if passed and 0.8 <= score < 0.9:
        issues.append({
            'type': 'MARGINAL_PASS',
            'severity': 'LOW',
            'description': f'Marginal score: {score:.2f}',
            'score': score,
            'threshold': 0.8,
            'impact': 'Close to threshold, review recommended'
        })

    # Issue 3: Near failure (0.7-0.79) - useful for failed tests
    if not passed and 0.7 <= score < 0.8:
        issues.append({
            'type': 'NEAR_FAILURE',
            'severity': 'MEDIUM',
            'description': f'Just below threshold: {score:.2f}',
            'score': score,
            'threshold': 0.8,
            'impact': 'Failed but close to passing - may need review'
        })

    return issues


# Main check registry - all checks run on each test
ALL_CHECKS = [
    check_grounding_compliance,
    check_source_requirements,
    check_impossible_behavior,
    check_ambiguity_behavior,
    check_completeness,
    check_score_quality,
]
