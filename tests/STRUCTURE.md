# Tests Directory Structure

This document describes the organization of the testing infrastructure.

## Directory Tree

```
tests/
├── runners/                   # Test execution scripts
│   ├── __init__.py
│   ├── run_tests.py                      # Main two-phase test runner (single-turn)
│   ├── run_multiturn_tests.py            # Multi-turn conversation test runner
│   ├── run_negative_inference_tests.py   # Negative inference test wrapper
│   └── run_quality_analysis.py           # Quality analysis runner
│
├── validators/                # Test validation logic
│   ├── __init__.py
│   ├── batch_validator.py   # Batch validation with LLM
│   └── llm_validator.py      # LLM-based test validation
│
├── analyzers/                 # Test quality analysis
│   ├── __init__.py
│   ├── quality_analyzer.py   # Main quality analyzer
│   ├── quality_checks.py     # Quality check definitions
│   └── quality_report.py     # Report generation
│
├── fixtures/                  # Test data and cases
│   ├── __init__.py
│   ├── test_suite.json                 # Main single-turn test suite (43 tests)
│   ├── multiturn_test_suite.json       # Multi-turn conversation tests (8 scenarios, 40+ turns)
│   └── negative_inference_test.json    # Negative inference tests
│
├── docs/                      # Test documentation
│   ├── README.md                      # Main testing guide
│   ├── MULTITURN_TESTING.md           # Multi-turn testing guide
│   └── MULTITURN_FIX_SUMMARY.md       # Implementation summary
│
├── utils/                     # Test utilities
│   └── __init__.py
│
├── test_multiturn.py              # Basic multi-turn test script (5 turns)
├── test_multiturn_edge_cases.py   # Edge case multi-turn tests (5 scenarios)
├── __init__.py
└── STRUCTURE.md                   # This file
```

## File Descriptions

### Test Runners (`runners/`)

| File | Purpose | Run Time | Entry Point |
|------|---------|----------|-------------|
| `run_tests.py` | Main single-turn test suite runner with two-phase validation | 5-10 min | `main()` |
| `run_multiturn_tests.py` | Multi-turn conversation test runner | 2-5 min | `main()` |
| `run_negative_inference_tests.py` | Tests negative inference (missing entities) | 2-3 min | `main()` |
| `run_quality_analysis.py` | Analyzes test quality and generates reports | 1-2 min | `main()` |

### Validators (`validators/`)

| File | Purpose |
|------|---------|
| `batch_validator.py` | Validates test answers in batch using LLM |
| `llm_validator.py` | Core LLM-based validation logic |

### Analyzers (`analyzers/`)

| File | Purpose |
|------|---------|
| `quality_analyzer.py` | Main quality analysis engine |
| `quality_checks.py` | Defines quality check rules |
| `quality_report.py` | Generates analysis reports |

### Fixtures (`fixtures/`)

| File | Tests | Format |
|------|-------|--------|
| `test_suite.json` | 43 single-turn tests | Standard test format |
| `multiturn_test_suite.json` | 8 conversations, 40+ turns | Conversation format |
| `negative_inference_test.json` | Negative inference tests | Standard test format |

### Documentation (`docs/`)

| File | Purpose |
|------|---------|
| `README.md` | Main testing guide with quick start examples |
| `MULTITURN_TESTING.md` | Comprehensive multi-turn testing documentation |
| `MULTITURN_FIX_SUMMARY.md` | Technical implementation details of multi-turn fix |

### Test Scripts (Root Level)

| File | Purpose | Run Time |
|------|---------|----------|
| `test_multiturn.py` | Basic 5-turn conversation test | ~10s |
| `test_multiturn_edge_cases.py` | Edge cases and challenging scenarios | ~60s |

## Test Categories

### Single-Turn Tests (`test_suite.json`)

| Category | Count | Description |
|----------|-------|-------------|
| transcript | 8 | Academic transcript queries |
| certifications | 4 | Professional certifications |
| work_experience | 4 | Employment history |
| skills | 3 | Technical skills |
| projects | 2 | Personal projects |
| impossible | 5 | Questions that should refuse |
| edge_case | 3 | Edge cases and ambiguity |
| ambiguity_detection | 2 | Ambiguity handling |
| ambiguity_control | 2 | Control tests for ambiguity |
| clarification_handling | 2 | Clarification requests |
| broad | 5 | Broad/summary queries |
| summary_requests | 1 | Summary generation |
| negative_inference | 1 | Negative inference logic |
| **Total** | **43** | |

### Multi-Turn Tests (`multiturn_test_suite.json`)

| Conversation ID | Category | Turns | Description |
|-----------------|----------|-------|-------------|
| multiturn_001 | pronoun_reference | 3 | Pronoun resolution across turns |
| multiturn_002 | short_followup | 2 | Single-word follow-ups |
| multiturn_003 | topic_switching | 3 | Topic changes with context |
| multiturn_004 | implicit_reference | 3 | Elliptical questions |
| multiturn_005 | work_experience | 3 | Domain-specific flows |
| multiturn_006 | ambiguous_pronoun | 2 | Ambiguous pronoun handling |
| multiturn_007 | mixed_topics | 4 | Rapid topic switching |
| multiturn_008 | long_conversation | 8 | History management over many turns |
| **Total** | **8** | **28** | |

## Running Tests

### Quick Start Commands

```bash
# Single-turn tests
python tests/runners/run_tests.py

# Multi-turn tests
python tests/runners/run_multiturn_tests.py

# Quick multi-turn smoke test
python tests/test_multiturn.py

# Edge case tests
python tests/test_multiturn_edge_cases.py

# Quality analysis
python tests/runners/run_quality_analysis.py
```

### Filtering Tests

```bash
# Single-turn: Run specific category
python tests/runners/run_tests.py --category certifications

# Multi-turn: Run specific conversation
python tests/runners/run_multiturn_tests.py --conversation-id multiturn_001

# Multi-turn: Run specific category
python tests/runners/run_multiturn_tests.py --category short_followup
```

## Test Output Files

| File | Generated By | Contains |
|------|--------------|----------|
| `test_answers.json` | `run_tests.py --phase collect` | RAG answers for single-turn tests |
| `test_validation_report.json` | `run_tests.py --phase validate` | Validation results |
| `quality_report.json` | `run_quality_analysis.py` | Quality analysis results |
| `quality_report.md` | `run_quality_analysis.py --format markdown` | Human-readable report |

## Development Workflow

### Adding Single-Turn Tests

1. Edit `tests/fixtures/test_suite.json`
2. Add test case with expected behavior
3. Run: `python tests/runners/run_tests.py --test-id YOUR_TEST_ID`

### Adding Multi-Turn Tests

1. Edit `tests/fixtures/multiturn_test_suite.json`
2. Add conversation with multiple turns
3. Run: `python tests/runners/run_multiturn_tests.py --conversation-id YOUR_CONV_ID`

### Creating Standalone Tests

```python
# tests/test_custom.py
import requests

API_URL = "http://localhost:8000/chat"
API_KEY = "dev-key-1"

def chat(question, session_id=None):
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id

    response = requests.post(
        API_URL,
        json=payload,
        headers={"X-API-Key": API_KEY}
    )
    return response.json()

# Your test logic here
r1 = chat("Your question")
assert r1["grounded"], "Should be grounded"
```

## CI/CD Integration

### Recommended Test Sequence

```bash
# 1. Quick smoke test
python tests/test_multiturn.py

# 2. Run all single-turn tests
python tests/runners/run_tests.py

# 3. Run all multi-turn tests
python tests/runners/run_multiturn_tests.py

# 4. Quality analysis
python tests/runners/run_quality_analysis.py --format json --output quality.json
```

### Exit Codes

- `0` - All tests passed
- `1` - Some tests failed or errors occurred

## Best Practices

### Test Design
1. **Single tests should be atomic:** Test one thing per test
2. **Multi-turn tests should tell a story:** Natural conversation flow
3. **Use descriptive IDs:** `certification_expiration_date` not `cert_001`
4. **Include expected keywords:** Help validation identify correct answers

### Test Maintenance
1. **Review failures carefully:** System may be correct, test may be wrong
2. **Update expected keywords:** As documents change, update expectations
3. **Add edge cases:** When bugs found, add regression tests
4. **Document limitations:** Note known issues in test descriptions

### Performance
1. **Use filtering:** Don't run all tests every time during development
2. **Batch validation:** Use two-phase testing for large test suites
3. **Parallel execution:** Run independent test categories in parallel (future)

## Related Documentation

- **Main README:** `../README.md` (project root)
- **Test Guide:** `docs/README.md` (this directory)
- **Multi-Turn Guide:** `docs/MULTITURN_TESTING.md`
- **Implementation Details:** `docs/MULTITURN_FIX_SUMMARY.md`
- **API Documentation:** `../docs/API.md` (if exists)

## Support

For issues with tests:
1. Check test output for specific error messages
2. Review documentation in `docs/`
3. Verify service is running: `curl http://localhost:8000/health`
4. Check Redis is running: `docker-compose ps redis`
5. Review service logs: `docker-compose logs api`

## Version History

- **v1.0** (2025-12-02) - Initial test suite with 43 single-turn tests
- **v1.1** (2025-12-15) - Added multi-turn conversation tests (8 scenarios)
- **v1.2** (2025-12-15) - Consolidated test structure and documentation
