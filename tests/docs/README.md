# Tests Directory

Organized test suite for the Personal RAG System.

## ⚠️ Important Note on Test Results

**Expected Behavior**: Some tests may not pass as expected because the system's ambiguity detection is **not aggressive enough**. The system may attempt to answer vague or ambiguous questions when it should request clarification instead.

**Common scenarios**:
- Short questions (e.g., "My projects?") receive answers instead of clarification requests
- Vague queries get broad/generic responses that may not match expected specific answers
- Tests may receive "close but not quite right" answers because the system guesses at ambiguous intent

**Why this limitation exists**:
- **Token Budget**: To maximize queries within input token limits, prompts are kept short. Sophisticated ambiguity detection would consume valuable tokens.
- **Cost/Latency**: More accurate ambiguity detection would require an additional LLM call before the main RAG pipeline, adding cost and latency. The current design uses only 1 LLM call per query to keep the system free and fast.
- **Design Trade-off**: The system prioritizes **availability** (always try to answer), **efficiency** (minimal tokens), and **cost** (free operation) over **precision** (only answer clear queries).

See the main documentation for more details.

## Directory Structure

```
tests/
├── runners/           # Test execution scripts
│   ├── run_tests.py                      # Main two-phase test runner
│   ├── run_multiturn_tests.py            # Multi-turn conversation test runner
│   ├── run_negative_inference_tests.py   # Negative inference test wrapper
│   └── run_quality_analysis.py           # Quality analysis runner
│
├── validators/        # Test validation logic
│   ├── batch_validator.py   # Batch validation with LLM
│   └── llm_validator.py      # LLM-based test validation
│
├── analyzers/         # Test quality analysis
│   ├── quality_analyzer.py   # Main quality analyzer
│   ├── quality_checks.py     # Quality check definitions
│   └── quality_report.py     # Report generation
│
├── fixtures/          # Test data and cases
│   ├── test_suite.json                 # Main single-turn test suite (43 tests)
│   ├── multiturn_test_suite.json       # Multi-turn conversation tests (8 scenarios)
│   └── negative_inference_test.json    # Negative inference tests
│
├── docs/              # Test documentation
│   ├── README.md                      # This file
│   ├── MULTITURN_TESTING.md           # Multi-turn testing guide
│   └── MULTITURN_FIX_SUMMARY.md       # Implementation summary
│
├── test_multiturn.py              # Basic multi-turn test script
├── test_multiturn_edge_cases.py   # Edge case multi-turn tests
│
└── utils/             # Test utilities (future)

## Quick Start

### Run Main Test Suite

```bash
# Run all tests (collect + validate)
python tests/runners/run_tests.py

# Run specific category
python tests/runners/run_tests.py --category skills

# Run specific test IDs
python tests/runners/run_tests.py --test-id skills_001 --test-id skills_002

# Just collect answers (no validation)
python tests/runners/run_tests.py --phase collect

# Just validate existing answers
python tests/runners/run_tests.py --phase validate
```

### Run Negative Inference Tests

```bash
# Run all negative inference tests
python tests/runners/run_negative_inference_tests.py

# Run specific category
python tests/runners/run_negative_inference_tests.py --category anti_memorization

# Run specific test
python tests/runners/run_negative_inference_tests.py --test-id neg_001_phd
```

### Run Quality Analysis

```bash
# Basic analysis
python tests/runners/run_quality_analysis.py

# Analyze specific category
python tests/runners/run_quality_analysis.py --category impossible

# Find false positives only
python tests/runners/run_quality_analysis.py --show-passed-only

# High severity issues only
python tests/runners/run_quality_analysis.py --severity-filter HIGH

# Generate markdown report
python tests/runners/run_quality_analysis.py --format markdown --output report.md
```

### Run Multi-Turn Conversation Tests

```bash
# Run all multi-turn tests
python tests/runners/run_multiturn_tests.py

# Run specific conversation
python tests/runners/run_multiturn_tests.py --conversation-id multiturn_001

# Run specific category
python tests/runners/run_multiturn_tests.py --category short_followup

# Verbose output
python tests/runners/run_multiturn_tests.py --verbose

# Quick test scripts
python tests/test_multiturn.py                  # Basic 5-turn test
python tests/test_multiturn_edge_cases.py       # Edge case scenarios
```

**Multi-Turn Test Categories:**
- `pronoun_reference` - Pronoun resolution ("they", "it", "which one")
- `short_followup` - Single-word follow-ups ("Expiration?", "When?")
- `topic_switching` - Topic changes with context
- `implicit_reference` - Elliptical questions
- `work_experience` - Domain-specific flows
- `mixed_topics` - Rapid topic switching
- `long_conversation` - 8+ turn conversations

See `docs/MULTITURN_TESTING.md` for detailed documentation.

## Two-Phase Testing Approach

The test system uses a two-phase approach to optimize performance:

### Phase 1: Collection
- Uses fast 1B parameter model for RAG queries
- Collects all answers quickly
- Saves results to `test_answers.json`

### Phase 2: Validation
- Uses larger 3B parameter model for validation
- Loads model once and validates all answers in batch
- Generates `test_validation_report.json`

This approach avoids repeated model loading/unloading overhead.

## Test Categories

Tests are organized into categories:

- **skills** - Technical skills and experience
- **certifications** - Professional certifications
- **education** - Academic background
- **projects** - Personal projects
- **ambiguity_control** - Ambiguity detection tests
- **negative_inference** - Missing entity detection tests
  - `anti_memorization` - Non-existent entities
  - `no_match` - Entities not in knowledge base
  - `boundary` - Edge cases

## Adding New Tests

1. Add test cases to `fixtures/test_suite.json`:

```json
{
  "test_id": "my_test_001",
  "category": "skills",
  "question": "What is my Python experience?",
  "expected_contains": ["Python", "development"],
  "expected_not_contains": ["Java"],
  "should_pass": true
}
```

2. Run tests:

```bash
python tests/runners/run_tests.py --test-id my_test_001
```

## Output Files

- `test_answers.json` - Collected RAG answers
- `test_validation_report.json` - Validation results
- `test_quality_analysis.json` - Quality analysis results
- `test_negative_inference_answers.json` - Negative inference answers
- `test_negative_inference_report.json` - Negative inference validation

## Configuration

### LLM Provider

Tests support both Groq (local) and Groq (cloud):

```bash
# Use Groq (default)
python tests/runners/run_tests.py --provider groq

# Use Groq
python tests/runners/run_tests.py --provider groq --groq-key YOUR_API_KEY
```

### Custom Models

```bash
# Use custom RAG model
python tests/runners/run_tests.py --rag-model llama3.2:1b-instruct-q4_K_M

# Use custom validation model
python tests/runners/run_tests.py --validation-model llama3.2:3b-instruct-q4_K_M
```

## Development

### Adding Validators

Create new validators in `validators/`:
- Inherit validation logic patterns
- Use consistent interfaces
- Document expected behavior

### Adding Quality Checks

Add checks to `analyzers/quality_checks.py`:
- Define check function
- Specify severity (HIGH, MEDIUM, LOW)
- Add to `ALL_CHECKS` list

### Adding Test Fixtures

Add new test suites to `fixtures/`:
- Follow JSON schema format
- Use descriptive test_ids
- Include expected_contains/expected_not_contains
