# Manual Multi-Turn Conversation Testing

This directory contains scripts for manually testing multi-turn conversations.

## Quick Start

### Option 1: Interactive Python Tester (Recommended)

```bash
python scripts/manual_tests/interactive_chat.py
```

**Features:**
- Type questions naturally
- Maintains conversation context automatically
- Shows session info and history
- Commands: `reset`, `history`, `session`, `help`, `quit`

**Example session:**
```
> What certifications do I have?
[Assistant] You have CKA, AWS CCP, AWS CAP...

> Expiration?
[Assistant] AWS CCP expires May 26, 2028...

> Which is newest?
[Assistant] AWS CAP was earned most recently...

> history
[Shows full conversation]

> reset
[Starts new conversation]
```

### Option 2: Shell Script (Linux/Mac/WSL)

```bash
bash scripts/manual_tests/test_conversation.sh
```

**What it does:**
- Runs a predefined 4-turn conversation
- Tests short follow-ups and pronoun resolution
- Shows pass/fail for each turn
- Validates session consistency

### Option 3: Windows Batch Script

```cmd
scripts\manual_tests\test_conversation.bat
```

**What it does:**
- Same as shell script but for Windows
- Uses Windows built-in curl
- Creates temporary response files

### Option 4: Verification Script

```bash
bash scripts/manual_tests/verify_multiturn_fix.sh
```

**What it tests:**
1. ✓ Short follow-ups WITH context are grounded
2. ✓ Short questions WITHOUT context ask for clarification
3. ✓ Session IDs remain consistent

**Exit codes:**
- `0` - All tests passed
- `1` - Some tests failed

## Test Scripts

### interactive_chat.py

**Purpose:** Interactive conversation testing with full control

**Usage:**
```bash
python scripts/manual_tests/interactive_chat.py
```

**Commands:**
| Command | Description |
|---------|-------------|
| `<question>` | Ask a question |
| `reset` | Start new conversation (new session) |
| `history` | Show conversation history |
| `session` | Show session ID and stats |
| `help` | Show help message |
| `quit` | Exit the tester |

**Example:**
```bash
$ python scripts/manual_tests/interactive_chat.py

> What certifications do I have?
[Session] New session created: abc-123-def

[Assistant]
You have CKA, AWS Certified Cloud Practitioner, and AWS Certified AI Practitioner...

[Metadata]
  Grounded: True
  Sources: 5
  Session: abc-123-def

> Expiration?

[Assistant]
AWS CCP expires May 26, 2028. AWS CAP expires June 1, 2028.

[Metadata]
  Grounded: True
  Sources: 5
  Session: abc-123-def
```

### test_conversation.sh

**Purpose:** Automated 4-turn conversation test

**Usage:**
```bash
# Linux/Mac/WSL
bash scripts/manual_tests/test_conversation.sh

# Or make it executable
chmod +x scripts/manual_tests/test_conversation.sh
./scripts/manual_tests/test_conversation.sh
```

**Output:**
```
================================
Multi-Turn Conversation Tester
================================

[Turn 1] What certifications do I have?
----------------------------------------
Answer: You have CKA, AWS CCP, AWS CAP...
Session ID: abc-123-def
Grounded: true

[Turn 2] Expiration?
----------------------------------------
Answer: AWS CCP expires May 26, 2028...
Grounded: true
✓ SUCCESS: Short follow-up was understood from context

[Turn 3] Which one was earned most recently?
----------------------------------------
Answer: AWS CAP was earned most recently...
Grounded: true
✓ SUCCESS: Pronoun reference was resolved

================================
SUMMARY
================================
Turn 1 (Context):      Grounded=true
Turn 2 (Short):        Grounded=true
Turn 3 (Pronoun):      Grounded=true
Turn 4 (Topic Switch): Grounded=true

✓ Multi-turn conversation WORKING
```

### verify_multiturn_fix.sh

**Purpose:** Verify the multi-turn fix is working

**Usage:**
```bash
bash scripts/manual_tests/verify_multiturn_fix.sh
```

**Tests:**
1. **Test 1:** Short follow-up WITH context should be grounded
2. **Test 2:** Short question WITHOUT context should ask for clarification
3. **Test 3:** Session ID should stay consistent

**Output:**
```
================================
Multi-Turn Fix Verification
================================

[Test 1] Short follow-up WITH context
=======================================
Turn 1: 'What certifications do I have?'
  Session: abc-123
  Grounded: true

Turn 2: 'Expiration?' (with session_id)
  Grounded: true
  Answer preview: AWS CCP expires May 26, 2028...

✓ PASS: Short follow-up was grounded (context was used)

[Test 2] Short question WITHOUT context
========================================
Question: 'Expiration?' (no session_id)
  Grounded: false
  Answer preview: Could you clarify which aspect...

✓ PASS: Correctly asked for clarification (no context)

[Test 3] Session ID consistency
================================
  Turn 1 Session: abc-123
  Turn 2 Session: abc-123

✓ PASS: Session ID is consistent across turns

================================
SUMMARY
================================
✓ Test 1: Short follow-up with context
✓ Test 2: Short question without context
✓ Test 3: Session ID consistency

✓✓✓ ALL TESTS PASSED ✓✓✓
Multi-turn conversation fix is WORKING
```

### test_conversation.bat

**Purpose:** Windows version of test_conversation.sh

**Usage:**
```cmd
scripts\manual_tests\test_conversation.bat
```

**Requirements:**
- Windows 10+ (has built-in curl)
- PowerShell or Command Prompt

## Manual Testing with curl

### Basic Pattern

```bash
# Turn 1: Get session ID
SESSION=$(curl -s -X POST http://localhost:8000/chat \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{"question": "Your first question"}' | jq -r '.session_id')

echo "Session: $SESSION"

# Turn 2: Use session ID
curl -s -X POST http://localhost:8000/chat \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Follow-up question\", \"session_id\": \"$SESSION\"}" | jq
```

### Test Scenarios

#### Scenario 1: Short Follow-ups
```bash
# Turn 1
SESSION=$(curl -s -X POST http://localhost:8000/chat \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{"question": "What certifications do I have?"}' | jq -r '.session_id')

# Turn 2 (short)
curl -s -X POST http://localhost:8000/chat \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Expiration?\", \"session_id\": \"$SESSION\"}" | jq '.answer'

# Turn 3 (even shorter)
curl -s -X POST http://localhost:8000/chat \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"When?\", \"session_id\": \"$SESSION\"}" | jq '.answer'
```

#### Scenario 2: Pronoun Resolution
```bash
# Turn 1
SESSION=$(curl -s -X POST http://localhost:8000/chat \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{"question": "What certifications do I have?"}' | jq -r '.session_id')

# Turn 2 (pronoun "they")
curl -s -X POST http://localhost:8000/chat \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"When do they expire?\", \"session_id\": \"$SESSION\"}" | jq '.answer'

# Turn 3 (pronoun "which one")
curl -s -X POST http://localhost:8000/chat \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Which one is newest?\", \"session_id\": \"$SESSION\"}" | jq '.answer'
```

#### Scenario 3: Topic Switching
```bash
# Turn 1
SESSION=$(curl -s -X POST http://localhost:8000/chat \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{"question": "What was my graduate GPA?"}' | jq -r '.session_id')

# Turn 2 (related but different)
curl -s -X POST http://localhost:8000/chat \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What about undergraduate?\", \"session_id\": \"$SESSION\"}" | jq '.answer'

# Turn 3 (new aspect)
curl -s -X POST http://localhost:8000/chat \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"When did I graduate?\", \"session_id\": \"$SESSION\"}" | jq '.answer'
```

## What to Look For

### ✅ Success Indicators

1. **Session ID consistency:**
   ```json
   Turn 1: "session_id": "abc-123"
   Turn 2: "session_id": "abc-123"  ← Same!
   ```

2. **Short questions with context are grounded:**
   ```json
   Question: "Expiration?" (with session_id)
   "grounded": true  ← Should be true
   ```

3. **Context-aware answers:**
   ```
   Turn 1: "What certifications..."
   Turn 2: "Expiration?"
   Answer: "AWS CCP expires..." ← References Turn 1
   ```

### ❌ Failure Indicators

1. **Session ID changes:**
   ```json
   Turn 1: "session_id": "abc-123"
   Turn 2: "session_id": "xyz-789"  ← Different! Bad!
   ```

2. **Asks for clarification with context:**
   ```json
   Question: "Expiration?" (with session_id)
   "grounded": false
   "answer": "Could you clarify..."  ← Should not happen
   ```

3. **Ignores previous context:**
   ```
   Turn 1: "What certifications..."
   Turn 2: "Which is newest?"
   Answer: "I don't have information..."  ← Should reference Turn 1
   ```

## Troubleshooting

### Service Not Running
```
Error: Could not connect to API server
```

**Solution:**
```bash
# Check if service is running
curl http://localhost:8000/health

# If not running, start it
docker-compose up -d
```

### Session Not Persisting
```
Session ID changes between turns
```

**Solution:**
```bash
# Check Redis is running
docker-compose ps redis

# If not running
docker-compose restart redis
```

### Short Questions Still Ask for Clarification
```
Question: "Expiration?" (with session_id)
Answer: "Could you clarify..."
```

**Solution:**
```bash
# Verify code changes are deployed
docker-compose restart api

# Check logs
docker-compose logs api | grep -i "conversation history"
```

## Integration with Automated Tests

These manual scripts complement the automated test suite:

```
Manual Testing (Ad-hoc)
├── interactive_chat.py         ← Interactive exploration
├── test_conversation.sh        ← Quick smoke test
└── verify_multiturn_fix.sh     ← Fix verification

Automated Testing (CI/CD)
├── tests/test_multiturn.py             ← Quick automated test
├── tests/test_multiturn_edge_cases.py  ← Edge case coverage
└── tests/runners/run_multiturn_tests.py ← Comprehensive suite
```

**When to use each:**
- **Manual scripts:** During development, debugging, demonstrations
- **Automated tests:** CI/CD, regression testing, comprehensive coverage

## Examples

### Example 1: Quick Smoke Test
```bash
# Fast verification that multi-turn works
bash scripts/manual_tests/verify_multiturn_fix.sh
```

### Example 2: Interactive Exploration
```bash
# Try different conversation patterns
python scripts/manual_tests/interactive_chat.py

> What certifications do I have?
> Expiration?
> Which is newest?
> reset
> What was my GPA?
> Undergraduate or graduate?
```

### Example 3: Manual curl Testing
```bash
# Test a specific scenario step by step
SESSION=$(curl -s -X POST http://localhost:8000/chat \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{"question": "What certifications do I have?"}' | jq -r '.session_id')

curl -s -X POST http://localhost:8000/chat \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Expiration?\", \"session_id\": \"$SESSION\"}" | jq
```

## See Also

- **Automated Tests:** `tests/README.md`
- **Multi-Turn Guide:** `tests/docs/MULTITURN_TESTING.md`
- **Implementation Details:** `tests/docs/MULTITURN_FIX_SUMMARY.md`
