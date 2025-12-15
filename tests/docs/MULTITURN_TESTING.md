# Multi-Turn Conversation Testing Guide

## Overview

This guide covers testing multi-turn conversation functionality in the Personal RAG system.

## Quick Start

```bash
# Run basic multi-turn tests
python tests/test_multiturn.py

# Run edge case tests
python tests/test_multiturn_edge_cases.py

# Run comprehensive multi-turn test suite
python tests/runners/run_multiturn_tests.py
```

## What Was Fixed

### Problem
Short follow-up questions (e.g., "Expiration?", "When?", "Why?") were incorrectly flagged as ambiguous even when conversation history provided clear context.

### Solution
Modified the ambiguity detection function to be context-aware:
- **File:** `app/core/chat_service.py`
- **Function:** `_is_truly_ambiguous()` (lines 66-112)
- **Change:** Now accepts optional `conversation_history` parameter
- **Behavior:** When history exists, allows short questions that would otherwise be flagged

### Example

**Before Fix:**
```
Turn 1: "What certifications do I have?"
  → "You have CKA, AWS CCP, AWS CAP..."
Turn 2: "Expiration?"
  → "Your question is too vague. Please clarify." ❌
```

**After Fix:**
```
Turn 1: "What certifications do I have?"
  → "You have CKA, AWS CCP, AWS CAP..."
Turn 2: "Expiration?"
  → "AWS CCP expires May 26, 2028. AWS CAP expires June 1, 2028." ✅
```

## Test Files

### Test Scripts

| File | Purpose | Run Time |
|------|---------|----------|
| `tests/test_multiturn.py` | Basic 5-turn conversation test | ~10s |
| `tests/test_multiturn_edge_cases.py` | Edge cases and challenging scenarios | ~60s |
| `tests/runners/run_multiturn_tests.py` | Comprehensive test runner | ~2-5min |

### Test Fixtures

| File | Contains |
|------|----------|
| `tests/fixtures/multiturn_test_suite.json` | 8 conversation scenarios with 40+ turns |

## Test Scenarios

### 1. Pronoun Reference (multiturn_001)
Tests that pronouns are correctly resolved using conversation context.
```
Turn 1: "What certifications do I have?"
Turn 2: "When do they expire?"  # "they" → certifications
Turn 3: "Which one was earned most recently?"  # "which one" → certifications
```

### 2. Short Follow-up (multiturn_002)
Tests single-word questions that rely entirely on context.
```
Turn 1: "What certifications do I have?"
Turn 2: "Expiration?"  # Single word, context-dependent
```

### 3. Topic Switching (multiturn_003)
Tests handling of topic changes while maintaining relevant context.
```
Turn 1: "What was my graduate GPA?"
Turn 2: "What about my undergraduate GPA?"  # Related topic
Turn 3: "When did I graduate with my Bachelor's degree?"  # New aspect
```

### 4. Implicit Reference (multiturn_004)
Tests elliptical questions and implicit references.
```
Turn 1: "What degrees did I earn?"
Turn 2: "When?"  # Extremely short
Turn 3: "And the GPAs?"  # Implicit reference to degrees
```

### 5. Work Experience (multiturn_005)
Tests domain-specific conversation flows.
```
Turn 1: "What companies have I worked for?"
Turn 2: "What did I do at Maven Wave?"
Turn 3: "How long?"  # Context: duration at Maven Wave
```

### 6. Ambiguous Pronoun (multiturn_006)
Tests handling of potentially ambiguous references.
```
Turn 1: "What companies have I worked for?"  # Returns multiple
Turn 2: "What did I do there?"  # "there" is ambiguous
```

### 7. Mixed Topics (multiturn_007)
Tests rapid topic switching across domains.
```
Turn 1: Education → Turn 2: Certifications → Turn 3: Work → Turn 4: Education
```

### 8. Long Conversation (multiturn_008)
Tests history management over 8+ turns.

## How It Works Internally

### Session Flow
```
1. User sends question with optional session_id
2. System retrieves or creates session
3. Conversation history loaded from session
4. Ambiguity check considers history ← KEY FIX
5. Hybrid retrieval: current question + context
6. LLM generates answer using history
7. Turn saved to session
8. Response includes session_id for next turn
```

### Context-Aware Retrieval
When conversation history exists:
1. **Primary retrieval:** Search with current question
2. **Context retrieval:** Search with recent conversation turns
3. **Merge & deduplicate:** Combine results
4. **Rerank:** Optional cross-encoder reranking

### History Management
- **Max tokens:** 200 (configurable via `SESSION_MAX_HISTORY_TOKENS`)
- **Max turns:** 5 (configurable via `SESSION_MAX_HISTORY_TURNS`)
- **Storage:** Redis (persistent) or Memory (ephemeral)
- **TTL:** 6 hours default (configurable via `SESSION_TTL_SECONDS`)

## Configuration

Edit `.env` file:

```bash
# Conversation history
SESSION_MAX_HISTORY_TOKENS=200    # Max tokens for history in prompt
SESSION_MAX_HISTORY_TURNS=5        # Max conversation turns to keep

# Session storage
SESSION_STORAGE_BACKEND=redis      # 'redis' (persistent) or 'memory' (ephemeral)
SESSION_REDIS_URL=redis://localhost:6379/0
SESSION_TTL_SECONDS=21600          # 6 hours

# Rate limiting
SESSION_QUERIES_PER_HOUR=10        # Max queries per session per hour
```

## Running Tests

### Basic Test
```bash
cd tests
python test_multiturn.py
```

**Expected output:**
```
[Turn 1] User: What certifications do I have?
  Grounded: True
[Turn 2] User: When do they expire?
  Grounded: True
...
[OK] Session IDs are consistent across all turns
[OK] Test completed successfully
```

### Edge Case Tests
```bash
cd tests
python test_multiturn_edge_cases.py
```

### Comprehensive Test Suite
```bash
# Run all scenarios
python tests/runners/run_multiturn_tests.py

# Run specific category
python tests/runners/run_multiturn_tests.py --category short_followup

# Run specific conversation
python tests/runners/run_multiturn_tests.py --conversation-id multiturn_002

# Verbose output
python tests/runners/run_multiturn_tests.py --verbose
```

## Test Results Interpretation

### Success Criteria
- ✓ Session ID consistent across all turns
- ✓ Context-dependent questions are grounded
- ✓ Pronouns correctly resolved
- ✓ Answers reference conversation history

### Common Failures

**Issue:** Turn 2 asks for clarification
```
Turn 2: "Expiration?"
  Answer: "Could you clarify which aspect..."
  Grounded: False
```
**Cause:** Ambiguity check not using conversation history
**Fix:** Ensure code changes applied and service restarted

**Issue:** Session IDs change between turns
```
Turn 1: session_id = abc-123
Turn 2: session_id = def-456  # Different!
```
**Cause:** Not passing session_id in subsequent requests
**Fix:** Ensure test script passes `session_id` from previous response

**Issue:** Context not being used
```
Turn 2 doesn't seem aware of Turn 1
```
**Cause:** Session storage not working (Redis down?)
**Fix:** Check Redis is running: `docker-compose ps redis`

## Troubleshooting

### Test Failures

1. **Check service is running**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check Redis is running** (for session persistence)
   ```bash
   docker-compose ps redis
   ```

3. **Check logs**
   ```bash
   docker-compose logs api | grep -i "conversation history"
   ```

4. **Restart services**
   ```bash
   docker-compose restart api
   ```

### Debugging Tips

1. **Enable verbose logging** in `tests/test_multiturn.py`:
   ```python
   # Print full responses
   print(f"Full answer: {response['answer']}")
   print(f"Sources: {response['sources']}")
   ```

2. **Check session storage**:
   ```bash
   # Connect to Redis
   docker-compose exec redis redis-cli

   # List all sessions
   KEYS session:*

   # View specific session
   GET session:abc-123-def-456
   ```

3. **Test manually with curl**:
   ```bash
   # Turn 1
   curl -X POST http://localhost:8000/chat \
     -H "X-API-Key: dev-key-1" \
     -H "Content-Type: application/json" \
     -d '{"question": "What certifications do I have?"}'

   # Turn 2 (use session_id from Turn 1)
   curl -X POST http://localhost:8000/chat \
     -H "X-API-Key: dev-key-1" \
     -H "Content-Type: application/json" \
     -d '{"question": "Expiration?", "session_id": "SESSION_ID_HERE"}'
   ```

## Adding New Tests

### Add to Existing Suite

Edit `tests/fixtures/multiturn_test_suite.json`:

```json
{
  "conversation_id": "multiturn_009",
  "category": "new_category",
  "description": "Test description",
  "turns": [
    {
      "turn_number": 1,
      "question": "First question",
      "expected_grounded": true,
      "expected_keywords": ["keyword1", "keyword2"],
      "notes": "Context for test"
    },
    {
      "turn_number": 2,
      "question": "Follow-up question",
      "expected_grounded": true,
      "expected_keywords": ["keyword3"],
      "context_dependent": true,
      "notes": "Depends on Turn 1"
    }
  ]
}
```

### Create Standalone Test

```python
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

# Test your scenario
r1 = chat("Your first question")
r2 = chat("Your follow-up", r1["session_id"])

# Verify
assert r2["grounded"], "Follow-up should be grounded"
print("Test passed!")
```

## Best Practices

### Writing Multi-Turn Tests
1. **Start simple:** Begin with clear context in Turn 1
2. **Be specific:** Use keywords to validate understanding
3. **Test edge cases:** Ambiguous pronouns, rapid topic changes
4. **Check grounding:** Context-dependent turns should be grounded
5. **Verify session:** Ensure session_id is consistent

### Conversation Design
1. **Establish context first:** Turn 1 should set clear topic
2. **Use natural language:** Test how real users would ask
3. **Test failures:** Include cases that should ask for clarification
4. **Mix topics:** Test context switching and tracking

## Limitations

### Current Limitations
1. **History budget:** Only 5 turns kept by default (older context lost)
2. **Ambiguous pronouns:** System may guess if multiple referents exist
3. **Complex chains:** Deep pronoun chains (A→B→C) may fail
4. **Cross-domain:** References to entities 10+ turns ago may not resolve

### Future Improvements
- [ ] Explicit entity tracking
- [ ] Conversation summarization for longer histories
- [ ] Clarification prompts for ambiguous references
- [ ] Support for conversational interruptions

## References

- **Code:** `app/core/chat_service.py` (lines 66-112, 435)
- **Session management:** `app/storage/` directory
- **Settings:** `app/settings.py` (SessionSettings class)
- **Test fixtures:** `tests/fixtures/multiturn_test_suite.json`
- **Documentation:** `tests/docs/MULTITURN_FIX_SUMMARY.md`

## Support

For issues with multi-turn conversations:
1. Check this documentation
2. Review test output for specific error messages
3. Check service logs: `docker-compose logs api`
4. Verify Redis is running if using persistent sessions
5. Restart services if needed: `docker-compose restart api`
