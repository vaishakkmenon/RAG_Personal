#!/bin/bash
# Verification script for multi-turn conversation fix
# Tests that the ambiguity detection is context-aware

API_URL="http://localhost:8000/chat"
API_KEY="dev-key-1"

echo "================================"
echo "Multi-Turn Fix Verification"
echo "================================"
echo ""
echo "This script verifies that the multi-turn conversation fix is working:"
echo "1. Short follow-ups WITH context should be grounded"
echo "2. Short questions WITHOUT context should ask for clarification"
echo ""

# Helper function to extract JSON field
get_field() {
    local json="$1"
    local field="$2"

    if command -v jq &> /dev/null; then
        echo "$json" | jq -r ".$field"
    else
        echo "$json" | grep -o "\"$field\":\"[^\"]*\"" | cut -d'"' -f4 | head -1
        if [ -z "${PIPESTATUS[0]}" ] || [ "${PIPESTATUS[0]}" != "0" ]; then
            echo "$json" | grep -o "\"$field\":[^,}]*" | cut -d':' -f2 | tr -d ' '
        fi
    fi
}

# Test 1: Short follow-up WITH context should work
echo "[Test 1] Short follow-up WITH context"
echo "======================================="
echo "Turn 1: 'What certifications do I have?'"

RESPONSE1=$(curl -s -X POST "$API_URL" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What certifications do I have?"}')

SESSION_ID=$(get_field "$RESPONSE1" "session_id")
GROUNDED1=$(get_field "$RESPONSE1" "grounded")

echo "  Session: $SESSION_ID"
echo "  Grounded: $GROUNDED1"

if [ -z "$SESSION_ID" ]; then
    echo ""
    echo "[ERROR] Failed to get session ID from Turn 1"
    echo "Response: $RESPONSE1"
    exit 1
fi

echo ""
echo "Turn 2: 'Expiration?' (with session_id)"
sleep 1

RESPONSE2=$(curl -s -X POST "$API_URL" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Expiration?\", \"session_id\": \"$SESSION_ID\"}")

GROUNDED2=$(get_field "$RESPONSE2" "grounded")
ANSWER2=$(get_field "$RESPONSE2" "answer")

echo "  Grounded: $GROUNDED2"
echo "  Answer preview: ${ANSWER2:0:80}..."
echo ""

if [ "$GROUNDED2" = "true" ]; then
    echo "✓ PASS: Short follow-up was grounded (context was used)"
    TEST1_PASS=true
else
    echo "✗ FAIL: Short follow-up was not grounded"
    echo "  Expected: true"
    echo "  Got: $GROUNDED2"
    TEST1_PASS=false
fi

echo ""
echo ""

# Test 2: Short question WITHOUT context should ask for clarification
echo "[Test 2] Short question WITHOUT context"
echo "========================================"
echo "Question: 'Expiration?' (no session_id)"

RESPONSE3=$(curl -s -X POST "$API_URL" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "Expiration?"}')

GROUNDED3=$(get_field "$RESPONSE3" "grounded")
ANSWER3=$(get_field "$RESPONSE3" "answer")

echo "  Grounded: $GROUNDED3"
echo "  Answer preview: ${ANSWER3:0:80}..."
echo ""

if [ "$GROUNDED3" = "false" ]; then
    echo "✓ PASS: Correctly asked for clarification (no context)"
    TEST2_PASS=true
else
    echo "✗ FAIL: Should have asked for clarification"
    echo "  Expected: false"
    echo "  Got: $GROUNDED3"
    TEST2_PASS=false
fi

echo ""
echo ""

# Test 3: Session ID consistency
echo "[Test 3] Session ID consistency"
echo "================================"

SESSION_ID2=$(get_field "$RESPONSE2" "session_id")

echo "  Turn 1 Session: $SESSION_ID"
echo "  Turn 2 Session: $SESSION_ID2"
echo ""

if [ "$SESSION_ID" = "$SESSION_ID2" ]; then
    echo "✓ PASS: Session ID is consistent across turns"
    TEST3_PASS=true
else
    echo "✗ FAIL: Session ID changed between turns"
    echo "  Expected: $SESSION_ID"
    echo "  Got: $SESSION_ID2"
    TEST3_PASS=false
fi

echo ""
echo ""

# Summary
echo "================================"
echo "SUMMARY"
echo "================================"

if [ "$TEST1_PASS" = true ]; then
    echo "✓ Test 1: Short follow-up with context"
else
    echo "✗ Test 1: Short follow-up with context"
fi

if [ "$TEST2_PASS" = true ]; then
    echo "✓ Test 2: Short question without context"
else
    echo "✗ Test 2: Short question without context"
fi

if [ "$TEST3_PASS" = true ]; then
    echo "✓ Test 3: Session ID consistency"
else
    echo "✗ Test 3: Session ID consistency"
fi

echo ""

if [ "$TEST1_PASS" = true ] && [ "$TEST2_PASS" = true ] && [ "$TEST3_PASS" = true ]; then
    echo "✓✓✓ ALL TESTS PASSED ✓✓✓"
    echo "Multi-turn conversation fix is WORKING"
    exit 0
else
    echo "✗✗✗ SOME TESTS FAILED ✗✗✗"
    echo "Multi-turn conversation has ISSUES"
    exit 1
fi
