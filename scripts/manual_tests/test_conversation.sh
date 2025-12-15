#!/bin/bash
# Multi-Turn Conversation Tester
# Tests a complete conversation flow with multiple turns

API_URL="http://localhost:8000/chat"
API_KEY="dev-key-1"

echo "================================"
echo "Multi-Turn Conversation Tester"
echo "================================"
echo ""

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo "Note: jq not found. Install for better formatting: apt-get install jq"
    echo ""
fi

# Turn 1: Establish context
echo "[Turn 1] What certifications do I have?"
echo "----------------------------------------"
RESPONSE1=$(curl -s -X POST "$API_URL" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What certifications do I have?"}')

if command -v jq &> /dev/null; then
    echo "Answer: $(echo "$RESPONSE1" | jq -r '.answer' | head -c 200)..."
    SESSION_ID=$(echo "$RESPONSE1" | jq -r '.session_id')
    GROUNDED=$(echo "$RESPONSE1" | jq -r '.grounded')
else
    echo "$RESPONSE1" | head -c 300
    SESSION_ID=$(echo "$RESPONSE1" | grep -o '"session_id":"[^"]*"' | cut -d'"' -f4)
    GROUNDED=$(echo "$RESPONSE1" | grep -o '"grounded":[^,}]*' | cut -d':' -f2)
fi

echo ""
echo "Session ID: $SESSION_ID"
echo "Grounded: $GROUNDED"
echo ""

if [ -z "$SESSION_ID" ]; then
    echo "ERROR: Failed to get session ID"
    exit 1
fi

sleep 1

# Turn 2: Short follow-up (tests the fix)
echo "[Turn 2] Expiration?"
echo "----------------------------------------"
RESPONSE2=$(curl -s -X POST "$API_URL" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Expiration?\", \"session_id\": \"$SESSION_ID\"}")

if command -v jq &> /dev/null; then
    echo "Answer: $(echo "$RESPONSE2" | jq -r '.answer' | head -c 200)..."
    GROUNDED2=$(echo "$RESPONSE2" | jq -r '.grounded')
    SESSION_ID2=$(echo "$RESPONSE2" | jq -r '.session_id')
else
    echo "$RESPONSE2" | head -c 300
    GROUNDED2=$(echo "$RESPONSE2" | grep -o '"grounded":[^,}]*' | cut -d':' -f2)
    SESSION_ID2=$(echo "$RESPONSE2" | grep -o '"session_id":"[^"]*"' | cut -d'"' -f4)
fi

echo ""
echo "Grounded: $GROUNDED2"
echo "Session ID: $SESSION_ID2"

# Validate Turn 2
if [ "$GROUNDED2" = "true" ]; then
    echo "✓ SUCCESS: Short follow-up was understood from context"
else
    echo "✗ FAIL: Short follow-up was not grounded (expected true)"
fi

if [ "$SESSION_ID" = "$SESSION_ID2" ]; then
    echo "✓ SUCCESS: Session ID is consistent"
else
    echo "✗ FAIL: Session ID changed between turns"
fi
echo ""

sleep 1

# Turn 3: Pronoun reference
echo "[Turn 3] Which one was earned most recently?"
echo "----------------------------------------"
RESPONSE3=$(curl -s -X POST "$API_URL" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Which one was earned most recently?\", \"session_id\": \"$SESSION_ID\"}")

if command -v jq &> /dev/null; then
    echo "Answer: $(echo "$RESPONSE3" | jq -r '.answer' | head -c 200)..."
    GROUNDED3=$(echo "$RESPONSE3" | jq -r '.grounded')
else
    echo "$RESPONSE3" | head -c 300
    GROUNDED3=$(echo "$RESPONSE3" | grep -o '"grounded":[^,}]*' | cut -d':' -f2)
fi

echo ""
echo "Grounded: $GROUNDED3"

if [ "$GROUNDED3" = "true" ]; then
    echo "✓ SUCCESS: Pronoun reference was resolved"
else
    echo "✗ FAIL: Pronoun reference was not grounded"
fi
echo ""

sleep 1

# Turn 4: Topic switch
echo "[Turn 4] What was my graduate GPA?"
echo "----------------------------------------"
RESPONSE4=$(curl -s -X POST "$API_URL" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What was my graduate GPA?\", \"session_id\": \"$SESSION_ID\"}")

if command -v jq &> /dev/null; then
    echo "Answer: $(echo "$RESPONSE4" | jq -r '.answer' | head -c 200)..."
    GROUNDED4=$(echo "$RESPONSE4" | jq -r '.grounded')
else
    echo "$RESPONSE4" | head -c 300
    GROUNDED4=$(echo "$RESPONSE4" | grep -o '"grounded":[^,}]*' | cut -d':' -f2)
fi

echo ""
echo "Grounded: $GROUNDED4"
echo ""

# Summary
echo "================================"
echo "SUMMARY"
echo "================================"
echo "Turn 1 (Context):     Grounded=$GROUNDED"
echo "Turn 2 (Short):       Grounded=$GROUNDED2"
echo "Turn 3 (Pronoun):     Grounded=$GROUNDED3"
echo "Turn 4 (Topic Switch): Grounded=$GROUNDED4"
echo ""

if [ "$GROUNDED2" = "true" ] && [ "$GROUNDED3" = "true" ]; then
    echo "✓ Multi-turn conversation WORKING"
else
    echo "✗ Multi-turn conversation has ISSUES"
fi
