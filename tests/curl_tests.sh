#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Base URL
BASE_URL="http://localhost:8000"

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function to print test results
print_result() {
  if [ $1 -eq 0 ]; then
    echo -e "${GREEN}✓ $2${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
  else
    echo -e "${RED}✗ $2${NC}"
    echo -e "${RED}Error: $3${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
  fi
}

echo "Running API Tests..."
echo "===================="

# Test 1: Health Check
echo "Testing GET /health"
RESPONSE=$(curl -s "${BASE_URL}/health")
echo $RESPONSE | grep -q "status.*ok"
print_result $? "Health Check" "$RESPONSE"

# Test 2: List Models
echo "Testing GET /v1/models"
RESPONSE=$(curl -s "${BASE_URL}/v1/models")
echo $RESPONSE | grep -q "object"
print_result $? "List Models" "$RESPONSE"

# Test 3: Get Specific Model
echo "Testing GET /v1/models/deepseek-r1-distil-8b"
RESPONSE=$(curl -s "${BASE_URL}/v1/models/deepseek-r1-distil-8b")
echo $RESPONSE | grep -q "deepseek-r1-distil-8b"
print_result $? "Get Model Info" "$RESPONSE"

# Test 4: Load Model
echo "Testing POST /v1/models/deepseek-r1-distil-8b/load"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/models/deepseek-r1-distil-8b/load")
echo $RESPONSE | grep -q "success"
print_result $? "Load Model" "$RESPONSE"

# Test 5: Basic Chat Completion
echo "Testing POST /v1/chat/completions"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-distil-8b",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "temperature": 0.7
  }')
echo $RESPONSE | grep -q "content"
print_result $? "Basic Chat Completion" "$RESPONSE"

# Test 6: Streaming Chat Completion
echo "Testing POST /v1/chat/completions (streaming)"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-distil-8b",
    "messages": [{"role": "user", "content": "Count to 5."}],
    "stream": true
  }')
echo $RESPONSE | grep -q "content"
print_result $? "Streaming Chat Completion" "$RESPONSE"

# Test 7: Error Handling - Invalid Model
echo "Testing Error Handling - Invalid Model"
RESPONSE=$(curl -s "${BASE_URL}/v1/models/invalid-model")
echo $RESPONSE | grep -q "404"
print_result $? "Invalid Model Error" "$RESPONSE"

# Test 8: Error Handling - Invalid Request
echo "Testing Error Handling - Invalid Request"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-distil-8b",
    "invalid_field": true
  }')
echo $RESPONSE | grep -q "error"
print_result $? "Invalid Request Error" "$RESPONSE"

# Test 9: Chat Completion with Base64 Image
echo "Testing POST /v1/chat/completions with base64 image"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-distil-8b",
    "messages": [{
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?"
        },
        {
          "type": "image",
          "image_url": {
            "url": "data:image/jpeg;base64,'$(base64 tests/sample.jpg)'"
          }
        }
      ]
    }],
    "temperature": 0.7
  }')
echo $RESPONSE | grep -q "content"
print_result $? "Chat Completion with Base64 Image" "$RESPONSE"

# Test 10: Chat Completion with Image URL
echo "Testing POST /v1/chat/completions with image URL"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-distil-8b",
    "messages": [{
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?"
        },
        {
          "type": "image",
          "image_url": "https://example.com/sample.jpg"
        }
      ]
    }],
    "temperature": 0.7
  }')
echo $RESPONSE | grep -q "content"
print_result $? "Chat Completion with Image URL" "$RESPONSE"

# Test 11: Error Handling - Invalid Image URL
echo "Testing Error Handling - Invalid Image URL"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-distil-8b",
    "messages": [{
      "role": "user",
      "content": [
        {
          "type": "image",
          "image_url": "https://invalid-url/nonexistent.jpg"
        }
      ]
    }]
  }')
echo $RESPONSE | grep -q "error"
print_result $? "Invalid Image URL Error" "$RESPONSE"

# Test 12: Unload Model
echo "Testing POST /v1/models/deepseek-r1-distil-8b/unload"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/models/deepseek-r1-distil-8b/unload")
echo $RESPONSE | grep -q "success"
print_result $? "Unload Model" "$RESPONSE"

# Print Summary
echo "===================="
echo "Tests Complete!"
echo "Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo "Failed: ${RED}${TESTS_FAILED}${NC}"
echo "Total: $((TESTS_PASSED + TESTS_FAILED))"

# Exit with failure if any tests failed
[ $TESTS_FAILED -eq 0 ] || exit 1 