#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Base URL and configuration
BASE_URL="http://localhost:8000"
MODELS=(
  "deepseek-r1-distil-32b"
  "deepseek-r1-distil-14b"
  "mixtral-8x7b"
  "idefics-80b"
)

# Test prompts of varying lengths
declare -A PROMPTS=(
  ["short"]="Explain what is quantum computing."
  ["medium"]="Write a detailed analysis of the impact of artificial intelligence on modern society, covering both benefits and potential risks."
  ["long"]="Write a comprehensive essay about the history of computing, starting from early mechanical calculators through vacuum tubes, transistors, integrated circuits, and modern quantum computers. Include key innovations, important figures, and major milestones."
)

# Function to monitor GPU stats
monitor_gpus() {
  while true; do
    echo -e "\n${YELLOW}GPU Status:${NC}"
    nvidia-smi --query-gpu=index,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv,noheader
    sleep 2
  done
}

# Function to check server health
check_health() {
  local response
  response=$(curl -s "${BASE_URL}/health")
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}Server health check passed${NC}"
    echo "$response" | python3 -m json.tool
    return 0
  else
    echo -e "${RED}Server health check failed${NC}"
    return 1
  fi
}

# Function to load a model
load_model() {
  local model=$1
  echo -e "\n${YELLOW}Loading model: $model${NC}"
  local response
  response=$(curl -s -X POST "${BASE_URL}/v1/models/${model}/load")
  if echo "$response" | grep -q "success"; then
    echo -e "${GREEN}Successfully loaded $model${NC}"
    return 0
  else
    echo -e "${RED}Failed to load $model: $response${NC}"
    return 1
  fi
}

# Function to calculate tokens/sec
calculate_throughput() {
  local start_time=$1
  local end_time=$2
  local tokens=$3
  local duration=$(echo "$end_time - $start_time" | bc)
  local throughput=$(echo "scale=2; $tokens / $duration" | bc)
  echo "$throughput"
}

# Function to run inference test
run_inference_test() {
  local model=$1
  local prompt_type=$2
  local prompt="${PROMPTS[$prompt_type]}"
  
  echo -e "\n${YELLOW}Testing $model with $prompt_type prompt${NC}"
  
  local start_time=$(date +%s.%N)
  local response
  response=$(curl -s -X POST "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "'$model'",
      "messages": [{"role": "user", "content": "'$prompt'"}],
      "temperature": 0.7
    }')
  local end_time=$(date +%s.%N)
  
  if echo "$response" | grep -q "content"; then
    local input_tokens=$(echo "$response" | jq -r '.usage.prompt_tokens')
    local output_tokens=$(echo "$response" | jq -r '.usage.completion_tokens')
    local total_tokens=$(echo "$response" | jq -r '.usage.total_tokens')
    local throughput=$(calculate_throughput $start_time $end_time $total_tokens)
    
    echo -e "${GREEN}Success:${NC}"
    echo "Input tokens: $input_tokens"
    echo "Output tokens: $output_tokens"
    echo "Total tokens: $total_tokens"
    echo "Throughput: $throughput tokens/sec"
    return 0
  else
    echo -e "${RED}Failed: $response${NC}"
    return 1
  fi
}

# Function to run parallel inference
run_parallel_inference() {
  local model=$1
  local num_requests=$2
  local prompt_type=$3
  
  echo -e "\n${YELLOW}Running $num_requests parallel requests for $model${NC}"
  
  for i in $(seq 1 $num_requests); do
    run_inference_test "$model" "$prompt_type" &
  done
  wait
}

# Function to test rate limiting
test_rate_limiting() {
  local model=$1
  local requests_per_second=5
  local duration=30
  
  echo -e "\n${YELLOW}Testing rate limiting for $model${NC}"
  echo "Sending $requests_per_second requests per second for $duration seconds"
  
  local start_time=$(date +%s)
  local count=0
  local success=0
  local failed=0
  
  while [ $(($(date +%s) - start_time)) -lt $duration ]; do
    for i in $(seq 1 $requests_per_second); do
      count=$((count + 1))
      if run_inference_test "$model" "short" > /dev/null 2>&1; then
        success=$((success + 1))
      else
        failed=$((failed + 1))
      fi
    done
    sleep 1
  done
  
  echo -e "\n${GREEN}Rate Limiting Results:${NC}"
  echo "Total requests: $count"
  echo "Successful: $success"
  echo "Failed: $failed"
  echo "Success rate: $(echo "scale=2; ($success / $count) * 100" | bc)%"
}

# Main test sequence
main() {
  echo -e "${YELLOW}Starting Stress Test${NC}"
  
  # Start GPU monitoring in background
  monitor_gpus &
  MONITOR_PID=$!
  
  # Initial health check
  check_health || exit 1
  
  # Load all models
  for model in "${MODELS[@]}"; do
    load_model "$model" || exit 1
    sleep 5  # Wait for model to stabilize
  done
  
  # Test each model with different prompt lengths
  for model in "${MODELS[@]}"; do
    for prompt_type in "short" "medium" "long"; do
      run_inference_test "$model" "$prompt_type"
      sleep 2
    done
  done
  
  # Parallel inference tests
  for model in "${MODELS[@]}"; do
    run_parallel_inference "$model" 3 "medium"
    sleep 5
  done
  
  # Rate limiting tests
  for model in "${MODELS[@]}"; do
    test_rate_limiting "$model"
    sleep 5
  done
  
  # Final health check
  check_health
  
  # Stop GPU monitoring
  kill $MONITOR_PID
  
  echo -e "\n${GREEN}Stress Test Complete${NC}"
}

# Run main test sequence
main 