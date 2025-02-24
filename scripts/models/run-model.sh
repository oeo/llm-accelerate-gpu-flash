#!/bin/bash

# Generic Model Runner Script
# Usage: ./run-model.sh --model MODEL_NAME [--port PORT] [--threads THREADS]

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Default values
MODEL_NAME=""
MODEL_PORT=8000
NUM_THREADS=$(nproc)  # Default to all CPU threads

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --port)
      MODEL_PORT="$2"
      shift 2
      ;;
    --threads)
      NUM_THREADS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Validate model name
if [ -z "$MODEL_NAME" ]; then
  echo "Error: Model name is required"
  echo "Usage: $0 --model MODEL_NAME [--port PORT] [--threads THREADS]"
  echo "Available models:"
  python3 -c "from lib.models import AVAILABLE_MODELS; print('\n'.join(f'  - {m}' for m in AVAILABLE_MODELS.keys()))"
  exit 1
fi

# Function to check if port is available
check_port() {
  if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null; then
    echo "Error: Port $1 is already in use"
    return 1
  fi
  return 0
}

# Function to setup environment
setup_env() {
  # CPU optimization
  export OMP_NUM_THREADS=$NUM_THREADS
  export MKL_NUM_THREADS=$NUM_THREADS
  export OPENBLAS_NUM_THREADS=$NUM_THREADS
  export VECLIB_MAXIMUM_THREADS=$NUM_THREADS
  export NUMEXPR_NUM_THREADS=$NUM_THREADS

  # PyTorch optimization
  export TORCH_CPU_PARALLELISM=$NUM_THREADS
  export TORCH_COMPILE_THREADS=$NUM_THREADS
  export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
  
  # Server configuration
  export MODEL_NAME=$MODEL_NAME
  export PORT=$MODEL_PORT
}

# Function to start model
start_model() {
  echo "Starting $MODEL_NAME on port $MODEL_PORT with $NUM_THREADS threads"
  cd "$PROJECT_ROOT"
  python3 server.py
}

# Function to monitor system resources
monitor_resources() {
  while true; do
    clear
    echo "Monitoring $MODEL_NAME"
    echo "=================="
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)" | sed "s/.,%/\%/g"
    echo
    echo "Memory Usage:"
    free -h
    echo
    echo "Process Info:"
    ps aux | grep python | grep -v grep
    sleep 5
  done
}

# Main execution
if ! check_port "$MODEL_PORT"; then
  exit 1
fi

setup_env

# Start model in background
start_model &
SERVER_PID=$!

# Start monitoring in background
monitor_resources &
MONITOR_PID=$!

# Wait for Ctrl+C
echo "Press Ctrl+C to stop the server and monitoring"
trap "kill $SERVER_PID $MONITOR_PID; exit" INT TERM
wait 