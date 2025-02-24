#!/bin/bash

# Script to run DeepSeek-R1-Distill-32B model
# This model requires 2 GPUs on the same NUMA node

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Source the template
source "$PROJECT_ROOT/scripts/models/model-template.sh"

# Model configuration
MODEL_NAME="DeepSeek-R1-Distill-32B"
MODEL_PORT=8000
GPU_IDS="0,1"  # Using GPUs 0 and 1
NUMA_NODE="1"  # These GPUs are on NUMA node 1
VRAM_PER_GPU=15  # Need ~15GB per GPU
NUM_CPU_CORES=12  # Using 12 cores from NUMA node 1

# Validate configuration
if ! validate_gpus "$GPU_IDS"; then
  echo "GPU validation failed"
  exit 1
fi

if ! check_port "$MODEL_PORT"; then
  echo "Port validation failed"
  exit 1
fi

# Setup environment
setup_env

# Change to project root directory
cd "$PROJECT_ROOT"

# Start model in background
start_model "server.py" &
SERVER_PID=$!

# Start monitoring in background
monitor_model_gpus &
MONITOR_PID=$!

# Wait for Ctrl+C
echo "Press Ctrl+C to stop the server and monitoring"
trap "kill $SERVER_PID $MONITOR_PID; exit" INT TERM
wait 