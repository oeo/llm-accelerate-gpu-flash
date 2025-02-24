#!/bin/bash

# Script to run a smaller model
# This model requires only 1 GPU

# Source the template
source "$(dirname "$0")/model-template.sh"

# Model configuration
MODEL_NAME="Smaller-Model"  # Replace with actual model name
MODEL_PORT=8001  # Using different port
GPU_IDS="2"  # Using GPU 2
NUMA_NODE="0"  # This GPU is on NUMA node 0
VRAM_PER_GPU=8  # Need ~8GB VRAM
NUM_CPU_CORES=12  # Using 12 cores from NUMA node 0

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

# Start model in background
start_model "smaller_model_server.py" &  # Replace with actual server script
SERVER_PID=$!

# Start monitoring in background
monitor_model_gpus &
MONITOR_PID=$!

# Wait for Ctrl+C
echo "Press Ctrl+C to stop the server and monitoring"
trap "kill $SERVER_PID $MONITOR_PID; exit" INT TERM
wait 