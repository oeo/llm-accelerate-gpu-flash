#!/bin/bash

# Template script for running LLM models
# Source this script in your model-specific scripts

# Default values (override these in your model script)
MODEL_NAME="default"
MODEL_PORT=8000
GPU_IDS=""  # Comma-separated list of GPU IDs
NUMA_NODE="" # NUMA node to use
VRAM_PER_GPU=15 # GB of VRAM to use per GPU
NUM_CPU_CORES=12 # Number of CPU cores to use

# Function to validate GPU configuration
validate_gpus() {
  local gpu_list=$1
  local required_gpus=$(echo $gpu_list | tr ',' ' ' | wc -w)
  
  # Check if GPUs exist
  for gpu_id in $(echo $gpu_list | tr ',' ' '); do
    if ! nvidia-smi -i $gpu_id &>/dev/null; then
      echo "Error: GPU $gpu_id not found"
      return 1
    fi
  done
  
  # Check VRAM availability
  for gpu_id in $(echo $gpu_list | tr ',' ' '); do
    local free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu_id)
    if [ ${free_memory%.*} -lt $((VRAM_PER_GPU * 1024)) ]; then
      echo "Error: GPU $gpu_id does not have enough free VRAM (needs ${VRAM_PER_GPU}GB)"
      return 1
    fi
  done
  
  return 0
}

# Function to check if port is available
check_port() {
  local port=$1
  if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null; then
    echo "Error: Port $port is already in use"
    return 1
  fi
  return 0
}

# Function to set environment variables for the model
setup_env() {
  # GPU configuration
  export CUDA_VISIBLE_DEVICES=$GPU_IDS
  export CUDA_DEVICE_MAX_CONNECTIONS="1"
  export CUDA_DEVICE_ORDER="PCI_BUS_ID"
  
  # NUMA configuration
  export NUMA_GPU_NODE_PREFERRED=$NUMA_NODE
  
  # PyTorch optimization
  export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024,expandable_segments:True"
  export TORCH_DISTRIBUTED_DEBUG="INFO"
  export CUDA_LAUNCH_BLOCKING="0"
  export CUBLAS_WORKSPACE_CONFIG=":16:8"
  
  # Thread optimization
  export OMP_NUM_THREADS=$NUM_CPU_CORES
  export MKL_NUM_THREADS=$NUM_CPU_CORES
  
  # PCIe optimization
  export NCCL_P2P_LEVEL="5"
  export NCCL_IB_DISABLE="1"
  export NCCL_NET_GDR_LEVEL="5"
  export NCCL_SOCKET_IFNAME="^lo,docker,virbr"
  export NCCL_ALGO="Ring"
}

# Function to start model server
start_model() {
  local model_script=$1
  local numa_cmd=""
  
  if [ ! -z "$NUMA_NODE" ]; then
    numa_cmd="numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE"
  fi
  
  # Start the server
  echo "Starting $MODEL_NAME on GPUs: $GPU_IDS (NUMA node: $NUMA_NODE)"
  echo "Server will be available on port $MODEL_PORT"
  
  $numa_cmd python $model_script
}

# Function to monitor specific GPUs
monitor_model_gpus() {
  while true; do
    clear
    echo "Monitoring $MODEL_NAME GPUs: $GPU_IDS"
    echo "================================"
    
    # Show GPU stats for specified GPUs
    for gpu_id in $(echo $GPU_IDS | tr ',' ' '); do
      nvidia-smi -i $gpu_id --query-gpu=index,gpu_name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=table
    done
    
    sleep 2
  done
} 