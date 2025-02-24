#!/bin/bash

# Server Startup Script with NUMA Optimizations
# This script starts the LLM server with optimal NUMA and CPU settings

# Environment variables for GPU optimization
export CUDA_VISIBLE_DEVICES="0,1"  # Use only GPUs 0 and 1 on NUMA node 1
export CUDA_DEVICE_MAX_CONNECTIONS="1"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NUMA_GPU_NODE_PREFERRED="1"  # Prefer NUMA node 1

# PyTorch optimization variables
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024,expandable_segments:True"
export TORCH_DISTRIBUTED_DEBUG="INFO"
export CUDA_LAUNCH_BLOCKING="0"
export CUBLAS_WORKSPACE_CONFIG=":16:8"

# Thread and CPU optimization
export OMP_NUM_THREADS=12  # Number of cores in NUMA node 1
export MKL_NUM_THREADS=12
export GOMP_CPU_AFFINITY="6-11,30-35"  # CPU cores in NUMA node 1
export OMP_PROC_BIND="close"
export OMP_PLACES="cores"

# PCIe and NUMA optimization
export NCCL_P2P_LEVEL="5"
export NCCL_IB_DISABLE="1"
export NCCL_NET_GDR_LEVEL="5"
export NCCL_SOCKET_IFNAME="^lo,docker,virbr"
export NCCL_ALGO="Ring"

# Function to check NUMA configuration
check_numa() {
  if ! command -v numactl &> /dev/null; then
    echo "Warning: numactl not found. Continuing without NUMA optimization."
    return 1
  fi
  
  echo "NUMA configuration:"
  numactl --hardware || true
}

# Function to optimize system settings
optimize_system() {
  # Run GPU optimization script if it exists
  if [ -f "./scripts/gpu-optimize.sh" ]; then
    echo "Running GPU optimization script..."
    bash ./scripts/gpu-optimize.sh || true
  fi
}

# Function to start the server
start_server() {
  echo "Starting LLM server with NUMA optimizations..."
  
  # Use numactl to bind to NUMA node 1 if available
  if command -v numactl &> /dev/null; then
    numactl --cpunodebind=1 --membind=1 python server.py
  else
    python server.py
  fi
}

# Main execution
echo "Initializing LLM server..."

# Run checks
check_numa

# Apply optimizations
optimize_system

# Start server
start_server 