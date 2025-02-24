#!/bin/bash

# GPU Optimization Script for LLM Server
# This script optimizes GPU settings for inference workload

# Function to check if we have sudo access
check_sudo() {
  if ! sudo -v; then
    echo "This script requires sudo privileges to set GPU parameters"
    exit 1
  fi
}

# Function to get total number of GPUs
get_gpu_count() {
  nvidia-smi --query-gpu=gpu_index --format=csv,noheader | wc -l
}

# Function to get NUMA node for a GPU
get_gpu_numa_node() {
  local gpu_id=$1
  local bus_id=$(nvidia-smi -i $gpu_id --query-gpu=pci.bus_id --format=csv,noheader)
  local numa_node=$(cat /sys/class/pci_bus/$(echo $bus_id | cut -d ':' -f2-3 | tr '.' '/')/device/numa_node)
  echo $numa_node
}

# Function to optimize GPU settings
optimize_gpu() {
  local gpu_id=$1
  local numa_node=$(get_gpu_numa_node $gpu_id)
  
  echo "Optimizing GPU $gpu_id (NUMA node $numa_node)..."
  
  # Set maximum performance mode
  sudo nvidia-smi -i $gpu_id -pm 1
  
  # Set power limit to 60W for A2 GPUs
  sudo nvidia-smi -i $gpu_id -pl 60
  
  # Set optimal memory and graphics clocks for A2
  sudo nvidia-smi -i $gpu_id -ac 6251,1770
  
  # Set compute mode to default (allows multiple processes)
  sudo nvidia-smi -i $gpu_id --compute-mode=DEFAULT
  
  # Enable persistence mode
  sudo nvidia-smi -i $gpu_id -pm ENABLED
  
  echo "GPU $gpu_id optimization complete"
}

# Function to check PCIe settings
check_pcie() {
  echo -e "\nChecking PCIe settings..."
  nvidia-smi -q | grep -A 4 "Max Link"
  
  echo -e "\nGPU Topology:"
  nvidia-smi topo -m
}

# Function to apply system optimizations
apply_system_optimizations() {
  echo "Applying system optimizations..."
  
  # Set maximum I/O priority
  sudo ionice -c 1 -n 0 -p $$
  
  # Disable CPU frequency scaling (requires root)
  if [ -w "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor" ]; then
    echo "Setting CPU governor to performance..."
    for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
      echo "performance" | sudo tee $governor
    done
  fi
  
  # Increase system limits
  sudo sysctl -w vm.swappiness=10
  sudo sysctl -w kernel.numa_balancing=0
}

# Function to show GPU memory stats
show_gpu_stats() {
  echo -e "\nGPU Memory Status:"
  nvidia-smi --query-gpu=index,gpu_name,memory.total,memory.free,memory.used --format=table
}

# Main execution
echo "Starting GPU optimization..."
check_sudo

# Get total number of GPUs
GPU_COUNT=$(get_gpu_count)
echo "Found $GPU_COUNT GPUs"

# Show initial GPU status
show_gpu_stats

# Optimize all GPUs
for gpu_id in $(seq 0 $((GPU_COUNT-1))); do
  optimize_gpu $gpu_id
done

# Apply system optimizations
apply_system_optimizations

# Check PCIe settings and show topology
check_pcie

# Show final GPU status
echo -e "\nFinal GPU Status:"
show_gpu_stats

echo "GPU optimization completed!"
echo "NUMA node assignments:"
for gpu_id in $(seq 0 $((GPU_COUNT-1))); do
  numa_node=$(get_gpu_numa_node $gpu_id)
  echo "GPU $gpu_id -> NUMA node $numa_node"
done 