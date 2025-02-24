#!/bin/bash

# Monitoring Script for LLM Server
# This script provides real-time monitoring of GPU and system performance

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check if required tools are available
check_requirements() {
  if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found${NC}"
    exit 1
  fi
  
  if ! command -v curl &> /dev/null; then
    echo -e "${RED}Error: curl not found${NC}"
    exit 1
  fi
}

# Function to monitor GPU status
monitor_gpu() {
  echo -e "\n${GREEN}GPU Status:${NC}"
  nvidia-smi --query-gpu=index,gpu_name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader
}

# Function to monitor PCIe bandwidth
monitor_pcie() {
  echo -e "\n${GREEN}PCIe Status:${NC}"
  nvidia-smi topo -m
}

# Function to check server health
check_server_health() {
  echo -e "\n${GREEN}Server Health Check:${NC}"
  local response
  response=$(curl -s http://localhost:8000/health)
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}Server is running${NC}"
    echo "$response" | python3 -m json.tool
  else
    echo -e "${RED}Server is not responding${NC}"
  fi
}

# Function to monitor NUMA status
monitor_numa() {
  echo -e "\n${GREEN}NUMA Status:${NC}"
  numastat -n
}

# Function to monitor system resources
monitor_system() {
  echo -e "\n${GREEN}System Resources:${NC}"
  echo "CPU Usage:"
  top -bn1 | grep "Cpu(s)" | sed "s/.,%/\%/g"
  echo -e "\nMemory Usage:"
  free -h
}

# Main monitoring loop
monitor_loop() {
  while true; do
    clear
    echo -e "${YELLOW}LLM Server Monitoring${NC}"
    echo "Press Ctrl+C to exit"
    echo "================================"
    
    monitor_gpu
    monitor_system
    check_server_health
    
    # Only show these every 5 iterations to reduce noise
    if [ $((COUNTER % 5)) -eq 0 ]; then
      monitor_pcie
      monitor_numa
    fi
    
    COUNTER=$((COUNTER + 1))
    sleep 2
  done
}

# Main execution
check_requirements
COUNTER=0
monitor_loop 