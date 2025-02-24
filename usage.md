# LLM Server Usage Guide

This guide explains how to use the multi-model LLM server with optimal GPU and NUMA configurations.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Model Management](#model-management)
- [Running Models](#running-models)
- [API Usage](#api-usage)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Required software:
- NVIDIA drivers and CUDA toolkit
- Python 3.8+ with PyTorch
- `numactl` for NUMA optimization
- `nvidia-smi` for GPU management
- `lsof` for port checking

Required hardware:
- NVIDIA GPUs (A2 or better recommended)
- Multi-socket system with NUMA topology

## Configuration

### Directory Structure

```
.
├── bin/                    # Executable scripts for specific models
│   └── run-deepseek-32b.sh  # DeepSeek 32B model runner
├── lib/
│   └── models/            # Model library and configurations
│       ├── __init__.py
│       ├── base_config.py
│       ├── model_configs.py
│       └── model_manager.py
├── scripts/
│   ├── gpu-optimize.sh     # GPU optimization script
│   ├── monitor.sh         # Standalone monitoring script
│   └── models/            # Model script templates
├── config.yml            # Main configuration file
├── server.py             # Main server implementation
└── requirements.txt      # Python dependencies
```

### Configuration File (config.yml)

The `config.yml` file controls all aspects of the server:

```yaml
# Example model configuration
models:
  deepseek-r1-distil-32b:
    gpus: [0, 1]  # Uses GPUs 0 and 1
    context_length: 4096
    memory_per_gpu: "15GB"
    
  deepseek-r1-14b:
    gpus: [2]     # Uses GPU 2
    context_length: 8192
    memory_per_gpu: "15GB"
    
  deepseek-r1-7b:
    gpus: [3]     # Uses GPU 3
    context_length: 16384
    memory_per_gpu: "8GB"
```

To modify model configurations:
1. Edit `config.yml`
2. Restart the server to apply changes

## Model Management

### Available Models

Currently supported models:

1. **DeepSeek-R1-Distill-32B**
   - Uses 2 GPUs (0,1) on NUMA node 1
   - 4K context length
   - ~30GB VRAM total

2. **DeepSeek-R1-14B**
   - Uses 1 GPU (2) on NUMA node 0
   - 8K context length
   - ~15GB VRAM

3. **DeepSeek-R1-7B**
   - Uses 1 GPU (3) on NUMA node 0
   - 16K context length
   - ~8GB VRAM

### Loading Models

Models can be loaded via API:

```bash
# Load 32B model
curl -X POST http://localhost:8000/v1/models/deepseek-r1-distil-32b/load

# Load 14B model
curl -X POST http://localhost:8000/v1/models/deepseek-r1-14b/load

# Load 7B model
curl -X POST http://localhost:8000/v1/models/deepseek-r1-7b/load
```

### Unloading Models

Free GPU memory by unloading models:

```bash
curl -X POST http://localhost:8000/v1/models/deepseek-r1-14b/unload
```

## Running Models

### Starting the Server

1. First, optimize all GPUs:
```bash
sudo ./scripts/gpu-optimize.sh
```

2. Start the server:
```bash
./bin/run-deepseek-32b.sh
```

3. Monitor performance:
```bash
./scripts/monitor.sh
```

### Using Different Models

Each model can be run with different configurations:

1. **32B Model (High Performance)**
```bash
# Uses GPUs 0,1 for maximum performance
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-distil-32b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

2. **14B Model (Extended Context)**
```bash
# Uses GPU 2 with 8K context
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{
    "model": "deepseek-r1-14b",
    "messages": [{"role": "user", "content": "Long context here..."}],
    "max_tokens": 4096
  }'
```

3. **7B Model (Maximum Context)**
```bash
# Uses GPU 3 with 16K context
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{
    "model": "deepseek-r1-7b",
    "messages": [{"role": "user", "content": "Very long context..."}],
    "max_tokens": 8192
  }'
```

## API Usage

### List Available Models
```bash
curl http://localhost:8000/v1/models
```

### Get Model Status
```bash
curl http://localhost:8000/v1/models/deepseek-r1-14b
```

### Generate Response (Streaming)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-14b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Monitoring

### Real-time Monitoring

Use the monitoring script:
```bash
./scripts/monitor.sh
```

This shows:
- GPU utilization per model
- Memory usage
- Temperature
- Power consumption

### Health Check
```bash
curl http://localhost:8000/health
```

Shows:
- Loaded models
- GPU memory status
- System resources

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**
```bash
# Check GPU memory
nvidia-smi

# Unload unused models
curl -X POST http://localhost:8000/v1/models/MODEL_NAME/unload
```

2. **NUMA Issues**
```bash
# Check NUMA topology
numactl --hardware

# Verify GPU-NUMA mapping
nvidia-smi topo -m
```

3. **Performance Issues**
- Ensure models use GPUs on the same NUMA node
- Monitor PCIe bandwidth between GPUs
- Check GPU temperature and throttling

### Best Practices

1. **Memory Management**
- Load only needed models
- Unload unused models
- Monitor memory fragmentation

2. **NUMA Optimization**
- Keep related GPUs on same NUMA node
- Match CPU cores to GPU NUMA node
- Use appropriate thread counts

3. **Context Length**
- Use 32B model for standard tasks
- Use 14B model for medium context
- Use 7B model for maximum context

4. **Multiple Models**
- Balance GPU usage across NUMA nodes
- Monitor system resources
- Use appropriate batch sizes 