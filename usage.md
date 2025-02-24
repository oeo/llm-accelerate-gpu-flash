# LLM Server Usage Guide

This guide explains how to use the multi-model LLM server with CPU optimizations.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Running Models](#running-models)
- [API Usage](#api-usage)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Required software:
- Python 3.8+
- PyTorch
- 64GB+ System RAM (varies by model)
- High core count CPU (AMD EPYC recommended)

## Configuration

### Directory Structure

```
.
├── scripts/
│   ├── models/
│   │   └── run-model.sh    # Generic model runner script
│   └── monitor.sh          # Standalone monitoring script
├── lib/
│   └── models/            # Model library and configurations
│       ├── __init__.py
│       ├── base_config.py
│       ├── model_configs.py
│       └── model_manager.py
├── config.yml             # Main configuration file
├── server.py              # Main server implementation
└── requirements.txt       # Python dependencies
```

### Configuration File (config.yml)

The `config.yml` file controls all aspects of the server and models:

```yaml
# Server Configuration
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1

# CPU Configuration
cpu:
  threads: 40  # EPYC 40 cores
  memory: "256GB"  # Total system memory

# Model Configurations
models:
  deepseek-r1-distil-32b:
    name: "DeepSeek-R1-Distill-32B"
    model_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    description: "DeepSeek R1 32B Distilled Model"
    context_length: 4096
    num_layers: 64
    device: "cpu"
    memory: "128GB"
    generation:
      temperature: 0.3
      top_p: 0.85
      max_tokens: 2048
```

To add or modify models:
1. Edit `config.yml`
2. Add/modify model configuration under the `models` section
3. Restart the server to apply changes

## Running Models

### Using the Model Runner

The `run-model.sh` script provides a flexible way to run any configured model:

```bash
# Basic usage
./scripts/models/run-model.sh --model MODEL_NAME

# Examples:
# Run 7B model with 32 threads (leaving some cores for system)
./scripts/models/run-model.sh --model deepseek-r1-7b --threads 32

# Run 14B model on different port
./scripts/models/run-model.sh --model deepseek-r1-14b --port 8001

# Run 32B model with all cores
./scripts/models/run-model.sh --model deepseek-r1-distil-32b --threads 40
```

Options:
- `--model`: Model name from config.yml (required)
- `--port`: Server port (default: 8000)
- `--threads`: Number of CPU threads to use (default: 40)

### Model Memory Requirements

Each model requires different amounts of system memory:

1. **DeepSeek-R1-Distill-32B**
   - ~128GB RAM
   - Best for high-accuracy tasks
   - 4K context length

2. **DeepSeek-R1-14B**
   - ~64GB RAM
   - Good balance of performance/memory
   - 8K context length

3. **DeepSeek-R1-7B**
   - ~32GB RAM
   - Fastest inference
   - 16K context length

## API Usage

### List Available Models
```bash
curl http://localhost:8000/v1/models
```

### Load a Model
```bash
curl -X POST http://localhost:8000/v1/models/deepseek-r1-7b/load
```

### Generate Response
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-7b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "stream": false
  }'
```

### Stream Response
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-7b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Monitoring

The model runner includes built-in monitoring:
- CPU usage per thread
- Memory usage and allocation
- Process information
- Model status

Additional monitoring via API:
```bash
curl http://localhost:8000/health
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Check model memory requirements
   - Reduce `num_threads` to lower memory usage
   - Consider using a smaller model

2. **Poor Performance**
   - Ensure enough CPU threads are allocated
   - Check CPU usage with monitoring
   - Consider adjusting thread count

3. **Port Conflicts**
   - Use `--port` to specify different port
   - Check running processes with `lsof -i :PORT`

### Best Practices

1. **Memory Management**
   - Run one model per server instance
   - Monitor memory usage
   - Leave some RAM for system overhead

2. **CPU Optimization**
   - Use 32-36 threads for lighter models (leaving cores for system)
   - Use all 40 cores for 32B model when maximum performance is needed
   - Monitor CPU usage and temperature
   - Consider process priority for critical workloads

3. **Context Length**
   - Use 32B model for standard tasks (4K)
   - Use 14B model for medium context (8K)
   - Use 7B model for long context (16K) 