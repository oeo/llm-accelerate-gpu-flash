# DeepSeek R1 LLM Server

A FastAPI-based server implementation for DeepSeek R1 language models, providing OpenAI-compatible API endpoints for chat completions and text generation. This server supports multiple models with optimized GPU memory management and automatic model loading.

## Features

- OpenAI-compatible API endpoints (`/v1/chat/completions`)
- Multi-GPU support with optimized memory management
- Automatic model loading on startup
- 8-bit quantization for efficient memory usage
- Flash Attention 2 support
- Real-time monitoring and health checks
- NUMA-aware optimizations
- Streaming and non-streaming responses
- Optimized for NVIDIA A2 GPUs
- Automatic device mapping for multi-GPU setups
- Comprehensive monitoring and diagnostics
- Production-ready error handling

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Models](#running-models)
- [API Usage](#api-usage)
- [Monitoring](#monitoring)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Required software:
- Python 3.8+
- PyTorch 2.2.0+
- CUDA-capable GPUs (4x NVIDIA A2 16GB recommended)
- 64GB+ System RAM

Optional but recommended:
- NVIDIA Container Toolkit (for Docker deployment)
- NUMA-enabled system
- PCIe Gen4 support

## Installation

1. Clone the repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify GPU setup:
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
```

4. Run system optimization:
```bash
sudo ./scripts/gpu-optimize.sh
```

## Configuration

### Directory Structure

```
.
├── scripts/
│   ├── gpu-optimize.sh     # GPU optimization script
│   ├── start-server.sh     # Server startup script with NUMA optimizations
│   ├── monitor.sh          # Standalone monitoring script
│   └── models/
│       ├── model-template.sh  # Base template for model scripts
│       └── run-model.sh      # Generic model runner script
├── lib/
│   └── models/            # Model library and configurations
│       ├── __init__.py
│       ├── base_config.py
│       ├── model_configs.py
│       └── model_manager.py
├── config.yml             # Main configuration file
├── server.py             # Main server implementation
└── requirements.txt      # Python dependencies
```

### Configuration File (config.yml)

The `config.yml` file controls all aspects of the server and models:

```yaml
# Server Configuration
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  auto_load_enabled: true  # Enable automatic model loading

# GPU Configuration
gpu:
  devices: [0, 1, 2, 3]  # Using all 4 GPUs
  memory_per_gpu: "15GB"  # Leave 1GB headroom
  optimization:
    torch_compile: true
    mixed_precision: "bf16"
    attn_implementation: "flash_attention_2"

# Model Configurations
models:
  deepseek-r1-distil-32b:
    name: "DeepSeek-R1-Distill-32B"
    auto_load: false  # Don't load on startup
    device_map: {...}  # Distributed across GPUs
    
  deepseek-r1-distil-14b:
    name: "DeepSeek-R1-Distill-14B"
    auto_load: true   # Load on startup
    device: "auto"    # Automatic device mapping
    
  deepseek-r1-distil-8b:
    name: "DeepSeek-R1-Distill-8B"
    auto_load: true   # Load on startup
    device: 0         # Single GPU
```

### Auto-Loading Models

Models can be configured to load automatically when the server starts:

1. Set `auto_load_enabled: true` in server config
2. Configure `auto_load: true` for specific models
3. Models will load during server initialization

### Environment Variables

The server respects the following environment variables:

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES="0,1,2,3"
CUDA_DEVICE_MAX_CONNECTIONS="1"
CUDA_DEVICE_ORDER="PCI_BUS_ID"

# PyTorch Optimization
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
TORCH_DISTRIBUTED_DEBUG="INFO"
TORCH_SHOW_CPP_STACKTRACES="1"

# NUMA Configuration
NUMA_GPU_NODE_PREFERRED="1"
```

## Running Models

### Using the Start Script

The `start-server.sh` script provides optimized startup:

```bash
# Start server with NUMA optimizations
./scripts/start-server.sh
```

### Using the Model Runner

The `run-model.sh` script provides flexible model management:

```bash
# Basic usage
./scripts/models/run-model.sh --model MODEL_NAME

# Examples:
# Run 8B model with custom port
./scripts/models/run-model.sh --model deepseek-r1-distil-8b --port 8001

# Run 14B model with specific thread count
./scripts/models/run-model.sh --model deepseek-r1-distil-14b --threads 32
```

### GPU Optimization

Run the GPU optimization script before starting the server:

```bash
# Optimize GPU settings
sudo ./scripts/gpu-optimize.sh
```

This script:
- Sets optimal GPU clock speeds
- Configures power limits
- Optimizes PCIe settings
- Sets up NUMA affinities

### Model Memory Requirements

Recommended GPU configurations:

1. **DeepSeek-R1-Distill-32B**
   - Distributed across 4 GPUs
   - ~15GB per GPU
   - 4K context length
   - Best for high-accuracy tasks
   - Recommended for: Complex reasoning, code generation

2. **DeepSeek-R1-Distill-14B**
   - Auto device mapping
   - ~30GB total VRAM
   - 8K context length
   - Good balance of performance/memory
   - Recommended for: General use, balanced performance

3. **DeepSeek-R1-Distill-8B**
   - Single GPU
   - ~15GB VRAM
   - 16K context length
   - Fastest inference
   - Recommended for: High-throughput, real-time applications

## API Usage

### List Available Models
```bash
curl http://localhost:8000/v1/models
```

### Load a Model
```bash
curl -X POST http://localhost:8000/v1/models/deepseek-r1-distil-8b/load
```

### Generate Response
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-distil-8b",
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
    "model": "deepseek-r1-distil-8b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### API Parameters

The chat completion endpoint accepts the following parameters:

```json
{
  "model": "string",          // Model ID to use
  "messages": [              // Array of messages
    {
      "role": "string",      // "system", "user", or "assistant"
      "content": "string"    // Message content
    }
  ],
  "temperature": 0.7,        // 0.0 to 2.0 (default: 0.7)
  "top_p": 0.95,            // 0.0 to 1.0 (default: 0.95)
  "max_tokens": 2048,        // Maximum tokens to generate
  "stream": false,           // Enable streaming responses
  "stop": ["string"]        // Optional array of stop sequences
}
```

## Monitoring

### Built-in Monitoring

The `/health` endpoint provides comprehensive monitoring:
```bash
curl http://localhost:8000/health
```

Returns:
- GPU memory usage per device
- System memory status
- Loaded models
- Device mapping
- Process statistics
- NUMA topology

### Monitoring Script

Use the dedicated monitoring script:
```bash
./scripts/monitor.sh
```

Features:
- Real-time GPU metrics
- Memory usage tracking
- NUMA status
- PCIe bandwidth
- Process information
- Temperature monitoring
- Power consumption

## Performance Optimization

### GPU Memory Optimization

1. **Memory Allocation**
   - Use 8-bit quantization
   - Enable gradient checkpointing
   - Implement proper cleanup

2. **Device Mapping**
   - Balance model layers across GPUs
   - Consider NUMA topology
   - Optimize for PCIe bandwidth

3. **Inference Settings**
   - Use Flash Attention 2
   - Enable torch.compile
   - Implement proper batching

### NUMA Optimization

1. **CPU Affinity**
   - Bind processes to NUMA nodes
   - Align with GPU placement
   - Optimize thread count

2. **Memory Access**
   - Local memory allocation
   - Minimize cross-node traffic
   - Monitor bandwidth

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**
   - Check GPU memory allocation in config
   - Use `nvidia-smi` to monitor usage
   - Consider using a smaller model
   - Check device mapping configuration

2. **Model Loading Failures**
   - Verify auto-load settings
   - Check GPU memory availability
   - Review server logs for errors
   - Ensure correct device mapping

3. **Performance Issues**
   - Run GPU optimization script
   - Check NUMA configuration
   - Monitor PCIe bandwidth
   - Review thread settings

### Best Practices

1. **Memory Management**
   - Leave 1GB headroom per GPU
   - Use appropriate device mapping
   - Monitor memory usage
   - Clean up unused models

2. **GPU Optimization**
   - Run optimization script before starting
   - Use NUMA-aware configurations
   - Monitor GPU temperatures
   - Check PCIe bandwidth

3. **Model Selection**
   - Use 32B for high accuracy (4K context)
   - Use 14B for balanced performance (8K context)
   - Use 8B for speed (16K context)
   - Consider auto-loading frequently used models

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

### Development Setup

1. Fork the repository
2. Create a virtual environment
3. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```
4. Run tests:
```bash
pytest tests/
```

## License

MIT

## Acknowledgments

- DeepSeek AI for the DeepSeek R1 models
- Hugging Face Transformers library
- FastAPI framework
- NVIDIA for GPU optimization guidance

