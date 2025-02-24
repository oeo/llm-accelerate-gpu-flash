# DeepSeek R1 32B API Server

A FastAPI-based server implementation for the DeepSeek R1 Distill Qwen 32B language model, providing OpenAI-compatible API endpoints for chat completions and text generation.

## Features

- OpenAI-compatible API endpoints (`/v1/chat/completions`, `/v1/completions`)
- Streaming and non-streaming response support
- Multi-GPU support with optimized memory management
- 4-bit quantization for efficient memory usage
- Flash Attention 2 support
- Automatic memory cleanup and optimization
- Health monitoring endpoint

## Requirements

- Python 3.8+
- CUDA-capable GPU(s) with at least 24GB VRAM
- 64GB+ System RAM

## Installation

1. Clone the repository:
```bash
git clone
cd
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment Variables

The following environment variables are pre-configured in the server:

```python
TORCH_CUDA_ARCH_LIST=8.0
CUDA_VISIBLE_DEVICES=0,1,2,3
ACCELERATE_USE_DEVICE_MAP=True
TORCH_DISTRIBUTED_DEBUG=INFO
CUDA_LAUNCH_BLOCKING=0
CUBLAS_WORKSPACE_CONFIG=:16:8
ACCELERATE_USE_CPU_OFFLOAD=True
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Usage

1. Start the server:
```bash
python server.py
```

The server will start on `http://0.0.0.0:8000` by default.

2. API Endpoints:

- List available models:
```bash
GET /v1/models
```

- Chat completions:
```bash
POST /v1/chat/completions
```

- Text completions:
```bash
POST /v1/completions
```

- Health check:
```bash
GET /health
```

## API Examples

### Chat Completion

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "deepseek-r1-distil-32b",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "temperature": 0.7,
        "stream": False
    }
)
print(response.json())
```

### Streaming Chat Completion

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "deepseek-r1-distil-32b",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "temperature": 0.7,
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

## Configuration

The server includes several configurable parameters in the code:

- Model configurations (temperature, top_p, max_tokens, etc.)
- Server host and port
- GPU memory allocation
- Generation parameters
- Quantization settings

## Monitoring

Use the `/health` endpoint to monitor:
- GPU memory usage
- System memory usage
- CPU usage
- Model status
- Device mapping

## License

MIT

## Acknowledgments

- DeepSeek AI for the DeepSeek R1 Distill Qwen 32B model
- Hugging Face Transformers library
- FastAPI framework

