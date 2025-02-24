from typing import Dict
import torch
from .base_config import ModelConfig

def create_balanced_device_map(num_layers: int, gpu_ids: list) -> Dict[str, str]:
    """Create a balanced device map for the model layers."""
    device_map = {}
    
    # Use CPU for all layers since no GPU is available
    device_map['model.embed_tokens'] = 'cpu'
    device_map['model.rotary_emb'] = 'cpu'
    
    # All layers go to CPU
    for i in range(num_layers):
        device_map[f'model.layers.{i}'] = 'cpu'
    
    # Final layers on CPU
    device_map['model.norm'] = 'cpu'
    device_map['lm_head'] = 'cpu'
    
    return device_map

# DeepSeek 32B Configuration (CPU)
deepseek_32b = ModelConfig(
    name="deepseek-r1-distil-32b",
    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    description="DeepSeek R1 32B Distilled Model",
    context_length=4096,
    device_map=create_balanced_device_map(64, ['cpu']),
    max_memory={
        "cpu": "128GB"  # EPYC has plenty of RAM
    }
)

# DeepSeek 14B Configuration (CPU)
deepseek_14b = ModelConfig(
    name="deepseek-r1-14b",
    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    description="DeepSeek R1 14B Distilled Model",
    context_length=8192,
    device_map=create_balanced_device_map(32, ['cpu']),
    max_memory={
        "cpu": "64GB"
    },
    temperature=0.7,
    top_p=0.9
)

# DeepSeek 7B Configuration (CPU)
deepseek_7b = ModelConfig(
    name="deepseek-r1-7b",
    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    description="DeepSeek R1 7B Distilled Model",
    context_length=16384,
    device_map=create_balanced_device_map(16, ['cpu']),
    max_memory={
        "cpu": "32GB"
    },
    temperature=0.8,
    top_p=0.95
)

# Dictionary of available models
AVAILABLE_MODELS = {
    "deepseek-r1-distil-32b": deepseek_32b,
    "deepseek-r1-14b": deepseek_14b,
    "deepseek-r1-7b": deepseek_7b
} 