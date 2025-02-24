from typing import Dict
import yaml
import torch
from pathlib import Path
from .base_config import ModelConfig

def create_balanced_device_map(num_layers: int, device: str) -> Dict[str, str]:
    """Create a device map for the model layers."""
    device_map = {}
    
    # Map all components to the specified device
    device_map['model.embed_tokens'] = device
    device_map['model.rotary_emb'] = device
    
    for i in range(num_layers):
        device_map[f'model.layers.{i}'] = device
    
    device_map['model.norm'] = device
    device_map['lm_head'] = device
    
    return device_map

def load_model_configs() -> Dict[str, ModelConfig]:
    """Load model configurations from config.yml and augment with dynamic information."""
    # Load config.yml
    config_path = Path(__file__).parent.parent.parent / 'config.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    available_models = {}
    
    # Create ModelConfig objects for each model
    for model_id, model_config in config['models'].items():
        # Get device from config
        device = model_config.get('device', 'cpu')
        
        # Create device map
        device_map = create_balanced_device_map(
            num_layers=model_config['num_layers'],
            device=device
        )
        
        # Create memory config
        max_memory = {
            device: model_config['memory']
        }
        
        # Create ModelConfig object
        model = ModelConfig(
            name=model_config['name'],
            model_id=model_config['model_id'],
            description=model_config['description'],
            context_length=model_config['context_length'],
            device_map=device_map,
            max_memory=max_memory,
            temperature=model_config['generation']['temperature'],
            top_p=model_config['generation']['top_p'],
            max_new_tokens=model_config['generation']['max_tokens'],
            # Add optimization settings from config
            use_compile=config['optimization']['torch_compile'],
            use_flash_attention=False,  # Not supported on CPU
            use_bettertransformer=False,  # Not needed on CPU
            low_cpu_mem_usage=config['optimization']['low_cpu_mem_usage']
        )
        
        available_models[model_id] = model
    
    return available_models

# Load configurations from config.yml
AVAILABLE_MODELS = load_model_configs() 