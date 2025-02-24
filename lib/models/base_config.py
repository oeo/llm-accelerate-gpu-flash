from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch
import time

@dataclass
class ModelConfig:
    name: str
    model_id: str
    description: str
    context_length: int
    revision: str = "main"
    torch_dtype: torch.dtype = torch.bfloat16
    device_map: Optional[Dict[str, int]] = None
    device: Optional[str] = None
    auto_load: bool = False  # Whether to load this model on server startup
    
    # Persistence settings
    persistent: bool = True  # Whether to keep the model loaded
    idle_timeout: int = 3600  # Seconds to wait before unloading (if not persistent)
    last_used: float = 0  # Timestamp of last usage
    unload_after_use: bool = False  # Whether to unload immediately after use
    
    # MoE settings
    is_moe: bool = False  # Whether this is a Mixture of Experts model
    num_experts: int = 8  # Number of experts (for MoE models)
    num_experts_per_token: int = 2  # Number of experts to use per token
    expert_stride: int = 1  # Stride for expert routing
    
    # Generation settings
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 2048
    
    # Stop tokens
    stop_tokens: List[str] = field(default_factory=lambda: [
        "<|endoftext|>",
        "<|end|>",
        "<|user|>",
        "<|assistant|>"
    ])
    
    # Optimization settings
    attn_implementation: str = "flash_attention_2"
    use_compile: bool = True
    use_cache: bool = True
    
    def should_unload(self, current_time: float) -> bool:
        """Check if model should be unloaded based on persistence settings."""
        if self.persistent:
            return False
        if self.unload_after_use:
            return True
        return (current_time - self.last_used) > self.idle_timeout
    
    def update_last_used(self):
        """Update the last used timestamp."""
        self.last_used = time.time()
    
    def to_model_kwargs(self) -> Dict:
        kwargs = {
            "trust_remote_code": True,
            "use_cache": self.use_cache
        }
        
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
            
        # Don't include device_map in kwargs if device is set to "auto"
        if self.device != "auto":
            kwargs["device_map"] = self.device_map
            
        # Add MoE-specific settings
        if self.is_moe:
            kwargs.update({
                "num_experts": self.num_experts,
                "num_experts_per_token": self.num_experts_per_token,
                "expert_stride": self.expert_stride
            })
            
        return kwargs
        
    def to_generation_config(self) -> Dict:
        """Convert model config to generation config dictionary."""
        config = {
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": 1.1,
            "use_cache": self.use_cache
        }
        
        # Add MoE-specific generation settings
        if self.is_moe:
            config.update({
                "num_experts_per_token": self.num_experts_per_token,
                "expert_stride": self.expert_stride
            })
            
        return config 