from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch

@dataclass
class ModelConfig:
    name: str
    model_id: str
    description: str
    context_length: int
    revision: str = "main"
    torch_dtype: torch.dtype = torch.bfloat16
    device_map: Optional[Dict[str, int]] = None
    max_memory: Optional[Dict[str, str]] = None
    
    # Generation settings
    temperature: float = 0.3
    top_p: float = 0.85
    max_new_tokens: int = 2048
    
    # Stop tokens
    stop_tokens: List[str] = field(default_factory=lambda: [
        "<|begin▁of▁sentence|>",
        "<|end▁of▁sentence|>",
        "<|User|>",
        "<|Assistant|>",
        "<think>",
        "</think>"
    ])
    
    # Flash attention and other optimizations
    use_flash_attention: bool = True
    use_bettertransformer: bool = True
    use_compile: bool = True
    
    # Memory and performance settings
    low_cpu_mem_usage: bool = True
    use_cache: bool = True
    
    def to_generation_config(self) -> Dict:
        return {
            "use_cache": self.use_cache,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": 0,
            "eos_token_id": 0,
            "num_return_sequences": 1
        }
    
    def to_model_kwargs(self) -> Dict:
        kwargs = {
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "trust_remote_code": True,
            "use_cache": self.use_cache
        }
        
        if self.device_map:
            kwargs["device_map"] = self.device_map
        
        if self.max_memory:
            kwargs["max_memory"] = self.max_memory
            
        if self.use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"
            
        return kwargs 