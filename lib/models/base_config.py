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
    device: Optional[str] = None
    auto_load: bool = False  # Whether to load this model on server startup
    
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
            
        return kwargs 