import torch
import logging
from typing import Dict, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from .base_config import ModelConfig
from .model_configs import AVAILABLE_MODELS

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.models: Dict[str, AutoModelForCausalLM] = {}
        self.tokenizers: Dict[str, AutoTokenizer] = {}
        self.configs: Dict[str, ModelConfig] = AVAILABLE_MODELS
        
        # Initialize accelerator with mixed precision
        self.accelerator = Accelerator(
            mixed_precision="bf16",  # Use bfloat16 mixed precision
            device_placement=True,
            kwargs_handlers=[self._get_device_map]
        )
        
        # Verify CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This implementation requires GPU.")
            
        logger.info(f"Using device: {self.accelerator.device}")
        logger.info(f"Mixed precision: {self.accelerator.mixed_precision}")
    
    def _get_device_map(self, config):
        """Get device map from config or create balanced one."""
        if hasattr(config, 'device_map') and config.device_map:
            return config.device_map
        elif hasattr(config, 'device') and config.device is not None:
            return config.device
        else:
            # Default to first GPU if no specific mapping
            return 0
    
    def load_model(self, model_name: str) -> bool:
        """Load a model by name."""
        if model_name not in self.configs:
            logger.error(f"Model {model_name} not found in configurations")
            return False
            
        if model_name in self.models:
            logger.info(f"Model {model_name} already loaded")
            return True
            
        try:
            config = self.configs[model_name]
            logger.info(f"Loading model {model_name} ({config.model_id})")
            
            # Load tokenizer
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
                config.model_id,
                trust_remote_code=True,
                revision=config.revision
            )
            
            # Load model with specific device map or device
            model_kwargs = config.to_model_kwargs()
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance
                **model_kwargs
            )
            
            # Move model to appropriate device(s)
            device_map = self._get_device_map(config)
            if isinstance(device_map, dict):
                model = model.cuda()  # This will respect the device_map
            else:
                model = model.to(f'cuda:{device_map}')
            
            # Apply accelerator optimizations
            model = self.accelerator.prepare(model)
            add_hook_to_module(model, AlignDevicesHook())
            
            # Apply torch compile if enabled
            if config.use_compile and hasattr(torch, 'compile'):
                logger.info(f"Applying torch.compile to {model_name}")
                model = torch.compile(
                    model,
                    backend="inductor",
                    mode="max-autotune",
                    fullgraph=True
                )
            
            self.models[model_name] = model
            logger.info(f"Successfully loaded {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}", exc_info=True)
            return False
    
    def get_model(self, model_name: str) -> Optional[AutoModelForCausalLM]:
        """Get a loaded model by name."""
        return self.models.get(model_name)
    
    def get_tokenizer(self, model_name: str) -> Optional[AutoTokenizer]:
        """Get a model's tokenizer by name."""
        return self.tokenizers.get(model_name)
    
    def get_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get a model's configuration by name."""
        return self.configs.get(model_name)
    
    def list_available_models(self) -> List[Dict]:
        """List all available model configurations."""
        return [
            {
                "name": name,
                "description": config.description,
                "context_length": config.context_length,
                "device_map": self._get_device_map(config)
            }
            for name, config in self.configs.items()
        ]
    
    def list_loaded_models(self) -> List[str]:
        """List currently loaded models."""
        return list(self.models.keys())
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model and free its resources."""
        if model_name not in self.models:
            return False
            
        try:
            # Delete model and tokenizer
            del self.models[model_name]
            del self.tokenizers[model_name]
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return True
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {str(e)}")
            return False
    
    def generate_response(
        self,
        model_name: str,
        messages: List[Dict],
        **kwargs
    ) -> Optional[str]:
        """Generate a response using the specified model."""
        model = self.get_model(model_name)
        tokenizer = self.get_tokenizer(model_name)
        config = self.get_config(model_name)
        
        if not all([model, tokenizer, config]):
            logger.error(f"Model {model_name} not fully loaded")
            return None
        
        try:
            # Format prompt
            prompt = self._format_chat_prompt(messages)
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Update generation config with any overrides
            generation_config = config.to_generation_config()
            generation_config.update(kwargs)
            
            # Generate response
            with torch.inference_mode(), torch.cuda.amp.autocast():
                outputs = model.generate(
                    **inputs,
                    **generation_config
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Clean up response
            for stop_token in config.stop_tokens:
                if stop_token in response:
                    response = response[:response.index(stop_token)]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response with {model_name}: {str(e)}")
            return None
    
    def _format_chat_prompt(self, messages: List[Dict]) -> str:
        """Format chat messages into a prompt."""
        prompt = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"{content}\n"
            elif role == "user":
                prompt += f"<|user|>{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>{content}"
        
        prompt += "<|assistant|>"
        return prompt 