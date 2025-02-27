import torch
import logging
from typing import Dict, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
        
        # Print diagnostic information
        logger.info("CUDA Environment:")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")
        logger.info(f"PyTorch version: {torch.__version__}")
        
        try:
            import subprocess
            nvidia_smi = subprocess.check_output(['nvidia-smi']).decode()
            logger.info(f"nvidia-smi output:\n{nvidia_smi}")
        except Exception as e:
            logger.warning(f"Could not run nvidia-smi: {e}")
        
        # Initialize accelerator with mixed precision
        self.accelerator = Accelerator(
            mixed_precision="bf16",  # Use bfloat16 mixed precision
            device_placement=True
        )
        
        # Verify CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This implementation requires GPU.")
            
        logger.info(f"Using device: {self.accelerator.device}")
        logger.info(f"Mixed precision: {self.accelerator.mixed_precision}")
    
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
            
            # Clear CUDA cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Load tokenizer
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
                config.model_id,
                trust_remote_code=True,
                revision=config.revision,
                padding_side="left",
                add_bos_token=True
            )
            
            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            
            # Get model kwargs
            model_kwargs = config.to_model_kwargs()
            if 'torch_dtype' in model_kwargs:
                del model_kwargs['torch_dtype']
            if 'device_map' in model_kwargs:
                del model_kwargs['device_map']
            
            # Load model with memory optimizations
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto" if config.device == "auto" else config.device_map,
                quantization_config=quantization_config,
                **model_kwargs
            )
            
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
            # Clean up on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
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
                "device": config.device if not config.device_map else "distributed"
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
    
    def _format_chat_prompt(self, messages: List[Dict]) -> str:
        """Format chat messages into a prompt."""
        prompt = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"<|system|>{content}</|system|>\n"
            elif role == "user":
                prompt += f"<|user|>{content}</|user|>\n"
            elif role == "assistant":
                prompt += f"<|assistant|>{content}</|assistant|>\n"
        
        prompt += "<|assistant|>"
        return prompt
    
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
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.context_length - 512  # Leave room for response
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Create stopping criteria for special tokens
            stop_tokens = [
                "</|assistant|>", "<|user|>", "<|system|>",
                "<think>", "</think>",
                *config.stop_tokens  # Include model's configured stop tokens
            ]
            stop_token_ids = []
            for stop_token in stop_tokens:
                stop_ids = tokenizer.encode(stop_token, add_special_tokens=False)
                stop_token_ids.extend(stop_ids)
            
            # Get base generation config from model config
            generation_config = config.to_generation_config()
            
            # Add tokenizer-specific settings
            generation_config.update({
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": stop_token_ids,  # Use stop tokens as EOS tokens
                "bos_token_id": tokenizer.bos_token_id
            })
            
            # Update with any overrides from kwargs
            generation_config.update(kwargs)
            
            # Generate response using the new autocast API
            with torch.inference_mode():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model.generate(
                        **inputs,
                        **generation_config
                    )
            
            # Decode response
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],  # Only decode new tokens
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Clean up response
            response = response.strip()
            
            # Handle any closing tags
            if '</' in response:
                response = response[:response.index('</')]
            
            # Handle think tags and other special tokens
            special_tokens = [
                "<think>", "</think>",
                "<|assistant|>", "</|assistant|>",
                "<|user|>", "</|user|>",
                "<|system|>", "</|system|>"
            ]
            
            # Remove any special tokens from the response
            for token in special_tokens:
                response = response.replace(token, "")
            
            # Final cleanup
            response = response.strip()
            
            # Extract content between think tags if present
            if "<think>" in response and "</think>" in response:
                start = response.find("<think>") + len("<think>")
                end = response.find("</think>")
                if start < end:
                    response = response[start:end].strip()
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response with {model_name}: {str(e)}")
            return None 