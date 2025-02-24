import torch
import psutil
from lib.models import ModelManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_cuda():
    """Setup CUDA environment and print debug info."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    # Print CUDA information
    print("\nCUDA Setup:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print("\nGPU Information:")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB")
    print(f"Current device: {torch.cuda.current_device()}")
    
    # Set default device and dtype
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float32)
    
    # Empty cache
    torch.cuda.empty_cache()

def main():
    print("Testing DeepSeek R1 Distill Llama 8B model...")
    
    try:
        # Setup CUDA
        setup_cuda()
        
        print(f"\nSystem Memory: {psutil.virtual_memory().available / (1024**3):.1f}GB available")
        
        # Initialize model manager
        model_manager = ModelManager()
        
        # Load the 8B model
        print("\nLoading model...")
        success = model_manager.load_model("deepseek-r1-distil-8b")
        if not success:
            print("Failed to load model")
            return
            
        # Test generation
        messages = [
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        print("\nGenerating response...")
        with torch.inference_mode():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                response = model_manager.generate_response(
                    "deepseek-r1-distil-8b",
                    messages,
                    temperature=0.7,
                    max_new_tokens=100
                )
        
        print("\nResponse:")
        print(response)
        
        # Print GPU memory usage
        print("\nGPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}:")
            print(f"  Allocated: {allocated:.2f}GB")
            print(f"  Reserved:  {reserved:.2f}GB")
        
    except Exception as e:
        logger.error(f"Error during model testing: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
    finally:
        # Clean up
        if hasattr(model_manager, 'models') and "deepseek-r1-distil-8b" in model_manager.models:
            model_manager.unload_model("deepseek-r1-distil-8b")
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 