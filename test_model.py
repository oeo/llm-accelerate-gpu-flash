import torch
import psutil
from lib.models import ModelManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("Testing DeepSeek R1 Distill Llama 8B model...")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.1f}GB")
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Load the 8B model
    print("\nLoading model...")
    try:
        success = model_manager.load_model("deepseek-r1-distil-8b")
        if not success:
            print("Failed to load model")
            return
            
        # Test generation
        messages = [
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        print("\nGenerating response...")
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
            mem = torch.cuda.memory_allocated(i) / 1024**3
            print(f"GPU {i}: {mem:.2f}GB")
        
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