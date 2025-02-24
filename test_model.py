import torch
import psutil
from lib.models import ModelManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("Testing DeepSeek R1 7B model...")
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.1f}GB")
    
    # Set default tensor type to float32 for CPU
    torch.set_default_tensor_type(torch.FloatTensor)
    
    # Disable CUDA device to ensure CPU usage
    if torch.cuda.is_available():
        torch.cuda.is_available = lambda: False
    
    # Initialize model manager with specific device
    model_manager = ModelManager()
    
    # Load the 7B model
    print("\nLoading model...")
    try:
        success = model_manager.load_model("deepseek-r1-7b")
        if not success:
            print("Failed to load model")
            return
            
        # Test generation
        messages = [
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        print("\nGenerating response...")
        response = model_manager.generate_response(
            "deepseek-r1-7b",
            messages,
            temperature=0.7,
            max_new_tokens=100
        )
        
        print("\nResponse:")
        print(response)
        
    except Exception as e:
        logger.error(f"Error during model testing: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
    finally:
        # Clean up
        if hasattr(model_manager, 'models') and "deepseek-r1-7b" in model_manager.models:
            model_manager.unload_model("deepseek-r1-7b")

if __name__ == "__main__":
    main() 