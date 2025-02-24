import torch
import psutil
from lib.models import ModelManager

def main():
    print("Testing DeepSeek R1 7B model...")
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.1f}GB")
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Load the 7B model
    print("\nLoading model...")
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

if __name__ == "__main__":
    main() 