"""Quick test script to check if large DeBERTa model can load."""

import torch
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def show_gpu_memory():
    """Display current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Total: {total:.2f} GB")
    else:
        print("No GPU available")

print("Testing DeBERTa-Large model loading...")
print("\nInitial GPU state:")
show_gpu_memory()

# Clear any existing GPU memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

print("\nAfter cleanup:")
show_gpu_memory()

# Try loading the model
model_name = "microsoft/deberta-v3-large"
print(f"\nLoading model: {model_name}")

try:
    # Load tokenizer first (lightweight)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    print("Tokenizer loaded successfully")
    
    # Load model
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    print("Model loaded successfully!")
    
    # Count parameters
    params_M = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params_M:.1f}M")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        print("\nMoving model to GPU...")
        model = model.cuda()
        print("Model on GPU")
        
        print("\nAfter loading to GPU:")
        show_gpu_memory()
        
        # Enable gradient checkpointing
        print("\nEnabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    print("\n✓ SUCCESS: Large model loaded and ready!")
    
    # Cleanup
    print("\nCleaning up...")
    if torch.cuda.is_available():
        model.cpu()
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("\nAfter cleanup:")
    show_gpu_memory()
    print("\n✓ All tests passed!")
    
except RuntimeError as e:
    print(f"\n✗ ERROR: {e}")
    if "out of memory" in str(e).lower():
        print("\n*** GPU Out of Memory ***")
        print("Suggestions:")
        print("1. Close any other programs using GPU")
        print("2. Use smaller batch sizes (try batch_size=2)")
        print("3. Use gradient accumulation (grad_accum=8 or more)")
        print("4. Consider using CPU for this model")
    show_gpu_memory()
except Exception as e:
    print(f"\n✗ UNEXPECTED ERROR: {e}")
    import traceback
    traceback.print_exc()
