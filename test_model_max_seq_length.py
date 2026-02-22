"""Test script to verify model's max_seq_length attribute is set correctly."""

from qwen_pipeline.config import Config
from qwen_pipeline.inference import load_trained_model


def test_max_seq_length():
    """Load model and verify max_seq_length attribute."""
    print("=" * 60)
    print("Testing Model max_seq_length Attribute")
    print("=" * 60)
    
    # Initialize config
    config = Config()
    print(f"\n1. Config loaded successfully")
    print(f"   - Config.max_seq_length: {config.max_seq_length}")
    print(f"   - Model path: {config.sft_output_dir}")
    
    # Load the trained model
    print(f"\n2. Loading model from {config.sft_output_dir}...")
    try:
        model, tokenizer = load_trained_model(config)
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return
    
    # Check if max_seq_length attribute exists
    print(f"\n3. Checking max_seq_length attribute...")
    if hasattr(model, 'max_seq_length'):
        print(f"   ✓ model.max_seq_length exists")
        print(f"   - Value: {model.max_seq_length}")
        print(f"   - Type: {type(model.max_seq_length)}")
        
        # Verify it matches config
        if model.max_seq_length == config.max_seq_length:
            print(f"   ✓ Matches config.max_seq_length ({config.max_seq_length})")
        else:
            print(f"   ✗ Mismatch! Expected {config.max_seq_length}, got {model.max_seq_length}")
    else:
        print(f"   ✗ model.max_seq_length attribute NOT FOUND")
        print(f"   Available attributes: {[attr for attr in dir(model) if 'max' in attr.lower()]}")
    
    # Additional model info
    print(f"\n4. Model information:")
    print(f"   - Model type: {type(model).__name__}")
    print(f"   - Device: {model.device if hasattr(model, 'device') else 'N/A'}")
    print(f"   - Tokenizer vocab size: {len(tokenizer)}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_max_seq_length()
