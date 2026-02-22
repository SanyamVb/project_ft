"""Test script to verify model's max_seq_length attribute is set correctly."""

from unsloth import FastLanguageModel
from qwen_pipeline.config import Config


def test_max_seq_length():
    """Load FastLanguageModel and verify max_seq_length attribute."""
    print("=" * 60)
    print("Testing FastLanguageModel max_seq_length Attribute")
    print("=" * 60)
    
    # Load config
    config = Config()
    print(f"\n1. Config loaded")
    print(f"   - config.max_seq_length: {config.max_seq_length}")
    print(f"   - config.llm_model: {config.llm_model}")
    
    # Load model with max_seq_length from config
    print(f"\n2. Loading model...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.llm_model,
            max_seq_length=config.max_seq_length,
            load_in_4bit=False,
            dtype=None,
        )
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check if max_seq_length attribute exists BEFORE setting it
    print(f"\n3. Checking max_seq_length BEFORE manual setting...")
    if hasattr(model, 'max_seq_length'):
        print(f"   ✓ model.max_seq_length EXISTS: {model.max_seq_length}")
    else:
        print(f"   ✗ model.max_seq_length NOT FOUND")
    
    # Manually set max_seq_length (the fix)
    print(f"\n4. Manually setting model.max_seq_length = {config.max_seq_length}")
    model.max_seq_length = config.max_seq_length
    
    # Check again AFTER setting it
    print(f"\n5. Checking max_seq_length AFTER manual setting...")
    if hasattr(model, 'max_seq_length'):
        print(f"   ✓ model.max_seq_length EXISTS: {model.max_seq_length}")
        print(f"   ✓ Matches config: {model.max_seq_length == config.max_seq_length}")
    else:
        print(f"   ✗ model.max_seq_length still NOT FOUND (unexpected!)")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_max_seq_length()
