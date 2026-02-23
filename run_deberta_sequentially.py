"""Helper to show commands for running DeBERTa experiments one at a time.

Due to GPU memory accumulation issues, it's best to run each experiment
in a separate Python process with full cleanup between runs.

This script shows you the commands to run manually, one at a time.
"""

experiments = [
    # Small model
    ("small", 1),
    ("small", 2),
    ("small", 3),
    
    # Base model  
    ("base", 1),
    ("base", 2),
    ("base", 3),
    
    # Large model
    ("large", 1),
    ("large", 2),
    ("large", 3),
]

print("="*80)
print("DeBERTa Sequential Experiment Guide")
print("="*80)
print("\nTo avoid GPU memory issues, run each experiment separately:")
print("\nNOTE: The current script runs all 3 epochs for each model size.")
print("To truly run one at a time, you need to run the training separately")
print("and use --eval_only mode for subsequent epochs.\n")

print("RECOMMENDED APPROACH - Run one model size at a time:\n")

print("# Small model (all 3 epochs)")
print("python compare_deberta_variants.py --skip_base --skip_large\n")

print("# Base model (all 3 epochs)")
print("python compare_deberta_variants.py --skip_small --skip_large\n")

print("# Large model (all 3 epochs)")
print("python compare_deberta_variants.py --skip_small --skip_base\n")

print("="*80)
print("\nAfter running all, analyze results with:")
print("python analyze_deberta_comparison.py")
print("="*80)


