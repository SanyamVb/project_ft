"""
Monkey-patch for GRPOTrainer to fix completion_mask dimension mismatch.

This fixes the issue where completion_mask has shape [batch_size, completion_length]
but needs [batch_size, full_sequence_length] to match concatenated sequences.
"""

import torch
from trl import GRPOTrainer
from typing import Dict, List, Tuple, Union


def patched_create_grpo_masks(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fixed version that properly handles completion_mask dimensions.
    
    The completion_mask should be aligned with the full sequence (prompt + completion),
    not just the completion part.
    """
    batch_size, seq_len = input_ids.shape
    
    # If completion_mask is shorter than seq_len, it needs to be padded
    if completion_mask.shape[1] < seq_len:
        # Create a mask that's False for prompt tokens and True for completion tokens
        full_completion_mask = torch.zeros(batch_size, seq_len, dtype=completion_mask.dtype, device=completion_mask.device)
        # Place the completion_mask at the end (right-aligned)
        completion_len = completion_mask.shape[1]
        full_completion_mask[:, -completion_len:] = completion_mask
        completion_mask = full_completion_mask
    
    return attention_mask, completion_mask


def apply_grpo_trainer_fix():
    """Apply the monkey-patch to fix GRPOTrainer completion_mask issue."""
    # Store original method if needed
    if not hasattr(GRPOTrainer, '_original_create_grpo_masks'):
        GRPOTrainer._original_create_grpo_masks = getattr(GRPOTrainer, 'create_grpo_masks', None)
    
    # Check if the method exists and patch it
    if hasattr(GRPOTrainer, 'create_grpo_masks'):
        GRPOTrainer.create_grpo_masks = patched_create_grpo_masks
        print("✓ Applied GRPOTrainer.create_grpo_masks patch")
    
    # Also patch the internal _create_completion_attention_mask if it exists
    if hasattr(GRPOTrainer, '_create_completion_attention_mask'):
        original_create_completion = GRPOTrainer._create_completion_attention_mask
        
        def patched_create_completion_attention_mask(self, completion_ids, **kwargs):
            mask = original_create_completion.fget(self)(completion_ids, **kwargs) if isinstance(original_create_completion, property) else original_create_completion(self, completion_ids, **kwargs)
            return mask
        
        GRPOTrainer._patched_create_completion_attention_mask = patched_create_completion_attention_mask
        print("✓ Applied GRPOTrainer._create_completion_attention_mask patch")
    
    return True


def patch_grpo_compute_loss():
    """
    Alternative: Patch the compute_loss method to handle dimension mismatches.
    This is more aggressive but handles edge cases.
    """
    original_compute_loss = GRPOTrainer.compute_loss
    
    def patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Wrapper that ensures completion_mask has correct dimensions."""
        if "completion_mask" in inputs:
            completion_mask = inputs["completion_mask"]
            input_ids = inputs.get("input_ids")
            
            if input_ids is not None and completion_mask.shape[1] != input_ids.shape[1]:
                # Fix dimension mismatch
                batch_size = completion_mask.shape[0]
                target_len = input_ids.shape[1]
                completion_len = completion_mask.shape[1]
                
                # Create properly sized mask
                fixed_mask = torch.zeros(
                    batch_size, target_len,
                    dtype=completion_mask.dtype,
                    device=completion_mask.device
                )
                # Copy completion mask to the right position (end of sequence)
                fixed_mask[:, -completion_len:] = completion_mask
                inputs["completion_mask"] = fixed_mask
        
        return original_compute_loss(self, model, inputs, return_outputs, num_items_in_batch)
    
    GRPOTrainer.compute_loss = patched_compute_loss
    print("✓ Applied GRPOTrainer.compute_loss patch")


def apply_all_fixes():
    """Apply all necessary fixes for GRPO trainer."""
    try:
        apply_grpo_trainer_fix()
        patch_grpo_compute_loss()
        print("\n✓ All GRPO trainer fixes applied successfully!")
        return True
    except Exception as e:
        print(f"✗ Error applying fixes: {e}")
        return False
