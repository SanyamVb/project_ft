"""
Fix for completion_mask dimension mismatch in Unsloth's compiled GRPOTrainer.

Unsloth's dynamic compiler (unsloth_zoo/compiler.py) rewrites TRL's GRPOTrainer
source code, which can break variable scoping in the loss computation, causing
completion_mask shape (B, completion_len) to mismatch per_token_loss shape.

This fix patches _compute_loss at the instance level to ensure completion_mask
is aligned with completion_ids before the loss computation runs.
"""

import types
import torch


def _fix_mask_dimensions(inputs):
    """Ensure completion_mask matches completion_ids dimensions in the inputs dict."""
    completion_mask = inputs.get("completion_mask")
    completion_ids = inputs.get("completion_ids")

    if completion_mask is None or completion_ids is None:
        return

    mask_len = completion_mask.shape[1]
    target_len = completion_ids.shape[1]

    if mask_len == target_len:
        return  # Already aligned

    batch_size = completion_mask.shape[0]

    if mask_len > target_len:
        # Trim mask to match completion_ids (keep the rightmost tokens)
        inputs["completion_mask"] = completion_mask[:, -target_len:]
    else:
        # Pad mask to match completion_ids (pad with zeros on the left)
        padded = torch.zeros(
            batch_size, target_len,
            dtype=completion_mask.dtype,
            device=completion_mask.device,
        )
        padded[:, -mask_len:] = completion_mask
        inputs["completion_mask"] = padded


def apply_completion_mask_fix(trainer):
    """
    Patch a GRPOTrainer instance to fix completion_mask dimension mismatches.

    Must be called AFTER trainer instantiation, before trainer.train().
    Works with both vanilla TRL GRPOTrainer and Unsloth's UnslothGRPOTrainer.
    """
    trainer_cls_name = type(trainer).__name__

    # Prefer _compute_loss (TRL internal), fall back to compute_loss (HF Trainer)
    if hasattr(trainer, '_compute_loss'):
        original_fn = trainer._compute_loss

        def patched_compute_loss(self, model, inputs):
            _fix_mask_dimensions(inputs)
            return original_fn(model, inputs)

        trainer._compute_loss = types.MethodType(patched_compute_loss, trainer)
        print(f"✓ Applied completion_mask fix on {trainer_cls_name}._compute_loss")
    elif hasattr(trainer, 'compute_loss'):
        original_fn = trainer.compute_loss

        def patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            _fix_mask_dimensions(inputs)
            return original_fn(model, inputs, return_outputs, num_items_in_batch)

        trainer.compute_loss = types.MethodType(patched_compute_loss, trainer)
        print(f"✓ Applied completion_mask fix on {trainer_cls_name}.compute_loss")
    else:
        print(f"⚠ Could not find _compute_loss or compute_loss on {trainer_cls_name}")
        return False

    return True
