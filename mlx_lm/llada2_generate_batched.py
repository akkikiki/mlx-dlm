"""
Batched generation for LLaDA2 - optimized for MLX on Apple Silicon.

This module provides batched inference for LLaDA2, processing multiple
sequences in parallel for significant speedup (~3-5x).
"""

import mlx.core as mx
from .llada2_generate import (
    get_num_transfer_tokens,
    create_block_diagonal_mask,
    sample_tokens,
)
from .models.cache import BatchKVCache


def generate_batched(
    model,
    input_ids_list: list,
    max_new_tokens: int = 256,
    block_length: int = 32,
    steps: int = 32,
    temperature: float = 0.0,
    top_p: float = None,
    top_k: int = None,
    top_nsigma: float = None,
    mask_id: int = 156895,
    eos_id: int = 156892,
    eos_early_stop: bool = True,
    use_cache: bool = False,
    pad_id: int = 0,
) -> list:
    """
    Generate tokens for multiple sequences in parallel (batched).

    Args:
        model: LLaDA2 MoE model instance
        input_ids_list: List of input token ID arrays, each of shape (seq_len,)
        max_new_tokens: Maximum number of new tokens to generate
        block_length: Size of each generation block
        steps: Number of refinement steps per block
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        top_nsigma: Top-nsigma sampling threshold
        mask_id: Token ID used for masked positions
        eos_id: End-of-sequence token ID
        eos_early_stop: Whether to stop early on EOS token
        use_cache: Whether to use KV-cache for faster generation
        pad_id: Padding token ID for variable-length sequences

    Returns:
        List of generated token ID arrays (including prompts)
    """
    # Note: KV-cache batching is now supported!

    batch_size = len(input_ids_list)
    if batch_size == 0:
        return []

    # Find max prompt length
    prompt_lengths = [len(ids) for ids in input_ids_list]
    max_prompt_length = max(prompt_lengths)

    # Calculate total length for all sequences
    total_gen_length = max_prompt_length + max_new_tokens
    num_blocks = (total_gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # Initialize batch: pad_id for left padding area, mask_id for generation area
    x = mx.full((batch_size, total_length), mask_id, dtype=mx.int32)

    # LEFT PADDING: Use pad_id (not mask_id!) for left padding to distinguish from generation masks
    # This ensures all sequences start generation from the same block boundary
    for i, input_ids in enumerate(input_ids_list):
        prompt_len = len(input_ids)
        # Calculate left padding needed
        left_pad_len = max_prompt_length - prompt_len

        # Create sequence: [PAD_ID (if needed), PROMPT, MASK_ID for generation...]
        prompt_array = mx.array(input_ids)

        # Position where prompt starts (after left padding)
        start_pos = left_pad_len
        end_pos = start_pos + prompt_len

        # Update row with PROPER padding using pad_id
        x_row = x[i]
        # LEFT: pad_id (distinct from mask_id - won't be denoised!)
        # MIDDLE: actual prompt tokens
        # RIGHT: mask_id tokens (for generation - will be denoised)
        x_row_updated = mx.concatenate([
            mx.full((start_pos,), pad_id, dtype=mx.int32),  # Use pad_id for left padding!
            prompt_array,                                     # Actual prompt
            x_row[end_pos:]                                  # Rest (mask_id for generation)
        ])

        # Reconstruct x with updated row
        if i == 0:
            x = mx.concatenate([x_row_updated[None, :], x[1:]], axis=0)
        elif i == batch_size - 1:
            x = mx.concatenate([x[:i], x_row_updated[None, :]], axis=0)
        else:
            x = mx.concatenate([x[:i], x_row_updated[None, :], x[i+1:]], axis=0)

    # Track actual prompt start positions for later extraction
    prompt_start_positions = [max_prompt_length - len(ids) for ids in input_ids_list]

    # Create KV-cache if requested (with left padding info)
    cache = None
    if use_cache:
        # Create a BatchKVCache for each model layer
        num_layers = len(model.layers)
        cache = [BatchKVCache(left_padding=prompt_start_positions) for _ in range(num_layers)]

    # Create block-diagonal attention mask (same for all in batch)
    full_mask = create_block_diagonal_mask(num_blocks, block_length)

    # Token transfer schedule
    denoising_steps = min(steps, block_length)
    num_transfer_schedule = get_num_transfer_tokens(block_length, denoising_steps)

    # With left padding, all prompts end at max_prompt_length
    # So we can skip blocks that are fully within the prompt area
    prefill_blocks = max_prompt_length // block_length

    # Prefill prompt blocks to populate cache (if using cache)
    if use_cache and prefill_blocks > 0:
        prefill_length = prefill_blocks * block_length
        prefill_tokens = x[:, :prefill_length]

        # Use block-diagonal mask for prefill (same as non-cached)
        prefill_mask = mx.concatenate(
            [full_mask[:, :, :prefill_length, :prefill_length]] * batch_size,
            axis=0
        )

        # Account for left padding by setting padding columns to -inf
        for b in range(batch_size):
            pad_start = prompt_start_positions[b]
            if pad_start > 0:
                col_indices = mx.arange(prefill_length)
                is_padding_col = col_indices < pad_start
                is_padding_col = is_padding_col[None, None, None, :]

                # Update the specific batch element
                seq_prefill_mask = prefill_mask[b:b+1]
                seq_prefill_mask = mx.where(
                    is_padding_col,
                    mx.array(float("-inf"), dtype=mx.bfloat16),
                    seq_prefill_mask
                )
                prefill_mask = mx.concatenate(
                    [prefill_mask[:b], seq_prefill_mask, prefill_mask[b+1:]],
                    axis=0
                )

        logits = model(prefill_tokens, cache=cache, mask=prefill_mask)
        mx.eval(logits)

    for num_block in range(prefill_blocks, num_blocks):
        block_start = num_block * block_length
        block_end = (num_block + 1) * block_length

        if use_cache:
            # With cache: forward only the current block
            # Record prefix offset for trimming
            prefix_offset = cache[0]._idx

            # Full-attention mask: block tokens attend to all cached + block tokens
            # Shape: (batch_size, 1, block_length, prefix_offset + block_length)
            batch_mask = mx.zeros(
                (batch_size, 1, block_length, prefix_offset + block_length),
                dtype=mx.bfloat16,
            )
        else:
            # Without cache: forward full sequence up to current block
            current_window_end = block_end
            cur_x = x[:, :current_window_end]
            cur_mask = full_mask[:, :, :current_window_end, :current_window_end]

            # Build per-sequence masks accounting for left padding
            batch_masks = []
            for b in range(batch_size):
                pad_start = prompt_start_positions[b]
                if pad_start > 0:
                    seq_mask = mx.array(cur_mask)
                    col_indices = mx.arange(current_window_end)
                    is_padding_col = col_indices < pad_start
                    is_padding_col = is_padding_col[None, None, None, :]
                    seq_mask = mx.where(
                        is_padding_col,
                        mx.array(float("-inf"), dtype=mx.bfloat16),
                        seq_mask
                    )
                else:
                    seq_mask = cur_mask
                batch_masks.append(seq_mask)
            batch_mask = mx.concatenate(batch_masks, axis=0)

        block_tokens = x[:, block_start:block_end]

        for step in range(denoising_steps):
            active_mask = block_tokens == mask_id
            num_masks = mx.sum(active_mask).item()
            if num_masks == 0:
                break

            # Trim cache back to prefix (remove stale block K/V from previous step)
            if use_cache:
                trim_amount = cache[0]._idx - prefix_offset
                if trim_amount > 0:
                    for c in cache:
                        c.trim(trim_amount)

            # Forward pass
            if use_cache:
                logits = model(block_tokens, cache=cache, mask=batch_mask)
                active_logits = logits  # Already block-sized
            else:
                logits = model(cur_x, mask=batch_mask)
                active_logits = logits[:, -block_length:, :]

            # Get logits for current block only
            active_logits = logits[:, -block_length:, :]  # (batch_size, block_length, vocab_size)

            # Sample tokens
            sampled_tokens, token_probs = sample_tokens(
                active_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                top_nsigma=top_nsigma,
            )

            # Determine which tokens to transfer
            num_to_transfer = int(num_transfer_schedule[step].item())

            # Mask out non-mask positions for confidence calculation
            confidence = mx.where(active_mask, token_probs, mx.array(-float("inf")))

            # Vectorized token selection (always use top-k, no threshold check)
            # Count active masks per sequence
            num_masks_per_seq = mx.sum(active_mask, axis=1)  # (batch_size,)

            # Determine k for each sequence: min(num_to_transfer, available_masks)
            k_per_seq = mx.minimum(num_to_transfer, num_masks_per_seq.astype(mx.int32))
            k_max = int(mx.max(k_per_seq).item())

            if k_max > 0:
                # Get top-k_max indices for all sequences (vectorized)
                # We use k_max and will mask out extras for sequences that need fewer
                top_k_indices = mx.argpartition(-confidence, kth=k_max-1, axis=1)[:, :k_max]  # (batch_size, k_max)

                # Create mask: which of the k_max indices should actually be used per sequence
                valid_k_mask = mx.arange(k_max)[None, :] < k_per_seq[:, None]  # (batch_size, k_max)

                # Vectorized boolean mask creation using broadcasting
                # Check if each position matches any valid top-k index
                indices_expanded = top_k_indices[:, :, None]  # (batch_size, k_max, 1)
                positions = mx.arange(block_length)[None, None, :]  # (1, 1, block_length)
                matches = indices_expanded == positions  # (batch_size, k_max, block_length)

                # Mask out indices beyond k for each sequence
                matches = matches & valid_k_mask[:, :, None]

                # Reduce: True if position matches any valid top-k index
                transfer_mask = mx.any(matches, axis=1)  # (batch_size, block_length)
            else:
                # All sequences have k=0 (no masks to transfer)
                transfer_mask = mx.zeros((batch_size, block_length), dtype=mx.bool_)

            # Update block tokens
            block_tokens = mx.where(transfer_mask, sampled_tokens, block_tokens)

            # Update full sequence
            if not use_cache:
                cur_x = mx.concatenate([cur_x[:, :-block_length], block_tokens], axis=1)

            # Check for EOS (early stopping per sequence)
            if eos_early_stop:
                for b in range(batch_size):
                    if mx.any(block_tokens[b] == eos_id).item():
                        # Build current sequence for EOS check
                        if use_cache:
                            x_so_far = mx.concatenate([x[:, :block_start], block_tokens], axis=1)
                        else:
                            x_so_far = cur_x

                        eos_mask = x_so_far[b] == eos_id
                        indices = mx.arange(x_so_far.shape[1])
                        eos_indices = mx.where(eos_mask, indices, mx.array(x_so_far.shape[1]))
                        eos_pos = int(mx.min(eos_indices).item())
                        if eos_pos < x_so_far.shape[1]:
                            prompt_len = prompt_lengths[b]
                            prefix = x_so_far[b, prompt_len:eos_pos]
                            if not mx.any(prefix == mask_id).item():
                                # Mark as done (we'll extract later)
                                pass

            mx.eval(block_tokens)

        # Commit finalized block to cache (if using cache)
        if use_cache:
            # Trim cache and commit final block
            trim_amount = cache[0]._idx - prefix_offset
            if trim_amount > 0:
                for c in cache:
                    c.trim(trim_amount)
            commit_logits = model(block_tokens, cache=cache, mask=batch_mask)
            mx.eval(commit_logits)

        # Update full sequence with finalized block
        x = mx.concatenate([x[:, :block_start], block_tokens, x[:, block_end:]], axis=1)

    # Extract individual sequences, removing left padding and trailing masks
    results = []
    for b in range(batch_size):
        seq = x[b]

        # Remove left padding (mask tokens at the beginning)
        # Start from where the actual prompt begins
        start_idx = prompt_start_positions[b]

        # Remove trailing mask tokens from the end
        seq_from_prompt = seq[start_idx:]
        non_mask = seq_from_prompt != mask_id
        if mx.any(non_mask).item():
            indices = mx.arange(seq_from_prompt.shape[0])
            non_mask_indices = mx.where(non_mask, indices, mx.array(-1))
            last_non_mask = int(mx.max(non_mask_indices).item())
            seq_final = seq_from_prompt[:last_non_mask + 1]
        else:
            seq_final = seq_from_prompt

        results.append(seq_final)

    return results
