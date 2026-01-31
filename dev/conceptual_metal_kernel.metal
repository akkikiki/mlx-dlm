/**
 * Conceptual Metal kernel for temperature_topnsigma_sample
 *
 * This is what MLX CONCEPTUALLY generates when you use @mx.compile.
 * The actual generated code is not exposed by MLX, but this shows
 * the structure and operations that would be in the fused kernel.
 *
 * MLX generates this from your Python code:
 *   logits = logits / temperature
 *   maximum = mx.max(logits, axis=-1)
 *   std = mx.std(logits, axis=-1)
 *   threshold = maximum - top_nsigma * std
 *   logits = mx.where(logits >= threshold, logits, -inf)
 *   probs = mx.softmax(logits, axis=-1)
 *   tokens = mx.random.categorical(log(probs))
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// KERNEL 1: Fused Temperature + Top-NSigma Filtering
// ============================================================================

kernel void fused_temperature_topnsigma_filter(
    // Input/Output buffers
    device const float* logits_in       [[buffer(0)]],   // Input logits
    device float* logits_out            [[buffer(1)]],   // Filtered logits

    // Parameters (passed as data, not code paths!)
    constant float& temperature         [[buffer(2)]],   // Temperature value
    constant float& top_nsigma          [[buffer(3)]],   // Top-nsigma value

    // Shape info
    constant uint& batch_size           [[buffer(4)]],
    constant uint& seq_len              [[buffer(5)]],
    constant uint& vocab_size           [[buffer(6)]],

    // Thread info
    uint3 gid                           [[thread_position_in_grid]],
    uint3 tid                           [[thread_position_in_threadgroup]],
    uint3 tpg                           [[threads_per_threadgroup]],

    // Shared memory for reductions
    threadgroup float* shared_mem       [[threadgroup(0)]]
) {
    // Each threadgroup handles one position (batch_size * seq_len)
    uint position = gid.x;  // Which position (0 to batch_size * seq_len - 1)
    uint tid_x = tid.x;     // Thread ID within threadgroup

    if (position >= batch_size * seq_len) return;

    // Base offset for this position's logits
    uint logits_offset = position * vocab_size;

    // ========================================================================
    // STEP 1: Temperature Scaling (element-wise, no reduction)
    // ========================================================================
    // Each thread handles multiple vocabulary items (strided access)
    for (uint v = tid_x; v < vocab_size; v += tpg.x) {
        uint idx = logits_offset + v;
        float logit = logits_in[idx];

        // Temperature scaling (ALWAYS applied - no if statement!)
        logit = logit / temperature;

        // Store in shared memory for reductions
        shared_mem[v] = logit;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========================================================================
    // STEP 2: Compute Maximum (parallel reduction)
    // ========================================================================
    float local_max = -INFINITY;
    for (uint v = tid_x; v < vocab_size; v += tpg.x) {
        local_max = max(local_max, shared_mem[v]);
    }

    // Parallel reduction to find global max
    threadgroup float* max_shared = shared_mem + vocab_size;
    max_shared[tid_x] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = tpg.x / 2; stride > 0; stride /= 2) {
        if (tid_x < stride) {
            max_shared[tid_x] = max(max_shared[tid_x], max_shared[tid_x + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float maximum = max_shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========================================================================
    // STEP 3: Compute Standard Deviation (parallel reduction)
    // ========================================================================
    // First compute mean
    float local_sum = 0.0f;
    for (uint v = tid_x; v < vocab_size; v += tpg.x) {
        local_sum += shared_mem[v];
    }

    threadgroup float* sum_shared = max_shared;
    sum_shared[tid_x] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpg.x / 2; stride > 0; stride /= 2) {
        if (tid_x < stride) {
            sum_shared[tid_x] += sum_shared[tid_x + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = sum_shared[0] / vocab_size;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Now compute variance
    float local_var_sum = 0.0f;
    for (uint v = tid_x; v < vocab_size; v += tpg.x) {
        float diff = shared_mem[v] - mean;
        local_var_sum += diff * diff;
    }

    sum_shared[tid_x] = local_var_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpg.x / 2; stride > 0; stride /= 2) {
        if (tid_x < stride) {
            sum_shared[tid_x] += sum_shared[tid_x + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float variance = sum_shared[0] / vocab_size;
    float std_dev = sqrt(variance);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========================================================================
    // STEP 4: Compute Threshold and Filter (ALWAYS applied!)
    // ========================================================================
    float threshold = maximum - top_nsigma * std_dev;

    // Apply filtering
    for (uint v = tid_x; v < vocab_size; v += tpg.x) {
        uint idx = logits_offset + v;
        float logit = shared_mem[v];

        // Top-nsigma filtering (data-dependent, but always executed!)
        if (logit < threshold) {
            logit = -INFINITY;
        }

        // Write filtered logits
        logits_out[idx] = logit;
    }
}


// ============================================================================
// KERNEL 2: Softmax
// ============================================================================

kernel void softmax_kernel(
    device const float* logits          [[buffer(0)]],
    device float* probs                 [[buffer(1)]],
    constant uint& batch_size           [[buffer(2)]],
    constant uint& seq_len              [[buffer(3)]],
    constant uint& vocab_size           [[buffer(4)]],
    uint3 gid                           [[thread_position_in_grid]],
    uint3 tid                           [[thread_position_in_threadgroup]],
    uint3 tpg                           [[threads_per_threadgroup]],
    threadgroup float* shared_mem       [[threadgroup(0)]]
) {
    uint position = gid.x;
    if (position >= batch_size * seq_len) return;

    uint logits_offset = position * vocab_size;
    uint tid_x = tid.x;

    // Find max for numerical stability
    float local_max = -INFINITY;
    for (uint v = tid_x; v < vocab_size; v += tpg.x) {
        local_max = max(local_max, logits[logits_offset + v]);
    }

    shared_mem[tid_x] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpg.x / 2; stride > 0; stride /= 2) {
        if (tid_x < stride) {
            shared_mem[tid_x] = max(shared_mem[tid_x], shared_mem[tid_x + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float max_val = shared_mem[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute exp and sum
    float local_sum = 0.0f;
    for (uint v = tid_x; v < vocab_size; v += tpg.x) {
        float exp_val = exp(logits[logits_offset + v] - max_val);
        shared_mem[vocab_size + v] = exp_val;  // Store exp values
        local_sum += exp_val;
    }

    shared_mem[tid_x] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpg.x / 2; stride > 0; stride /= 2) {
        if (tid_x < stride) {
            shared_mem[tid_x] += shared_mem[tid_x + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sum_exp = shared_mem[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize to get probabilities
    for (uint v = tid_x; v < vocab_size; v += tpg.x) {
        probs[logits_offset + v] = shared_mem[vocab_size + v] / sum_exp;
    }
}


// ============================================================================
// KERNEL 3: Categorical Sampling
// ============================================================================

kernel void categorical_sample_kernel(
    device const float* probs           [[buffer(0)]],
    device int* tokens                  [[buffer(1)]],
    device float* token_probs           [[buffer(2)]],
    device uint* random_state           [[buffer(3)]],  // Random number generator state
    constant uint& batch_size           [[buffer(4)]],
    constant uint& seq_len              [[buffer(5)]],
    constant uint& vocab_size           [[buffer(6)]],
    uint gid                            [[thread_position_in_grid]]
) {
    if (gid >= batch_size * seq_len) return;

    uint probs_offset = gid * vocab_size;

    // Generate random number (0, 1)
    // Using a simple LCG for demonstration (MLX uses better RNG)
    uint seed = random_state[gid];
    seed = 1664525u * seed + 1013904223u;
    random_state[gid] = seed;
    float rand_val = float(seed) / float(0xFFFFFFFFu);

    // Categorical sampling: find first index where cumsum > rand_val
    float cumsum = 0.0f;
    int sampled_token = 0;

    for (uint v = 0; v < vocab_size; v++) {
        cumsum += probs[probs_offset + v];
        if (cumsum > rand_val) {
            sampled_token = v;
            break;
        }
    }

    // Store results
    tokens[gid] = sampled_token;
    token_probs[gid] = probs[probs_offset + sampled_token];
}


// ============================================================================
// Notes on MLX's Actual Implementation
// ============================================================================

/*
 * MLX's actual generated code would be MORE optimized:
 *
 * 1. Better memory coalescing patterns
 * 2. Templated for different data types (float16, float32, bfloat16)
 * 3. Auto-tuned thread block sizes
 * 4. Vectorized loads/stores (float4, etc.)
 * 5. Bank conflict avoidance in shared memory
 * 6. More sophisticated random number generation
 * 7. Possibly fused into even fewer kernels
 *
 * But the CONCEPTUAL structure is the same:
 *   - Fixed operations (no if statements for parameter values)
 *   - Parallel reductions for max/std
 *   - Shared memory for fast communication
 *   - Parameters as data (temperature, top_nsigma)
 */
