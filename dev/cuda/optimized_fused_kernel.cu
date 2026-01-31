/**
 * OPTIMIZED FUSED KERNEL - Reduced Memory Operations
 *
 * Strategy: Fuse all operations to minimize memory traffic
 * - Pass 1: Compute max, sum, and sum_of_squares (no intermediate write!)
 * - Pass 2: Apply threshold filter (final output)
 *
 * Memory operations: 2 reads + 1 write = 3 operations (vs original 7)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>

// ============================================================================
// OPTIMIZED FUSED KERNEL
// ============================================================================

template<int BLOCK_SIZE>
__global__ void temperature_topnsigma_fused(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int vocab_size,
    const float temperature,
    const float top_nsigma
) {
    __shared__ float shared_max[BLOCK_SIZE];
    __shared__ float shared_sum[BLOCK_SIZE];
    __shared__ float shared_sum_sq[BLOCK_SIZE];

    const int pos_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int offset = pos_idx * vocab_size;

    // ========================================================================
    // PASS 1: Compute max, sum, and sum_of_squares in ONE PASS
    // ========================================================================
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Grid-stride loop: scale and accumulate statistics
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float val = input[offset + i] / temperature;  // Scale on-the-fly
        local_max = fmaxf(local_max, val);
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Store to shared memory
    shared_max[tid] = local_max;
    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    // Parallel reduction for max
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    // Parallel reduction for sum
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    // Parallel reduction for sum of squares
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        __syncthreads();
    }

    // Compute statistics
    float maximum = shared_max[0];
    float sum = shared_sum[0];
    float sum_sq = shared_sum_sq[0];
    __syncthreads();

    // Compute mean and std using: variance = E[X²] - E[X]²
    float mean = sum / float(vocab_size);
    float variance = (sum_sq / float(vocab_size)) - (mean * mean);
    float std_dev = sqrtf(variance);
    float threshold = maximum - top_nsigma * std_dev;

    // ========================================================================
    // PASS 2: Apply filter (scale and filter in one pass)
    // ========================================================================
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float val = input[offset + i] / temperature;  // Re-scale (from cache)
        output[offset + i] = (val >= threshold) ? val : -INFINITY;
    }
}

// ============================================================================
// SUPER OPTIMIZED: Using Warp Primitives for Faster Reductions
// ============================================================================

template<int BLOCK_SIZE>
__global__ void temperature_topnsigma_fused_warp(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int vocab_size,
    const float temperature,
    const float top_nsigma
) {
    // Smaller shared memory (one element per warp)
    __shared__ float shared_max[BLOCK_SIZE / 32];
    __shared__ float shared_sum[BLOCK_SIZE / 32];
    __shared__ float shared_sum_sq[BLOCK_SIZE / 32];

    const int pos_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int offset = pos_idx * vocab_size;

    // ========================================================================
    // PASS 1: Compute statistics (max, sum, sum_sq)
    // ========================================================================
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float val = input[offset + i] / temperature;
        local_max = fmaxf(local_max, val);
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Warp-level reduction for max
    for (int delta = 16; delta > 0; delta >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, delta));
    }

    // Warp-level reduction for sum
    for (int delta = 16; delta > 0; delta >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, delta);
    }

    // Warp-level reduction for sum_sq
    for (int delta = 16; delta > 0; delta >>= 1) {
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, delta);
    }

    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        shared_max[warp_id] = local_max;
        shared_sum[warp_id] = local_sum;
        shared_sum_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0) {
        float final_max = (lane_id < BLOCK_SIZE / 32) ? shared_max[lane_id] : -FLT_MAX;
        float final_sum = (lane_id < BLOCK_SIZE / 32) ? shared_sum[lane_id] : 0.0f;
        float final_sum_sq = (lane_id < BLOCK_SIZE / 32) ? shared_sum_sq[lane_id] : 0.0f;

        for (int delta = 16; delta > 0; delta >>= 1) {
            final_max = fmaxf(final_max, __shfl_down_sync(0xffffffff, final_max, delta));
            final_sum += __shfl_down_sync(0xffffffff, final_sum, delta);
            final_sum_sq += __shfl_down_sync(0xffffffff, final_sum_sq, delta);
        }

        // Thread 0 writes results back to shared memory for all threads to read
        if (lane_id == 0) {
            shared_max[0] = final_max;
            shared_sum[0] = final_sum;
            shared_sum_sq[0] = final_sum_sq;
        }
    }
    __syncthreads();

    // All threads read the final results
    float maximum = shared_max[0];
    float sum = shared_sum[0];
    float sum_sq = shared_sum_sq[0];

    // Compute statistics
    float mean = sum / float(vocab_size);
    float variance = (sum_sq / float(vocab_size)) - (mean * mean);
    float std_dev = sqrtf(variance);
    float threshold = maximum - top_nsigma * std_dev;

    // ========================================================================
    // PASS 2: Apply filter
    // ========================================================================
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float val = input[offset + i] / temperature;
        output[offset + i] = (val >= threshold) ? val : -INFINITY;
    }
}

// ============================================================================
// C++ LAUNCHER FUNCTIONS
// ============================================================================

extern "C" {

void launch_temperature_topnsigma_fused(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    float temperature,
    float top_nsigma,
    cudaStream_t stream
) {
    const int num_positions = batch_size * seq_len;
    const int threads_per_block = 256;

    dim3 grid(num_positions);
    dim3 block(threads_per_block);

    temperature_topnsigma_fused<256><<<grid, block, 0, stream>>>(
        input,
        output,
        vocab_size,
        temperature,
        top_nsigma
    );
}

void launch_temperature_topnsigma_fused_warp(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    float temperature,
    float top_nsigma,
    cudaStream_t stream
) {
    const int num_positions = batch_size * seq_len;
    const int threads_per_block = 256;

    dim3 grid(num_positions);
    dim3 block(threads_per_block);

    temperature_topnsigma_fused_warp<256><<<grid, block, 0, stream>>>(
        input,
        output,
        vocab_size,
        temperature,
        top_nsigma
    );
}

} // extern "C"
