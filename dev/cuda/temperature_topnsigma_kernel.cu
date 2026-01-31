/**
 * CUDA Kernel Implementation for Temperature + Top-NSigma Sampling
 *
 * This kernel fuses temperature scaling and top-nsigma filtering into a single
 * GPU kernel for optimal performance.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>

// ============================================================================
// CUDA KERNEL: Temperature + Top-NSigma Filter
// ============================================================================

template<int BLOCK_SIZE>
__global__ void temperature_topnsigma_kernel(
    const float* __restrict__ input,      // Input logits [num_positions, vocab_size]
    float* __restrict__ output,           // Output logits [num_positions, vocab_size]
    const int vocab_size,
    const float temperature,
    const float top_nsigma
) {
    // Shared memory for parallel reductions
    __shared__ float shared_mem[BLOCK_SIZE];

    // Each block processes one position (batch_idx, seq_idx)
    const int pos_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int offset = pos_idx * vocab_size;

    // ========================================================================
    // STEP 1: Temperature Scaling
    // ========================================================================
    // Grid-stride loop: each thread processes multiple elements
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        output[offset + i] = input[offset + i] / temperature;
    }
    __syncthreads();

    // ========================================================================
    // STEP 2: Find Maximum (Parallel Reduction)
    // ========================================================================
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, output[offset + i]);
    }
    shared_mem[tid] = local_max;
    __syncthreads();

    // Tree reduction for max
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + stride]);
        }
        __syncthreads();
    }
    float maximum = shared_mem[0];
    __syncthreads();

    // ========================================================================
    // STEP 3: Compute Mean (Parallel Reduction)
    // ========================================================================
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        local_sum += output[offset + i];
    }
    shared_mem[tid] = local_sum;
    __syncthreads();

    // Tree reduction for sum
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    float mean = shared_mem[0] / float(vocab_size);
    __syncthreads();

    // ========================================================================
    // STEP 4: Compute Variance and Standard Deviation (Parallel Reduction)
    // ========================================================================
    float local_var_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float diff = output[offset + i] - mean;
        local_var_sum += diff * diff;
    }
    shared_mem[tid] = local_var_sum;
    __syncthreads();

    // Tree reduction for variance sum
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    float variance = shared_mem[0] / float(vocab_size);
    float std_dev = sqrtf(variance);
    __syncthreads();

    // ========================================================================
    // STEP 5: Apply Top-NSigma Filter
    // ========================================================================
    float threshold = maximum - top_nsigma * std_dev;

    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float val = output[offset + i];
        output[offset + i] = (val >= threshold) ? val : -INFINITY;
    }
}

// ============================================================================
// OPTIMIZED KERNEL WITH WARP PRIMITIVES (CUDA 9+)
// ============================================================================

template<int BLOCK_SIZE>
__global__ void temperature_topnsigma_kernel_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int vocab_size,
    const float temperature,
    const float top_nsigma
) {
    // Shared memory for warp-level reductions
    __shared__ float shared_mem[BLOCK_SIZE / 32];  // One per warp

    const int pos_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int offset = pos_idx * vocab_size;

    // ========================================================================
    // STEP 1: Temperature Scaling
    // ========================================================================
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        output[offset + i] = input[offset + i] / temperature;
    }
    __syncthreads();

    // ========================================================================
    // STEP 2: Find Maximum (Warp-Level + Block-Level Reduction)
    // ========================================================================
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, output[offset + i]);
    }

    // Warp-level reduction
    for (int delta = 16; delta > 0; delta >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, delta));
    }

    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        shared_mem[warp_id] = local_max;
    }
    __syncthreads();

    // Final reduction across warps (only first warp participates)
    if (warp_id == 0) {
        float maximum = (lane_id < BLOCK_SIZE / 32) ? shared_mem[lane_id] : -FLT_MAX;
        for (int delta = 16; delta > 0; delta >>= 1) {
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, delta));
        }
        // Thread 0 writes final result to shared memory
        if (lane_id == 0) {
            shared_mem[0] = maximum;
        }
    }
    __syncthreads();

    // All threads read the final result
    float maximum = shared_mem[0];
    __syncthreads();

    // ========================================================================
    // STEP 3: Compute Mean
    // ========================================================================
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        local_sum += output[offset + i];
    }

    // Warp-level reduction
    for (int delta = 16; delta > 0; delta >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, delta);
    }

    if (lane_id == 0) {
        shared_mem[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float total_sum = (lane_id < BLOCK_SIZE / 32) ? shared_mem[lane_id] : 0.0f;
        for (int delta = 16; delta > 0; delta >>= 1) {
            total_sum += __shfl_down_sync(0xffffffff, total_sum, delta);
        }
        // Thread 0 writes final result to shared memory
        if (lane_id == 0) {
            shared_mem[0] = total_sum;
        }
    }
    __syncthreads();

    // All threads read the final result and compute mean
    float mean = shared_mem[0] / float(vocab_size);
    __syncthreads();

    // ========================================================================
    // STEP 4: Compute Variance
    // ========================================================================
    float local_var_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float diff = output[offset + i] - mean;
        local_var_sum += diff * diff;
    }

    // Warp-level reduction
    for (int delta = 16; delta > 0; delta >>= 1) {
        local_var_sum += __shfl_down_sync(0xffffffff, local_var_sum, delta);
    }

    if (lane_id == 0) {
        shared_mem[warp_id] = local_var_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float total_var_sum = (lane_id < BLOCK_SIZE / 32) ? shared_mem[lane_id] : 0.0f;
        for (int delta = 16; delta > 0; delta >>= 1) {
            total_var_sum += __shfl_down_sync(0xffffffff, total_var_sum, delta);
        }
        // Thread 0 writes final result to shared memory
        if (lane_id == 0) {
            shared_mem[0] = total_var_sum;
        }
    }
    __syncthreads();

    // All threads read the final result and compute variance/std
    float variance = shared_mem[0] / float(vocab_size);
    float std_dev = sqrtf(variance);
    __syncthreads();

    // ========================================================================
    // STEP 5: Apply Top-NSigma Filter
    // ========================================================================
    float threshold = maximum - top_nsigma * std_dev;

    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float val = output[offset + i];
        output[offset + i] = (val >= threshold) ? val : -INFINITY;
    }
}

// ============================================================================
// C++ LAUNCHER FUNCTIONS
// ============================================================================

extern "C" {

void launch_temperature_topnsigma(
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

    temperature_topnsigma_kernel<256><<<grid, block, 0, stream>>>(
        input,
        output,
        vocab_size,
        temperature,
        top_nsigma
    );
}

void launch_temperature_topnsigma_optimized(
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

    temperature_topnsigma_kernel_optimized<256><<<grid, block, 0, stream>>>(
        input,
        output,
        vocab_size,
        temperature,
        top_nsigma
    );
}

} // extern "C"
