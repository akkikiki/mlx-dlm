/**
 * Debug version of the CUDA kernel with detailed output
 *
 * This version prints intermediate values to help diagnose issues.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

// ============================================================================
// DEBUG KERNEL: Basic Version with Printfs
// ============================================================================

template<int BLOCK_SIZE>
__global__ void temperature_topnsigma_debug(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int vocab_size,
    const float temperature,
    const float top_nsigma,
    float* __restrict__ debug_info  // [num_positions, 4] for max, mean, std, threshold
) {
    __shared__ float shared_mem[BLOCK_SIZE];

    const int pos_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int offset = pos_idx * vocab_size;

    // ========================================================================
    // STEP 1: Temperature Scaling
    // ========================================================================
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        output[offset + i] = input[offset + i] / temperature;
    }
    __syncthreads();

    // Debug: Print first few values for position 0
    if (pos_idx == 0 && tid == 0) {
        printf("Position 0, After temperature scaling:\n");
        printf("  output[0] = %.6f\n", output[offset + 0]);
        printf("  output[1] = %.6f\n", output[offset + 1]);
        printf("  output[35] = %.6f\n", output[offset + 35]);
    }

    // ========================================================================
    // STEP 2: Find Maximum
    // ========================================================================
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, output[offset + i]);
    }
    shared_mem[tid] = local_max;
    __syncthreads();

    // Tree reduction
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + stride]);
        }
        __syncthreads();
    }
    float maximum = shared_mem[0];
    __syncthreads();

    if (pos_idx == 0 && tid == 0) {
        printf("  Maximum = %.6f\n", maximum);
    }

    // ========================================================================
    // STEP 3: Compute Mean
    // ========================================================================
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        local_sum += output[offset + i];
    }
    shared_mem[tid] = local_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    float total_sum = shared_mem[0];
    float mean = total_sum / float(vocab_size);
    __syncthreads();

    if (pos_idx == 0 && tid == 0) {
        printf("  Total sum = %.6f\n", total_sum);
        printf("  Mean = %.6f\n", mean);
    }

    // ========================================================================
    // STEP 4: Compute Variance
    // ========================================================================
    float local_var_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float diff = output[offset + i] - mean;
        local_var_sum += diff * diff;
    }
    shared_mem[tid] = local_var_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    float total_var_sum = shared_mem[0];
    float variance = total_var_sum / float(vocab_size);
    float std_dev = sqrtf(variance);
    __syncthreads();

    if (pos_idx == 0 && tid == 0) {
        printf("  Variance = %.6f\n", variance);
        printf("  Std dev = %.6f\n", std_dev);
    }

    // ========================================================================
    // STEP 5: Apply Top-NSigma Filter
    // ========================================================================
    float threshold = maximum - top_nsigma * std_dev;

    if (pos_idx == 0 && tid == 0) {
        printf("  Threshold = %.6f\n", threshold);
        printf("  top_nsigma = %.6f\n", top_nsigma);
    }

    // Count infs for debugging
    int local_inf_count = 0;
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float val = output[offset + i];
        if (val >= threshold) {
            output[offset + i] = val;
        } else {
            output[offset + i] = -INFINITY;
            local_inf_count++;
        }
    }

    // Count total infs
    shared_mem[tid] = (float)local_inf_count;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }

    if (pos_idx == 0 && tid == 0) {
        printf("  Total infs set: %d\n", (int)shared_mem[0]);
        printf("  output[35] after filter = %.6f\n", output[offset + 35]);
    }

    // Save debug info
    if (tid == 0) {
        debug_info[pos_idx * 4 + 0] = maximum;
        debug_info[pos_idx * 4 + 1] = mean;
        debug_info[pos_idx * 4 + 2] = std_dev;
        debug_info[pos_idx * 4 + 3] = threshold;
    }
}

// ============================================================================
// Launcher
// ============================================================================

extern "C" {

void launch_temperature_topnsigma_debug(
    const float* input,
    float* output,
    float* debug_info,
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

    temperature_topnsigma_debug<256><<<grid, block, 0, stream>>>(
        input,
        output,
        vocab_size,
        temperature,
        top_nsigma,
        debug_info
    );

    // Force sync to see printf output
    cudaStreamSynchronize(stream);
}

} // extern "C"
