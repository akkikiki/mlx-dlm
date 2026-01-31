/**
 * Header file for Temperature + Top-NSigma CUDA kernel
 */

#ifndef TEMPERATURE_TOPNSIGMA_KERNEL_H
#define TEMPERATURE_TOPNSIGMA_KERNEL_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Launch the temperature + top-nsigma kernel (basic version)
 *
 * @param input: Input logits tensor [batch_size, seq_len, vocab_size]
 * @param output: Output filtered logits [batch_size, seq_len, vocab_size]
 * @param batch_size: Batch size
 * @param seq_len: Sequence length
 * @param vocab_size: Vocabulary size
 * @param temperature: Temperature scaling factor
 * @param top_nsigma: Number of standard deviations for filtering
 * @param stream: CUDA stream (optional, default=0)
 */
void launch_temperature_topnsigma(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    float temperature,
    float top_nsigma,
    cudaStream_t stream = 0
);

/**
 * Launch the optimized temperature + top-nsigma kernel (with warp primitives)
 *
 * Uses warp shuffle instructions for faster reductions.
 * Requires CUDA 9.0 or later.
 *
 * @param input: Input logits tensor [batch_size, seq_len, vocab_size]
 * @param output: Output filtered logits [batch_size, seq_len, vocab_size]
 * @param batch_size: Batch size
 * @param seq_len: Sequence length
 * @param vocab_size: Vocabulary size
 * @param temperature: Temperature scaling factor
 * @param top_nsigma: Number of standard deviations for filtering
 * @param stream: CUDA stream (optional, default=0)
 */
void launch_temperature_topnsigma_optimized(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    float temperature,
    float top_nsigma,
    cudaStream_t stream = 0
);

#ifdef __cplusplus
}
#endif

#endif // TEMPERATURE_TOPNSIGMA_KERNEL_H
