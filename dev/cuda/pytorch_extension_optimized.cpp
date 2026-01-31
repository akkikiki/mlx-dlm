/**
 * PyTorch extension for optimized fused kernel
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// Forward declarations (must match extern "C" in .cu file)
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
    );

    void launch_temperature_topnsigma_fused_warp(
        const float* input,
        float* output,
        int batch_size,
        int seq_len,
        int vocab_size,
        float temperature,
        float top_nsigma,
        cudaStream_t stream
    );
}

// Wrapper functions
torch::Tensor temperature_topnsigma_fused(
    torch::Tensor logits,
    float temperature,
    float top_nsigma
) {
    TORCH_CHECK(logits.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(logits.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(logits.dim() == 3, "Input must be 3D");

    const int batch_size = logits.size(0);
    const int seq_len = logits.size(1);
    const int vocab_size = logits.size(2);

    auto output = torch::empty_like(logits);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(logits.device().index());

    launch_temperature_topnsigma_fused(
        logits.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        vocab_size,
        temperature,
        top_nsigma,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor temperature_topnsigma_fused_warp(
    torch::Tensor logits,
    float temperature,
    float top_nsigma
) {
    TORCH_CHECK(logits.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(logits.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(logits.dim() == 3, "Input must be 3D");

    const int batch_size = logits.size(0);
    const int seq_len = logits.size(1);
    const int vocab_size = logits.size(2);

    auto output = torch::empty_like(logits);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(logits.device().index());

    launch_temperature_topnsigma_fused_warp(
        logits.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        vocab_size,
        temperature,
        top_nsigma,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused", &temperature_topnsigma_fused, "Fused kernel (basic)");
    m.def("fused_warp", &temperature_topnsigma_fused_warp, "Fused kernel (warp-optimized)");
}
