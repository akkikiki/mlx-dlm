/**
 * PyTorch C++ Extension for Temperature + Top-NSigma CUDA Kernel
 *
 * Build with:
 *   python setup.py build_ext --inplace
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "temperature_topnsigma_kernel.h"

// Forward declarations
void launch_temperature_topnsigma(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    float temperature,
    float top_nsigma,
    cudaStream_t stream
);

void launch_temperature_topnsigma_optimized(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    float temperature,
    float top_nsigma,
    cudaStream_t stream
);

// ============================================================================
// PyTorch Wrapper Functions
// ============================================================================

torch::Tensor temperature_topnsigma_cuda(
    torch::Tensor logits,
    float temperature,
    float top_nsigma,
    bool use_optimized
) {
    // Input validation
    TORCH_CHECK(logits.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(logits.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(logits.dim() == 3, "Input tensor must be 3D (batch, seq_len, vocab_size)");
    TORCH_CHECK(temperature > 0, "Temperature must be positive");
    TORCH_CHECK(top_nsigma >= 0, "top_nsigma must be non-negative");

    // Get dimensions
    const int batch_size = logits.size(0);
    const int seq_len = logits.size(1);
    const int vocab_size = logits.size(2);

    // Allocate output tensor
    auto output = torch::empty_like(logits);

    // Get CUDA stream (compatible with PyTorch 1.x and 2.x)
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(logits.device().index());

    // Launch kernel
    if (use_optimized) {
        launch_temperature_topnsigma_optimized(
            logits.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            seq_len,
            vocab_size,
            temperature,
            top_nsigma,
            stream
        );
    } else {
        launch_temperature_topnsigma(
            logits.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            seq_len,
            vocab_size,
            temperature,
            top_nsigma,
            stream
        );
    }

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor temperature_topnsigma(
    torch::Tensor logits,
    float temperature = 1.0,
    float top_nsigma = 2.0
) {
    return temperature_topnsigma_cuda(logits, temperature, top_nsigma, false);
}

torch::Tensor temperature_topnsigma_optimized(
    torch::Tensor logits,
    float temperature = 1.0,
    float top_nsigma = 2.0
) {
    return temperature_topnsigma_cuda(logits, temperature, top_nsigma, true);
}

// ============================================================================
// PyTorch Extension Bindings
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Temperature + Top-NSigma CUDA kernel for PyTorch";

    m.def("temperature_topnsigma",
          &temperature_topnsigma,
          "Apply temperature scaling and top-nsigma filtering (basic version)",
          py::arg("logits"),
          py::arg("temperature") = 1.0,
          py::arg("top_nsigma") = 2.0);

    m.def("temperature_topnsigma_optimized",
          &temperature_topnsigma_optimized,
          "Apply temperature scaling and top-nsigma filtering (optimized with warp primitives)",
          py::arg("logits"),
          py::arg("temperature") = 1.0,
          py::arg("top_nsigma") = 2.0);
}
