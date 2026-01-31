"""
Setup script for building the Temperature + Top-NSigma CUDA extension

Build with:
    python setup.py build_ext --inplace

Usage:
    import temperature_topnsigma_cuda
    output = temperature_topnsigma_cuda.temperature_topnsigma(logits, temperature=1.5, top_nsigma=2.0)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# CUDA architecture flags (adjust for your GPU)
# Common values:
#   - RTX 30xx: 8.6
#   - RTX 40xx: 8.9
#   - A100: 8.0
#   - H100: 9.0
cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9;9.0")

setup(
    name="temperature_topnsigma_cuda",
    version="1.0.0",
    author="Your Name",
    description="CUDA kernel for fused temperature + top-nsigma sampling",
    ext_modules=[
        CUDAExtension(
            name="temperature_topnsigma_cuda",
            sources=[
                "pytorch_extension.cpp",
                "temperature_topnsigma_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",  # For profiling
                    "--expt-relaxed-constexpr",
                    f"-arch=compute_{cuda_arch_list.split(';')[0].replace('.', '')}",
                ],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
    ],
)
