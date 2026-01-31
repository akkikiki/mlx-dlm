"""
Setup script for the OPTIMIZED fused kernel
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="temperature_topnsigma_optimized",
    ext_modules=[
        CUDAExtension(
            name="temperature_topnsigma_optimized",
            sources=[
                "pytorch_extension_optimized.cpp",
                "optimized_fused_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "--expt-relaxed-constexpr",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
