# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    CUDAExtension,
    CUDA_HOME,
    BuildExtension,
)
_DEBUG = False
_DEBUG_LEVEL = 0

# Common flags for both release and debug builds.
# extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
# extra_compile_args = ["-std=c++17"]
extra_compile_args = []
extra_compile_args += ["-DNDEBUG", "-O3", "-lineinfo"]
extra_compile_args = {
    'gcc': extra_compile_args,
    'nvcc': [*extra_compile_args, "--ptxas-options=-v"]
}

modules = []

if CUDA_HOME:
    modules.append(
        CUDAExtension(
            "deltacnn.cuda",
            [
                "src/cuda/common.cu",
                "src/cuda/conv_torch_wrapper.cpp", 
                "src/cuda/conv_kernel.cu", 
                "src/cuda/deconv_kernel.cu", 
                "src/cuda/other_nn_layers.cu",
            ],
			extra_compile_args=extra_compile_args,
            language='c++17'
        )
    )

setup(
    name="torchdeltacnn",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=modules,
    cmdclass={"build_ext": BuildExtension},
)