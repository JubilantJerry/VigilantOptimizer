#!/usr/bin/env python3

import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension, CUDAExtension

path = os.path.dirname(os.path.abspath(__file__))
include_dirs = [os.path.join(path, 'include')]

ext_modules = []

ext_modules.append(CppExtension(
    name='vigilant._ext.cpu',
    sources=[
        'src/vigilant_ffi_cpu.cpp',
        'src/vigilant_update.cpp',
        'src/cuda_util_cpu.cpp'],
    extra_compile_args=['-O3', '-ffast-math']))

if torch.cuda.is_available():
    ext_modules.append(CUDAExtension(
        name='vigilant._ext.cuda',
        sources=[
            'src/vigilant_ffi_cuda.cpp',
            'src/vigilant_update.cu'],
        extra_compile_args={
            'cxx': ['-O3', '-ffast-math'],
            'nvcc': ['-O3', '-use_fast_math']}))
else:
    print("CUDA is not available. Only the CPU library will be built")

setup(
    name='vigilant',
    packages=['vigilant'],
    package_dir={'vigilant': 'vigilant_pkg'},
    ext_modules=ext_modules,
    include_dirs=include_dirs,
    cmdclass={'build_ext': BuildExtension},
)
