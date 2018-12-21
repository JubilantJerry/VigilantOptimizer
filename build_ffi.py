#!/usr/bin/env python3

import os
import torch
from torch.utils.ffi import create_extension

path = os.path.dirname(os.path.abspath(__file__))
include_dirs = [
    '/usr/local/cuda-9.2/include',
    os.path.join(path, 'include')]

ffi_cpu = create_extension(
    'vigilant._ext.cpu',
    include_dirs=[os.path.join(path, 'include')],
    headers=['include/vigilant_ffi_cpu.h'],
    sources=['src/vigilant_ffi_cpu.c'],
    extra_objects=[
        os.path.join(path, 'build/vigilant_update_cpu.so')
    ],
    define_macros=[('CUDA_EMULATE', None)],
    relative_to=__file__,
    with_cuda=False
)

ffi_cuda = create_extension(
    'vigilant._ext.cuda',
    include_dirs=[
        '/usr/local/cuda-9.2/include',
        os.path.join(path, 'include')
    ],
    headers=['include/vigilant_ffi_cuda.h'],
    sources=['src/vigilant_ffi_cuda.c'],
    extra_objects=[
        os.path.join(path, 'build/vigilant_update_cuda.so')
    ],
    define_macros=[],
    relative_to=__file__,
    with_cuda=True
)

if __name__ == '__main__':
    ffi_cpu.build()
    if torch.cuda.is_available():
        ffi_cuda.build()
    else:
        "CUDA is not available. Only the CPU library has been built"
