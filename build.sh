#!/usr/bin/env bash
set -x
gcc -I include --std=c99 -c -fPIC -O3 -g -DCUDA_EMULATE src/cuda_util_cpu.c -o build/cuda_util_cpu.o &&
gcc -I include --std=c99 --shared -fPIC -O3 -g -ffast-math -DCUDA_EMULATE build/cuda_util_cpu.o -x c src/vigilant_update.cu -o build/vigilant_update_cpu.so &&
nvcc -I include --std=c++11 --shared --compiler-options '-fPIC' -O3 -g --use_fast_math src/vigilant_update.cu -o build/vigilant_update_cuda.so &&
./build_ffi.py
