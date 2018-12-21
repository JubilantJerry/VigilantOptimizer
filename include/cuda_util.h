#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#ifndef CUDA_EMULATE
    #include <cuda_runtime.h>

    #define TO_DIM3(x) x
    #define TOPLEVEL __global__
    #define DEVICE_FN __device__
    #define CONSTANT __constant__
    #define SHARED(type, name, ...) __shared__ type name __VA_ARGS__
    #define LOCAL(type, name) type name
    #define GRID_DIM gridDim
    #define BLOCK_IDX blockIdx
    #define BLOCK_DIM blockDim
    #define THREAD_IDX threadIdx
    #define CUDA_MALLOC(dest, size) cudaMalloc((void **)dest, size)
    #define CUDA_FREE cudaFree
    #define CUDA_MEMCPY cudaMemcpy
    #define CUDA_MEMCPY_CONST cudaMemcpyToSymbol
    #define DEREF_SHARED(val) val
    #define DEREF_LOCAL(val) val
    #define LAUNCH_PARAMS(...) (__VA_ARGS__)
    #define __ESC__(...) __VA_ARGS__
    #define LAUNCH_TOPLEVEL(function, params, ...) \
        function<<<__ESC__ params>>>(__VA_ARGS__)
    #define SYNC_TOPLEVEL() cudaThreadSynchronize()
    #define LAUNCH_SYNCED_THREADS(function, ...) \
        function(__VA_ARGS__);\
        __syncthreads()
    #define TOPLEVEL_SIMP TOPLEVEL
    #define LAUNCH_TOPLEVEL_SIMP(function, params, ...) \
        LAUNCH_TOPLEVEL(function, params, __VA_ARGS__)

#else
    #ifdef __cplusplus
        #include <cmath>
        extern "C" {
    #else
        #include <math.h>
    #endif

    typedef struct dim3 {
        int x;
        int y;
        int z;
    } dim3;

    #ifdef __cplusplus
        }
    #endif

    typedef void* cudaStream_t;

    extern dim3 gridDimDbg, blockIdxDbg, blockDimDbg, threadIdxDbg;

    #define TO_DIM3(x) ((dim3){x, 1, 1})
    #define TOPLEVEL
    #define DEVICE_FN
    #define CONSTANT
    #define SHARED(type, name, ...) \
        type name[gridDimDbg.x][gridDimDbg.y][gridDimDbg.z]__VA_ARGS__
    #define LOCAL(type, name, ...) \
        type name[gridDimDbg.x][gridDimDbg.y][gridDimDbg.z]\
                 [blockDimDbg.x][blockDimDbg.y][blockDimDbg.z]__VA_ARGS__
    #define GRID_DIM gridDimDbg
    #define BLOCK_IDX blockIdxDbg
    #define BLOCK_DIM blockDimDbg
    #define THREAD_IDX threadIdxDbg
    #define CUDA_MALLOC(dest, size) *(void **)dest = malloc(size)
    #define CUDA_FREE free
    #define CUDA_MEMCPY(dest, src, size, _) memcpy(dest, src, size)
    #define CUDA_MEMCPY_CONST(dest, src, size, ...) memcpy(&dest, src, size)
    #define DEREF_SHARED(val) \
        val[blockIdxDbg.x][blockIdxDbg.y][blockIdxDbg.z]
    #define DEREF_LOCAL(val) \
        val[blockIdxDbg.x][blockIdxDbg.y][blockIdxDbg.z]\
           [threadIdxDbg.x][threadIdxDbg.y][threadIdxDbg.z]
    #define LAUNCH_PARAMS(numBlocks, threadsPerBlock, ...) \
        gridDimDbg = numBlocks;\
        blockDimDbg = threadsPerBlock;
    #define LAUNCH_TOPLEVEL(function, params, ...) \
        params\
        function(__VA_ARGS__)
    #define SYNC_TOPLEVEL() ((void) 0)
    #ifndef LAUNCH_SYNCED_THREADS
    #define LAUNCH_SYNCED_THREADS(function, ...) \
        {\
        int gridX = gridDimDbg.x;\
        int gridY = gridDimDbg.y;\
        int gridZ = gridDimDbg.z;\
        int blockX = blockDimDbg.x;\
        int blockY = blockDimDbg.y;\
        int blockZ = blockDimDbg.z;\
        for (int bx = 0; bx < gridX; bx++) {\
        for (int by = 0; by < gridY; by++) {\
        for (int bz = 0; bz < gridZ; bz++) {\
            for (int tz = 0; tz < blockZ; tz++) {\
            for (int ty = 0; ty < blockY; ty++) {\
            for (int tx = 0; tx < blockX; tx++) {\
                blockIdxDbg.x = bx;\
                blockIdxDbg.y = by;\
                blockIdxDbg.z = bz;\
                threadIdxDbg.x = tx;\
                threadIdxDbg.y = ty;\
                threadIdxDbg.z = tz;\
                function(__VA_ARGS__);\
            }\
            }\
            }\
        }\
        }\
        }\
        }
    #endif
    #define TOPLEVEL_SIMP
    #define LAUNCH_TOPLEVEL_SIMP(function, params, ...) \
        params\
        LAUNCH_SYNCED_THREADS(function, __VA_ARGS__)
#endif

#endif