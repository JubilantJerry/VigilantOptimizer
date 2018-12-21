#include "vigilant_update.h"

#define THREADS_PER_BLOCK 1024
#define FINITE_DIFF_CONST 0.145f


#ifdef CUDA_EMULATE
static float rsqrtf(float value) {
    return 1.0f / sqrt(value);
}
#endif


TOPLEVEL_SIMP static void statsUpdateKernel(statsUpdateArgs args) {

    int idx = BLOCK_IDX.x * BLOCK_DIM.x + THREAD_IDX.x;
    if (idx >= args.numElems) {
        return;
    }

    float grad = args.grads[idx];

    // Update the statistics collector
    float sampleSize = args.sampleSize[idx];
    float mean = args.mean[idx];
    float meanSq = args.meanSq[idx];
    
    float sampleFrac = 1.0f / sampleSize;
    mean = (1.0f - sampleFrac) * mean + sampleFrac * grad;
    meanSq = (1.0f - sampleFrac) * meanSq + sampleFrac * (grad * grad);
    if (meanSq <= 0.0f) {
        meanSq = 1.0f;
    }

    float sqMean = mean * mean;
    float var = meanSq - sqMean;
    sampleSize += 2 * (sqMean * sampleSize < var) - 1;
    if (sampleSize < args.minSample) {
        sampleSize = args.minSample;
    }

    args.sampleSize[idx] = sampleSize;
    args.mean[idx] = mean;
    args.meanSq[idx] = meanSq;

    float rmsDenom = rsqrtf(meanSq);
    float weight = sqMean * rmsDenom;
    args.weight[idx] = weight;
    args.weightedSampleSize[idx] = weight * sampleSize;

    // Update the time 
    int time = args.time[idx];
    float acceler = 0.0f;

    time += 1;
    if (time > sampleSize / 2) {
        float oldMean = args.oldMean[idx];
        float oldMeanSign = (oldMean > 0.0f) ? 1.0f : -1.0f;
        acceler = 2 * (oldMeanSign * (oldMean - mean) <
                       oldMeanSign * (oldMean * FINITE_DIFF_CONST)) - 1;

        time = 0;
        args.oldMean[idx] = mean;
    }
    args.time[idx] = time;

    args.weightedAcceler[idx] = weight * acceler;
}

void statsUpdateImpl(statsUpdateArgs args, cudaStream_t stream) {

    int threadsPerBlock = THREADS_PER_BLOCK;
    #ifdef CUDA_EMULATE
        threadsPerBlock = args.numElems;
    #endif

    LAUNCH_TOPLEVEL_SIMP(
        statsUpdateKernel,
        LAUNCH_PARAMS(
            TO_DIM3((args.numElems + threadsPerBlock - 1) / threadsPerBlock),
            TO_DIM3(threadsPerBlock),
            0,
            stream
        ),
        args
    );
}


TOPLEVEL_SIMP static void stepUpdateKernel(stepUpdateArgs args) {

    int idx = BLOCK_IDX.x * BLOCK_DIM.x + THREAD_IDX.x;
    if (idx >= args.numElems) {
        return;
    }

    float grad = args.grads[idx];
    float stepDecay = args.stepDecay;
    float stepFactorOverSampleSize = args.stepFactorOverSampleSize;

    // Compute a new step size
    float meanSq = args.meanSq[idx];
    float step = args.step[idx];
    if (meanSq <= 0.0f) {
        meanSq = 1.0f;
    }

    step = stepDecay * step + \
           (1.0 - stepDecay) * grad * rsqrtf(meanSq);

    // Update the data
    float prevUpdate = args.prevUpdate[idx];
    float update = stepFactorOverSampleSize * step;
    if (update > args.maxUpdate) {
        update = args.maxUpdate;
    }
    if (update < -args.maxUpdate) {
        update = -args.maxUpdate;
    }

    args.step[idx] = step;
    args.data[idx] += (prevUpdate - update) - stepDecay * update;
    args.prevUpdate[idx] = stepDecay * update;
}


void stepUpdateImpl(stepUpdateArgs args, cudaStream_t stream) {
    int threadsPerBlock = THREADS_PER_BLOCK;
    #ifdef CUDA_EMULATE
        threadsPerBlock = args.numElems;
    #endif

    LAUNCH_TOPLEVEL_SIMP(
        stepUpdateKernel,
        LAUNCH_PARAMS(
            TO_DIM3((args.numElems + threadsPerBlock - 1) / threadsPerBlock),
            TO_DIM3(threadsPerBlock),
            0,
            stream
        ),
        args
    );
}
