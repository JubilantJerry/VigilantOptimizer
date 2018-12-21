#ifndef VIGILANT_UPDATE_H
#define VIGILANT_UPDATE_H

#include <cuda_util.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct statsUpdateArgs {
    int numElems;
    int minSample;
    int *sampleSize;
    float *mean;
    float *meanSq;
    float *oldMean;
    int *time;
    float *weight;
    float *weightedSampleSize;
    float *weightedAcceler;
    float *grads;
} statsUpdateArgs;

void statsUpdateImpl(statsUpdateArgs args, cudaStream_t stream);

typedef struct stepUpdateArgs {
    int numElems;
    float maxUpdate;
    float *mean;
    float *meanSq;
    float *step;
    float *prevUpdate;
    float stepDecay;
    float stepFactorOverSampleSize;
    float *grads;
    float *data;
} stepUpdateArgs;

void stepUpdateImpl(stepUpdateArgs args, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
