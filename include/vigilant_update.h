#ifndef VIGILANT_UPDATE_H
#define VIGILANT_UPDATE_H

#include <cuda_util.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct statsUpdateArgs {
    int numElems;
    int minSample;
    int * __restrict__ sampleSize;
    float * __restrict__ mean;
    float * __restrict__ meanSq;
    float * __restrict__ oldMean;
    int * __restrict__ time;
    float * __restrict__ weight;
    float * __restrict__ weightedSampleSize;
    float * __restrict__ weightedAcceler;
    float * __restrict__ grads;
} statsUpdateArgs;

void statsUpdateImpl(statsUpdateArgs args);

typedef struct stepUpdateArgs {
    int numElems;
    float * __restrict__ mean;
    float * __restrict__ meanSq;
    float * __restrict__ step;
    float * __restrict__ prevUpdate;
    float stepDecay;
    float stepFactorOverSampleSize;
    float * __restrict__ grads;
    float * __restrict__ data;
} stepUpdateArgs;

void stepUpdateImpl(stepUpdateArgs args);

#ifdef __cplusplus
}
#endif

#endif
