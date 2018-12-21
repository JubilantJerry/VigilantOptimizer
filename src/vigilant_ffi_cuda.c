#include <THC/THC.h>
#include "vigilant_update.h"

extern THCState *state;

int statsUpdate(
        int minSample,
        THCudaIntTensor *sampleSize,
        THCudaTensor *mean,
        THCudaTensor *meanSq,
        THCudaTensor *oldMean,
        THCudaIntTensor *time,
        THCudaTensor *weight,
        THCudaTensor *weightedSampleSize,
        THCudaTensor *weightedAcceler,
        THCudaTensor *grads) {

    statsUpdateArgs args;
    args.numElems = THCudaTensor_numel(state, grads);
    args.minSample = minSample;
    args.sampleSize = THCudaIntTensor_data(state, sampleSize);
    args.mean = THCudaTensor_data(state, mean);
    args.meanSq = THCudaTensor_data(state, meanSq);
    args.oldMean = THCudaTensor_data(state, oldMean);
    args.time = THCudaIntTensor_data(state, time);
    args.weight = THCudaTensor_data(state, weight);
    args.weightedSampleSize = THCudaTensor_data(state, weightedSampleSize);
    args.weightedAcceler = THCudaTensor_data(state, weightedAcceler);
    args.grads = THCudaTensor_data(state, grads);

    statsUpdateImpl(args, THCState_getCurrentStream(state));

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        THError(cudaGetErrorString(err));
    }
    return 1;
}

int stepUpdate(
        float maxUpdate,
        THCudaTensor *mean,
        THCudaTensor *meanSq,
        THCudaTensor *step,
        THCudaTensor *prevUpdate,
        float stepDecay,
        float stepFactorOverSampleSize,
        THCudaTensor *grads,
        THCudaTensor *data) {

    stepUpdateArgs args;
    args.numElems = THCudaTensor_numel(state, grads);
    args.maxUpdate = maxUpdate;
    args.mean = THCudaTensor_data(state, mean);
    args.meanSq = THCudaTensor_data(state, meanSq);
    args.step = THCudaTensor_data(state, step);
    args.prevUpdate = THCudaTensor_data(state, prevUpdate);
    args.stepDecay = stepDecay;
    args.stepFactorOverSampleSize = stepFactorOverSampleSize;
    args.grads = THCudaTensor_data(state, grads);
    args.data = THCudaTensor_data(state, data);

    stepUpdateImpl(args, THCState_getCurrentStream(state));

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        THError(cudaGetErrorString(err));
    }
    return 1;
}