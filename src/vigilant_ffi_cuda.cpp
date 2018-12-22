#include "vigilant_ffi_cuda.h"

int statsUpdate(
        int minSample,
        at::Tensor sampleSize,
        at::Tensor mean,
        at::Tensor meanSq,
        at::Tensor oldMean,
        at::Tensor time,
        at::Tensor weight,
        at::Tensor weightedSampleSize,
        at::Tensor weightedAcceler,
        at::Tensor grads) {

    statsUpdateArgs args;
    args.numElems = grads.numel();
    args.minSample = minSample;
    args.sampleSize = sampleSize.data<int>();
    args.mean = mean.data<float>();
    args.meanSq = meanSq.data<float>();
    args.oldMean = oldMean.data<float>();
    args.time = time.data<int>();
    args.weight = weight.data<float>();
    args.weightedSampleSize = weightedSampleSize.data<float>();
    args.weightedAcceler = weightedAcceler.data<float>();
    args.grads = grads.data<float>();

    statsUpdateImpl(args);

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        AT_ERROR(cudaGetErrorString(err));
    }
    return 1;
}

int stepUpdate(
        at::Tensor mean,
        at::Tensor meanSq,
        at::Tensor step,
        at::Tensor prevUpdate,
        float stepDecay,
        float stepFactorOverSampleSize,
        at::Tensor grads,
        at::Tensor data) {

    stepUpdateArgs args;
    args.numElems = grads.numel();
    args.mean = mean.data<float>();
    args.meanSq = meanSq.data<float>();
    args.step = step.data<float>();
    args.prevUpdate = prevUpdate.data<float>();
    args.stepDecay = stepDecay;
    args.stepFactorOverSampleSize = stepFactorOverSampleSize;
    args.grads = grads.data<float>();
    args.data = data.data<float>();

    stepUpdateImpl(args);

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        AT_ERROR(cudaGetErrorString(err));
    }
    return 1;
}