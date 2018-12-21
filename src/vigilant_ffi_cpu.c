#include <TH/TH.h>
#include "vigilant_update.h"

int statsUpdate(
        int minSample,
        THIntTensor *sampleSize,
        THFloatTensor *mean,
        THFloatTensor *meanSq,
        THFloatTensor *oldMean,
        THIntTensor *time,
        THFloatTensor *weight,
        THFloatTensor *weightedSampleSize,
        THFloatTensor *weightedAcceler,
        THFloatTensor *grads) {

    statsUpdateArgs args;
    args.numElems = THFloatTensor_numel(grads);
    args.minSample = minSample;
    args.sampleSize = THIntTensor_data(sampleSize);
    args.mean = THFloatTensor_data(mean);
    args.meanSq = THFloatTensor_data(meanSq);
    args.oldMean = THFloatTensor_data(oldMean);
    args.time = THIntTensor_data(time);
    args.weight = THFloatTensor_data(weight);
    args.weightedSampleSize = THFloatTensor_data(weightedSampleSize);
    args.weightedAcceler = THFloatTensor_data(weightedAcceler);
    args.grads = THFloatTensor_data(grads);

    statsUpdateImpl(args, NULL);

    return 1;
}

int stepUpdate(
        float maxUpdate,
        THFloatTensor *mean,
        THFloatTensor *meanSq,
        THFloatTensor *step,
        THFloatTensor *prevUpdate,
        float stepDecay,
        float stepFactorOverSampleSize,
        THFloatTensor *grads,
        THFloatTensor *data) {

    stepUpdateArgs args;
    args.numElems = THFloatTensor_numel(grads);
    args.maxUpdate = maxUpdate;
    args.mean = THFloatTensor_data(mean);
    args.meanSq = THFloatTensor_data(meanSq);
    args.step = THFloatTensor_data(step);
    args.prevUpdate = THFloatTensor_data(prevUpdate);
    args.stepDecay = stepDecay;
    args.stepFactorOverSampleSize = stepFactorOverSampleSize;
    args.grads = THFloatTensor_data(grads);
    args.data = THFloatTensor_data(data);

    stepUpdateImpl(args, NULL);

    return 1;
}