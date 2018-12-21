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
    THCudaTensor *grads
);

int stepUpdate(
    float maxUpdate,
    THCudaTensor *mean,
    THCudaTensor *meanSq,
    THCudaTensor *step,
    THCudaTensor *prevUpdate,
    float stepDecay,
    float stepFactorOverSampleSize,
    THCudaTensor *grads,
    THCudaTensor *data
);
