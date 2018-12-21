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
    THFloatTensor *grads
);

int stepUpdate(
    float maxUpdate,
    THFloatTensor *mean,
    THFloatTensor *meanSq,
    THFloatTensor *step,
    THFloatTensor *prevUpdate,
    float stepDecay,
    float stepFactorOverSampleSize,
    THFloatTensor *grads,
    THFloatTensor *data
);
