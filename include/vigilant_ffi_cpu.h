#ifndef VIGILANT_FFI_CPU_H
#define VIGILANT_FFI_CPU_H

#define CUDA_EMULATE

#include <torch/extension.h>
#include "vigilant_update.h"

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
    at::Tensor grads
);

int stepUpdate(
    at::Tensor mean,
    at::Tensor meanSq,
    at::Tensor step,
    at::Tensor deviation,
    float stepDecay,
    float stepFactorOverSampleSize,
    float baseLr,
    float deviationDecay,
    at::Tensor grads,
    at::Tensor data
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("statsUpdate", &statsUpdate, "Vigilant statsUpdate");
  m.def("stepUpdate", &stepUpdate, "Vigilant stepUpdate");
}

#endif