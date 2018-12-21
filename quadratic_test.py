#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import torch
from vigilant.optim import Vigilant

NUM_AVG = 1
NUM_DIM = 100


def main():
    data = (torch.zeros(NUM_DIM) if NUM_DIM > 1 else torch.tensor(0.0)) + 0.1
    param = [torch.nn.Parameter(data) for i in range(NUM_AVG)]
    optim = [Vigilant([param[i]], max_sample=None) for i in range(NUM_AVG)]

    progresses = []
    mean = torch.tensor(0.0)
    std = torch.tensor(0.1)

    time = 0

    while True:
        try:
            lr = 0.001 / (1.0 + 0.007 * time)
            time += 1

            avg_progress = 0.0
            for i in range(NUM_AVG):
                for g in optim[i].param_groups:
                    g['lr'] = lr

                optim[i].zero_grad()
                loss = 0.5 * param[i] ** 2
                loss.sum(dim=0).backward()
                param[i].grad.add_(torch.normal(mean, std))
                optim[i].step()

                avg_progress += (-0.5 * torch.log(loss)).sum(dim=0).item()
            avg_progress /= NUM_AVG * NUM_DIM

            progresses.append(avg_progress)

        except KeyboardInterrupt:
            break

    count = len(progresses)
    lo = 0
    hi = 0
    mean = 0.0
    i = 0
    smooth_progresses = []

    while True:
        start_ind = max(i / np.e, 0)
        end_ind = i * np.e + 1
        i += 1

        if end_ind >= count:
            break

        total = mean * (hi - lo)
        while lo < start_ind:
            total -= progresses[lo]
            lo += 1
        while hi < end_ind:
            total += progresses[hi]
            hi += 1
        mean = total / (hi - lo)
        smooth_progresses.append(mean)

    plt.plot(np.log10(1 + np.arange(len(smooth_progresses))),
             smooth_progresses)
    plt.show()


if __name__ == '__main__':
    main()
