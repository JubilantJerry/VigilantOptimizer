import torch
from torch.optim.optimizer import Optimizer
from ._ext import cpu as extCpu
from .curpos import get_pos, set_pos

if torch.cuda.is_available():
    from ._ext import cuda as extCuda


ACCELER_CONST = 5.0
RAMP_UP_POWER = 2.0


class Vigilant(Optimizer):
    """Implements Vigilant algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize
        or dicts defining parameter groups

        max_latency (int, optional): effective number of samples
        averaged to compute the update direction. Can be None to
        allow arbitrarily slow changes to the update direction.
        Default: 10

        init_lr (float, optional): initial learning rate.
        Default: 0.0001

        min_sample (int, optional): minimum sample size allowed
        to analyze the gradient. Default: 4
    """

    def __init__(self, params,
                 base_lr=0.0001,
                 min_sample=4,
                 max_sample=3000,
                 max_latency=10,
                 explore_dist=5):
        if not 0.0 <= base_lr:
            raise ValueError("Invalid base learning rate: {}"
                             .format(base_lr))
        if not 1 <= min_sample:
            raise ValueError("Invalid minimum sample size: {}"
                             .format(min_sample))
        if not min_sample <= max_sample:
            raise ValueError("Invalid maximum sample size: {}"
                             .format(max_sample))
        if not (max_latency is None or 1 <= max_latency):
            raise ValueError("Invalid max latency value: {}"
                             .format(max_latency))
        if not 1 <= explore_dist:
            raise ValueError("Invalid exploration distance: {}"
                             .format(explore_dist))

        init_step_factor = base_lr * min_sample / (max_sample ** RAMP_UP_POWER)

        defaults = dict(init_step_factor=init_step_factor,
                        base_lr=base_lr,
                        min_sample=min_sample,
                        max_sample=max_sample,
                        max_latency=max_latency,
                        explore_dist=explore_dist)
        super(Vigilant, self).__init__(params, defaults)

        self.print_info = {'lr': 0.0}

    def __setstate__(self, state):
        super(Vigilant, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_sum = 0
            sample_size_sum = 0
            acceler_sum = 0
            step_factor = None

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Vigilant does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state['sample_size'] = (
                        torch.zeros_like(p.data, dtype=torch.int) +
                        group['min_sample'])
                    state['mean'] = torch.zeros_like(p.data)
                    state['mean_sq'] = torch.zeros_like(p.data)

                    state['old_mean'] = torch.zeros_like(p.data)
                    state['time'] = \
                        torch.zeros_like(p.data, dtype=torch.int)

                    state['step'] = torch.zeros_like(p.data)
                    state['deviation'] = torch.zeros_like(p.data)

                    state['weight'] = torch.zeros_like(p.data)
                    state['weighted_sample_size'] = torch.zeros_like(p.data)
                    state['weighted_acceler'] = torch.zeros_like(p.data)

                if state['sample_size'].dtype != torch.int:
                    state['sample_size'] = state['sample_size'].int()

                if state['time'].dtype != torch.int:
                    state['time'] = state['time'].int()

                if 'step_factor' in state:
                    step_factor = state['step_factor']

            if step_factor is None:
                for p in group['params']:
                    self.state[p]['step_factor'] = (
                        self.state[p]['step'].new_tensor(
                            group['init_step_factor']))
                    step_factor = self.state[p]['step_factor']
                    break

            for p in group['params']:
                state = self.state[p]
                lib = extCuda if p.data.is_cuda else extCpu
                min_sample = group['min_sample']
                max_sample = group['max_sample']

                lib.statsUpdate(
                    min_sample,
                    state['sample_size'],
                    state['mean'],
                    state['mean_sq'],
                    state['old_mean'],
                    state['time'],
                    state['weight'],
                    state['weighted_sample_size'],
                    state['weighted_acceler'],
                    p.grad.data
                )

                weight = state['weight']
                weighted_sample_size = state['weighted_sample_size']
                weighted_acceler = state['weighted_acceler']
                weight_sum += weight.sum(
                    dim=tuple(range(weight.dim())))
                sample_size_sum += weighted_sample_size.sum(
                    dim=tuple(range(weighted_sample_size.dim())))
                acceler_sum += weighted_acceler.sum(
                    dim=tuple(range(weighted_acceler.dim())))

            weight_sum = weight_sum.item()
            if weight_sum <= 0.0:
                weight_sum = 1.0
            sample_size_mean = sample_size_sum.item() / weight_sum

            acceler_mean = (acceler_sum / weight_sum).item()
            step_factor.mul_(ACCELER_CONST ** acceler_mean).clamp_(max=1.0)

            max_latency = group['max_latency']
            if max_latency is None:
                sample_size_clipped = sample_size_mean
            else:
                sample_size_clipped = min(sample_size_mean, max_latency)
            step_decay = 1.0 - 1.0 / sample_size_clipped
            step_factor_over_sample_size = \
                step_factor.item() / sample_size_mean

            if sample_size_mean >= max_sample:
                step_factor.div_(2.0)
                for p in group['params']:
                    state = self.state[p]
                    state['sample_size'].div_(2).clamp_(min=min_sample)
                    state['time'].zero_()
                    state['old_mean'].copy_(state['mean'])

            deviation_decay = 1.0 - 1.0 / group['explore_dist']

            for p in group['params']:
                state = self.state[p]
                lib = extCuda if p.data.is_cuda else extCpu

                lib.stepUpdate(
                    state['mean'],
                    state['mean_sq'],
                    state['step'],
                    state['deviation'],
                    step_decay,
                    step_factor_over_sample_size,
                    group['base_lr'],
                    deviation_decay,
                    p.grad.data,
                    p.data
                )

        self.print_info['lr'] = step_factor_over_sample_size

        return loss

    def remove_deviation(self):
        for group in self.param_groups:
            for p in group['params']:
                deviation = self.state[p]['deviation']
                p.data.add_(-deviation)

    def restore_deviation(self):
        for group in self.param_groups:
            for p in group['params']:
                deviation = self.state[p]['deviation']
                p.data.add_(deviation)

    def show_step_factor(self):
        (_, col) = get_pos()
        print('\n\n\n', end='')
        print('\033[3A', end='')
        (line, _) = get_pos()
        set_pos(line, col)

        lr = self.print_info['lr']
        print('\n\n\033[2K\n' + "Learning rate: %0.8f" % lr, end='')
        set_pos(line, col)
