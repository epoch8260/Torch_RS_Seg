# -*- coding: utf-8 -*-
"""
Copyright (c) Yuwei Jin
Created on 2020-11-13 16:18
Written by Yuwei Jin (642281525@qq.com)
"""
import math
import bisect
import weakref
from functools import wraps
from typing import Union, Any, Optional

from torch.optim import Optimizer


def _map_value_to_list(input: Optional[Union[float, tuple, list]], target_list: list) -> list:
    """
    map the given value to a list that has the same length of target list.
    Note, the mapped elements in value would become float type.
    """
    if isinstance(input, tuple):  # covert tuple to list
        input = list(input)
    # when applying different gamma values for different param groups, the dimention between gamma and the base lrs of optimizer must be same!
    if isinstance(input, list):
        assert len(input) == len(target_list)
        value = list(map(float, input))
    else:
        input = float(input)
        value = [input for _ in target_list]

    return value


class _LRScheduler:
    def __init__(self,
                 optimizer: Optimizer,
                 adjust_period: Optional[int] = None,
                 eta_min_lr: Optional[Union[float, tuple, list]] = None):
        """Base class for learning scheduler class.
            Args:
                optimizer: Wrapped optimizer.
                adjust_period: Maximum number of adjust learning rate. Default: None.
                eta_min_lr: Minimum learning rate. Default: 0.
        """

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer

        self.adjust_period = int(adjust_period) if adjust_period is not None else 0
        self.warm_base_lrs = []
        self.warmup_period = 0  # warmup epochs

        # set initial learning rate for optimizer
        self.last_epoch = 1  # 从last_start开始后已经记录了多少个epoch
        if self.last_epoch == 1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified in param_groups[{}] when resuming an optimizer".format(i))

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `train.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned core here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0

        # get the base lrs of optimizer
        self.base_lrs = list(map(lambda group: group['initial_lr'], self.optimizer.param_groups))

        # set eta_min_lrs
        eta_min_lr = eta_min_lr if eta_min_lr is not None else 0.0
        eta_min_lrs = _map_value_to_list(eta_min_lr, self.base_lrs)
        for eta_min_lr, base_lr in zip(eta_min_lrs, self.base_lrs):
            assert eta_min_lr < base_lr
        self.eta_min_lrs = eta_min_lrs

    def warmup_init(self, warm_base_lr: Union[float, list, tuple], warmup_period: int):
        """Set the begin learning rate and warmup period for warm up.
            Args:
                warm_base_lr: Beginning learning rate of warm_up
                warmup_period: Number of warm_up epochs.
        """
        # ensure warmup_period > 0
        assert warmup_period > 0
        assert warm_base_lr is not None

        # ensure warmup_begin_lr is list or float type.
        warmup_base_lrs = _map_value_to_list(warm_base_lr, self.base_lrs)
        for warm_base_lr, base_lr in zip(warmup_base_lrs, self.base_lrs):
            assert warm_base_lr <= base_lr

        self.warm_base_lrs = warmup_base_lrs
        self.warmup_period = warmup_period

        for param_group, lr in zip(self.optimizer.param_groups, warmup_base_lrs):
            param_group['lr'] = lr

    def get_lr(self) -> list or tuple:
        # Compute learning rate using chainable form of the scheduler
        raise NotImplemented

    def get_warmup_lr(self) -> list:
        """Return the computed learning rate by current scheduler"""
        lrs = []
        for warm_base_lr, base_lr in zip(self.warm_base_lrs, self.base_lrs):
            lr = warm_base_lr + self.last_epoch * (base_lr - warm_base_lr) / self.warmup_period
            lrs.append(lr)
        return lrs

    def update_lr(self) -> None:
        values = self.get_lr()
        for data in zip(self.optimizer.param_groups, values):
            param_group, lr = data
            param_group['lr'] = lr

    def get_last_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, epoch=None) -> None:
        """Update the learning rate of next epoch. Note: it mast be called after optimizer.step().
            Args:
                epoch: Current epoch. Default: None.
        """
        # if self.optimizer._step_count == 0:
        #     warning ("Detected call of `train.step()` before `optimizer.step()`. ")

        if epoch is None:
            self.last_epoch += 1
            self.update_lr()
        else:
            if epoch < self.last_epoch:
                raise ValueError(
                    'Input epoch should be less than last.epoch. Get epoch: {}, last_epoch: {}'.format(epoch,
                                                                                                       self.last_epoch))
            self.last_epoch = epoch
            self.update_lr()


class ExponentialLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch"""

    def __init__(self,
                 optimizer: Optimizer,
                 gamma: Optional[Union[float, list, tuple]] = 0.9,
                 eta_min_lr: Optional[Union[float, list, tuple]] = None,
                 adjust_period: int = None):
        """
        Args:
            optimizer: Wrapped optimizer.
            gamma: Multiplicative factor of learning rate decay.
            eta_min_lr: learning rate. When the original learning rate is decayed to eta_min, the decay process is stopped.
            adjust_period: Maximum number of update epochs.
        """
        super(ExponentialLR, self).__init__(optimizer=optimizer, eta_min_lr=eta_min_lr, adjust_period=adjust_period)

        # when applying different gamma values for different param groups, the dimention between gamma and the base lrs of optimizer must be same!
        self.gamma = _map_value_to_list(gamma, self.base_lrs)

    def _get_closed_form_lr(self, curr_epoch) -> list:
        lrs = [(base_lr - eta_min_lr) * pow(gamma, curr_epoch) + eta_min_lr for gamma, base_lr, eta_min_lr in
               zip(self.gamma, self.base_lrs, self.eta_min_lrs)]
        return lrs

    def get_lr(self):
        if self.adjust_period:
            if self.last_epoch <= self.adjust_period:
                return self._get_closed_form_lr(self.last_epoch)
            else:
                return self.get_last_lr()
        else:
            return self._get_closed_form_lr(self.last_epoch)


class WarmExponentialLR(_LRScheduler):
    """Decay the learning rate of each parameter group by gamma every epoch with warmup strategy"""

    def __init__(self,
                 optimizer: Optimizer,
                 gamma: Optional[Union[float, list, tuple]] = 0.9,
                 eta_min_lr: Optional[Union[float, list, tuple]] = None,
                 adjust_period: int = None,
                 warm_period: int = 5,
                 warm_start_lr: Optional[Union[float, list, tuple]] = 1e-6):
        """
        Args:
           optimizer: Wrapped optimizer.
           gamma: Multiplicative factor of learning rate decay.
           eta_min_lr: Minimum learning rate. When the original learning rate is decayed to eta_min, the decay process is stopped.
           adjust_period: Maximum number of update epochs.
           warm_period: Warm up period. For WarmupExpontionalLR, it should be epoch.
           warm_start_lr: Warm_start_lr.
        """
        super().__init__(optimizer, adjust_period, eta_min_lr)
        self.warmup_init(warm_base_lr=warm_start_lr, warmup_period=warm_period)
        self.gamma = _map_value_to_list(gamma, self.base_lrs)

    def _get_closed_form_lr(self, curr_epoch) -> list:
        lrs = [(base_lr - eta_min_lr) * pow(gamma, curr_epoch) + eta_min_lr for gamma, base_lr, eta_min_lr in
               zip(self.gamma, self.base_lrs, self.eta_min_lrs)]
        return lrs

    def get_lr(self):
        if self.last_epoch <= self.warmup_period:
            return self.get_warmup_lr()
        else:
            if self.adjust_period == 0:
                curr_epoch = self.last_epoch - self.warmup_period
                return self._get_closed_form_lr(curr_epoch)
            else:
                if self.warmup_period < self.last_epoch <= self.adjust_period:
                    curr_epoch = self.last_epoch - self.warmup_period
                    return self._get_closed_form_lr(curr_epoch)
                else:
                    return self.get_last_lr()


class PolyLR(_LRScheduler):
    """Decays the learning rate of each parameter group by polynomial strategy every epoch"""

    def __init__(self,
                 optimizer: Optimizer,
                 max_iterations: int,
                 power: Optional[Union[float, list, tuple]] = 0.9,
                 eta_min_lr: Optional[Union[float, list, tuple]] = None):
        """
            Args:
                optimizer: Wrapped optimizer.
                power: Multiplicative factor of learning rate decay.
                eta_min_lr: Minimum learning rate. When the original learning rate is decayed to eta_min, the decay
                           process is stopped.
                max_iterations number of update epochs.
        """
        assert max_iterations > 0
        super(PolyLR, self).__init__(optimizer, eta_min_lr=eta_min_lr, adjust_period=max_iterations)

        self.power = _map_value_to_list(power, self.base_lrs)

    def _get_closed_form_lr(self, curr_epoch) -> list:
        lrs = []
        for power, base_lr, eta_min_lr in zip(self.power, self.base_lrs, self.eta_min_lrs):
            lr = (base_lr - eta_min_lr) * (1 - float(curr_epoch) / self.adjust_period) ** power + eta_min_lr
            lrs.append(lr)
        return lrs

    def get_lr(self):
        if self.last_epoch <= self.adjust_period:
            return self._get_closed_form_lr(self.last_epoch)
        else:
            return self.get_last_lr()


class WarmPolyLR(_LRScheduler):
    """Decays the learning rate of each parameter group by polynomial strategy every epoch with warmup strategy"""

    def __init__(self,
                 optimizer: Optimizer,
                 max_iterations: int,
                 power: Optional[Union[float, list, tuple]] = 0.9,
                 eta_min_lr: Optional[Union[float, list, tuple]] = None,
                 warm_period: int = 5,
                 warm_start_lr: Optional[Union[float, list, tuple]] = 1e-6
                 ):
        """
            Args:
                optimizer: Wrapped optimizer.
                power: Multiplicative factor of learning rate decay.
                eta_min_lr: Minimum learning rate. When the original learning rate is decayed to eta_min, the decay process is
                        stopped.
                max_iterations: Maximum number of update epochs.
        """

        super(WarmPolyLR, self).__init__(optimizer, eta_min_lr=eta_min_lr, adjust_period=max_iterations)
        self.power = _map_value_to_list(power, self.base_lrs)
        self.warmup_init(warm_base_lr=warm_start_lr, warmup_period=warm_period)

    def _get_closed_form_lr(self, curr_epoch) -> list:
        lrs = []
        for power, base_lr, eta_min_lr in zip(self.power, self.base_lrs, self.eta_min_lrs):
            lr = (base_lr - eta_min_lr) * (1 - float(curr_epoch) / self.adjust_period) ** power + eta_min_lr
            lrs.append(lr)
        return lrs

    def get_lr(self):
        if self.last_epoch <= self.warmup_period:
            return self.get_warmup_lr()
        elif self.warmup_period < self.last_epoch <= self.adjust_period:
            curr_epoch = self.last_epoch - self.warmup_period
            return self._get_closed_form_lr(curr_epoch)
        else:
            return self.get_last_lr()


class StepLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every step_size epochs."""

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        """
            Args:
                optimizer: Wrapped optimizer.
                step_size: Period of learning rate decay.
                gamma: Multiplicative factor of learning rate decay.
        """
        super(StepLR, self).__init__(optimizer, adjust_period=None, eta_min_lr=None)
        self.step_size = step_size
        self.gamma = float(gamma)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            lr = base_lr * self.gamma ** (self.last_epoch // self.step_size)
            lrs.append(lr)
        return lrs


class WarmStepLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every step_size epochs after warm_up."""

    def __init__(self,
                 optimizer,
                 step_size: int,
                 gamma: float = 0.1,
                 warm_period: int = 5,
                 warm_start_lr: Optional[Union[float, list, tuple]] = 1e-6
                 ):
        """
            Args:
                optimizer: Wrapped optimizer.
                step_size: Period of learning rate decay.
                gamma: Multiplicative factor of learning rate decay.
        """
        super(WarmStepLR, self).__init__(optimizer, adjust_period=None, eta_min_lr=None)
        self.step_size = step_size
        self.gamma = float(gamma)
        self.warmup_init(warm_base_lr=warm_start_lr, warmup_period=warm_period)

    def get_lr(self):
        if self.last_epoch <= self.warmup_period:
            return self.get_warmup_lr()
        else:
            curr_epoch = self.last_epoch - self.warmup_period
            lrs = [base_lr * self.gamma ** (curr_epoch // self.step_size) for base_lr in self.base_lrs]
            return lrs


class MultiStepLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, milestones: Union[tuple, list], gamma: float = 0.1):
        """Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the
        milestones.
            Args:
                milestones: List of epoch indices. Must be increasing.
                gamma:  Multiplicative factor of learning rate decay. Default: 0.1.
        """
        super(MultiStepLR, self).__init__(optimizer=optimizer, adjust_period=None, eta_min_lr=None)

        self.milestones = list(sorted(milestones))
        self.gamma = float(gamma)

    def get_lr(self):
        lrs = [base_lr * self.gamma ** bisect.bisect_right(self.milestones, self.last_epoch) for base_lr in
               self.base_lrs]
        return lrs


class WarmMultiStepLR(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 milestones: Union[tuple, list],
                 gamma: float = 0.1,
                 warm_period: int = 5,
                 warm_start_lr: Optional[Union[float, list, tuple]] = 1e-6):
        """Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the
        milestones.
            Args:
                milestones: List of epoch indices. Must be increasing.
                gamma:  Multiplicative factor of learning rate decay. Default: 0.1.
        """
        super(WarmMultiStepLR, self).__init__(optimizer=optimizer, adjust_period=None, eta_min_lr=None)

        self.milestones = list(sorted(milestones))
        self.gamma = float(gamma)
        self.warmup_init(warm_base_lr=warm_start_lr, warmup_period=warm_period)

    def get_lr(self):
        if self.last_epoch <= self.warmup_period:
            return self.get_warmup_lr()
        else:
            curr_epoch = self.last_epoch - self.warmup_period
            lrs = [base_lr * self.gamma ** bisect.bisect_right(self.milestones, curr_epoch) for base_lr in
                   self.base_lrs]
            return lrs


class CosineLR(_LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing schedule"""

    def __init__(self,
                 optimizer,
                 T_max: int,
                 eta_min_lr: Optional[Union[list, tuple, float]] = None):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            eta_min_lr (float): Minimum learning rate.
            T_max: Period of learning rate decay.
        """
        super(CosineLR, self).__init__(optimizer=optimizer, adjust_period=T_max, eta_min_lr=eta_min_lr)
        self.T_max = T_max

    def get_lr(self):
        if self.last_epoch <= self.adjust_period:
            lrs = []
            for base_lr, eta_min in zip(self.base_lrs, self.eta_min_lrs):
                lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(self.last_epoch * math.pi / self.T_max))
                lrs.append(lr)
            return lrs
        else:
            return self.get_last_lr()


class WarmCosineLR(_LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing schedule"""

    def __init__(self,
                 optimizer: Optimizer,
                 T_max: int,
                 eta_min_lr: Optional[Union[list, tuple, float]] = None,
                 warm_period: int = 5,
                 warm_start_lr: Optional[Union[float, list, tuple]] = 1e-6):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            eta_min_lr (float): Minimum learning rate.
            T_max: Period of warmup cosine learning rate decay.
        """
        super().__init__(optimizer, adjust_period=T_max, eta_min_lr=eta_min_lr)

        self.eta_min_lr = eta_min_lr
        self.T_max = T_max
        self.warmup_init(warm_base_lr=warm_start_lr, warmup_period=warm_period)

    def get_lr(self):
        if self.last_epoch <= self.warmup_period:
            return self.get_warmup_lr()
        elif self.warmup_period < self.last_epoch <= self.adjust_period:
            # curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr_scheduler']
            curr_epoch = self.last_epoch - self.warmup_period
            T_max = self.T_max - self.warmup_period
            lrs = []
            for base_lr, eta_min in zip(self.base_lrs, self.eta_min_lrs):
                lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(curr_epoch * math.pi / T_max))
                lrs.append(lr)
            return lrs
        else:
            return self.get_last_lr()


class STANetLR(_LRScheduler):
    def __init__(self, total_epoch: int, start_epoch: int, optimizer: Optimizer):
        """
        :param start_epoch: iter at starting learning rate
        """
        super().__init__(optimizer)

        self.total_epoch = total_epoch
        self.start_epoch = start_epoch

    def get_lr(self):
        if self.last_epoch < self.start_epoch:
            return self.base_lrs
        else:
            lrs = []
            for base_lr in self.base_lrs:
                d = self.last_epoch - self.start_epoch
                lr = base_lr - d * base_lr / (self.total_epoch - self.start_epoch)
                lrs.append(lr)
            return lrs


if __name__ == '__main__':
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.a = nn.Conv2d(3, 3, 1)
            self.b = nn.Conv2d(3, 3, 1)

        def forward(self, x):
            x = self.a
            return x

    import torch

    model = Net()
    # params = [
    #     {"params": model.a.parameters(), "lr_scheduler": 0.001},
    #     {"params": model.b.parameters(), "lr_scheduler": 0.01},
    # ]

    epochs = 60
    steps = 800

    lr = []
    x = []

    params = [
        {'params': model.a.parameters(), 'lr_scheduler': 0.001},
        {'params': model.b.parameters()}
    ]
    optimizer = torch.optim.Adam(params=params, weight_decay=0.0005, lr=0.001)

    lr_scheduler = StepLR(optimizer=optimizer, step_size=20, gamma=0.1)

    k = 0

    for i in range(epochs):
        for j in range(steps):
            optimizer.zero_grad()
            optimizer.step()

        curr_lr = optimizer.state_dict()['param_groups'][0]['lr']
        lr.append(curr_lr)
        k += 1
        x.append(k)

        print("the {} iter: {:.10f}".format(k, curr_lr))

        lr_scheduler.step()

    from matplotlib import pyplot as plt

    plt.plot(x, lr, 'r', linewidth=2)
    plt.ylabel('learning rate')
    plt.xlabel('epochs')
    plt.grid()
    plt.show()