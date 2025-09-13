from abc import ABC, abstractmethod
from typing import Sequence, Self

from typing_extensions import Optional


class DecayStrategy(ABC):

    @abstractmethod
    def update(self, value: float) -> Self:
        pass

    @abstractmethod
    def decay(self, value: float) -> float:
        pass

class NullDecay(DecayStrategy):

    def update(self, value: float) -> Self:
        return self

    def decay(self, value: float) -> float:
        return value

class LinearDecay(DecayStrategy):

    def __init__(self, step: float):
        self.step = step

    def update(self, value: float) -> Self:
        return self

    def decay(self, value: float) -> float:
        return max(0.0, value - self.step)

class ExponentialDecay(DecayStrategy):

    def __init__(self, factor: float):
        self.factor = factor

    def update(self, value: float) -> Self:
        return self

    def decay(self, value: float) -> float:
        return value * self.factor

class BetaDecayStrategy(DecayStrategy):

    def __init__(self, target: float, decay_range: Sequence[float], memory_decay: float = 0.95, gamma: float = 0.5, init: float | None = None):
        self.low, self.high = sorted(decay_range)
        self.target = float(target)
        self.memory_decay = float(memory_decay)
        self.gamma = float(gamma)

        if init is None:
            init = target

        # initialize based on implied target
        memory = 1.0 / (1.0 - memory_decay)
        self.success_count = init * memory
        self.failure_count = (1.0 - init) * memory

    def update(self, value: float) -> Self:
        self.success_count *= self.memory_decay
        self.failure_count *= self.memory_decay
        self.success_count += value
        self.failure_count += 1.0 - value
        return self

    def decay_target(self, strategy: DecayStrategy = NullDecay()) -> float:
        self.target = strategy.decay(self.target)
        return self.target

    def decay(self, value: float) -> float:
        total = self.success_count + self.failure_count
        p_hat = (self.success_count / total) if total > 0 else 0.5
        factor = (self.target / (p_hat + 1e-8)) ** self.gamma
        adjusted = value * factor

        print(self.target, p_hat, factor, adjusted)

        if adjusted < self.low:
            adjusted = self.low
        if adjusted > self.high:
            adjusted = self.high

        return adjusted
