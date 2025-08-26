from abc import ABC, abstractmethod

class DecayStrategy(ABC):

    @abstractmethod
    def decay(self, value: float) -> float:
        pass

class NullDecay(DecayStrategy):

    def decay(self, value: float) -> float:
        return value

class LinearDecay(DecayStrategy):

    def __init__(self, step: float):
        self.step = step

    def decay(self, value: float) -> float:
        return max(0.0, value - self.step)

class ExponentialDecay(DecayStrategy):

    def __init__(self, factor: float):
        assert 0.0 <= factor <= 1.0, "factor must be in [0, 1]"
        self.factor = factor

    def decay(self, value: float) -> float:
        return value * self.factor
