from abc import ABC, abstractmethod
import torch

class IDataset(ABC):

    @abstractmethod
    def load(self, stack_level: int = 2) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
        pass
