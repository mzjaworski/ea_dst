from abc import ABC, abstractmethod
import torch.utils.data
import torch

class IDataset(ABC):

    @abstractmethod
    def load(self) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
        pass
