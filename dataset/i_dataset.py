from abc import ABC, abstractmethod
import torch.utils.data
import torch

# wrapper for mapping whole dataset into memory instead of performing IO operations
class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, device: torch.device):
        self.data = []
        for i in range(len(dataset)):
            x, y = dataset[i]
            x = x.to(device)
            y = torch.tensor(y, device=device)
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class IDataset(ABC):

    @abstractmethod
    def load(self) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
        pass
