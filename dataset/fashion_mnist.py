from typing import Tuple
import dataset.i_dataset
import torch.utils.data
import torchvision
import torch

class FashionMNIST(dataset.i_dataset.IDataset):

    def __init__(self, train_batch: int, test_batch: int, resize: Tuple[float, float], flatten: bool):
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.resize = resize
        self.flatten = flatten

    def load(self) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.resize),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
            torchvision.transforms.Lambda(lambda x: torch.flatten(x)) if self.flatten else torchvision.transforms.Lambda(lambda x: x)
        ])

        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.test_batch, shuffle=False, num_workers=2, pin_memory=True)

        return train_loader, test_loader