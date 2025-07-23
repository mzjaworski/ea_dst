from typing import Tuple
import dataset.i_dataset
import torch.utils.data
import torchvision
import torch

class FashionMNIST(dataset.i_dataset.IDataset):

    def __init__(self, train_batch: int, test_batch: int, resize: Tuple[float, float], flatten: bool, preload: bool = False, device: torch.device = torch.device("cpu")):
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.resize = resize
        self.flatten = flatten
        self.preload = preload
        self.device = device

    def load(self) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.resize),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
            torchvision.transforms.Lambda(lambda x: torch.flatten(x)) if self.flatten else torchvision.transforms.Lambda(lambda x: x)
        ])

        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

        if self.preload:
            trainset = dataset.i_dataset.InMemoryDataset(trainset, device=self.device)
            testset = dataset.i_dataset.InMemoryDataset(testset, device=self.device)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch, shuffle=True, num_workers=0 if self.preload else 2, pin_memory=False if self.preload else True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.test_batch, shuffle=False, num_workers=0 if self.preload else 2, pin_memory=False if self.preload else True)

        return train_loader, test_loader