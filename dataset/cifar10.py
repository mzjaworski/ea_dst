from typing import Tuple
import dataset.i_dataset
import dataset.transform
import torch.utils.data
import torchvision
import torch
import utilities

class CIFAR10(dataset.i_dataset.IDataset):

    def __init__(self, train_batch: int, test_batch: int, resize: Tuple[float, float], flatten: bool, preload: bool = False,
                 device: torch.device = torch.device("cpu"), corruption: torch.nn.Module = dataset.transform.NullCorruption()):
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.resize = resize
        self.flatten = flatten
        self.preload = preload
        self.device = device
        self.corruption = corruption if not isinstance(corruption, dict) else utilities.resolve_class(corruption["type"])(**corruption["kwargs"])

    def load(self) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
        transform = torchvision.transforms.Compose([
            # torchvision.transforms.Resize(self.resize),
            # torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # self.corruption,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torch.nn.functional.pad(x.unsqueeze(0), (4, 4, 4, 4), mode='reflect').squeeze()),
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomCrop(32),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),

            torchvision.transforms.Lambda(lambda x: torch.flatten(x)) if self.flatten else torchvision.transforms.Lambda(lambda x: x)  # needed for CNNs
        ])

        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            torchvision.transforms.Lambda(lambda x: torch.flatten(x)) if self.flatten else torchvision.transforms.Lambda(lambda x: x) # needed for CNNs
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

        if self.preload:
            trainset = dataset.i_dataset.InMemoryDataset(trainset, device=self.device)
            testset = dataset.i_dataset.InMemoryDataset(testset, device=self.device)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch, shuffle=True, num_workers=0 if self.preload else 2, pin_memory=False if self.preload else True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.test_batch, shuffle=False, num_workers=0 if self.preload else 2, pin_memory=False if self.preload else True)

        return train_loader, test_loader
