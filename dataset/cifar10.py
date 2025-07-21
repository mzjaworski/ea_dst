import dataset.i_dataset
import torch.utils.data
import torchvision
import torch

class CIFAR10(dataset.i_dataset.IDataset):

    def __init__(self, train_batch: int = 32, test_batch: int = 512, flatten: bool = True):
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.flatten = flatten

    def load(self) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            torchvision.transforms.Lambda(lambda x: torch.flatten(x)) if self.flatten else torchvision.transforms.Lambda(
                lambda x: x)  # needed for CNNs
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.test_batch, shuffle=False, num_workers=2, pin_memory=True)

        return train_loader, test_loader
