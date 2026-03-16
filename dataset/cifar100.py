from typing import Tuple
import dataset.i_dataset
import torch.utils.data
import torchvision
import torch
import utilities

from torch.utils.data import default_collate
import torchvision.transforms.v2

class CIFAR100(dataset.i_dataset.IDataset):

    def __init__(self, train_batch: int, test_batch: int, resize: Tuple[float, float], device: torch.device = torch.device("cpu")):
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.resize = resize
        self.device = device

    def load(self) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.resize),
            torchvision.transforms.RandomCrop(self.resize),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            torchvision.transforms.RandomErasing(p=0.25)
        ])

        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.resize),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        cutmix = torchvision.transforms.v2.CutMix(num_classes=100, alpha=0.5)
        mixup = torchvision.transforms.v2.MixUp(num_classes=100, alpha=0.5)
        cutmix_or_mixup = torchvision.transforms.v2.RandomChoice([cutmix, mixup])

        def collate_fn(batch):
            return cutmix_or_mixup(*default_collate(batch))

        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch, shuffle=True, num_workers=6, pin_memory=False, drop_last=True, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.test_batch, shuffle=False, num_workers=6, pin_memory=False, drop_last=True)

        return train_loader, test_loader
