from typing import Tuple
import dataset.i_dataset
import torch.utils.data
import torchvision
import torch

class STL10(dataset.i_dataset.IDataset):

    def __init__(self, train_batch: int, test_batch: int, resize: Tuple[float, float], flatten: bool):
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.resize = resize
        self.flatten = flatten

    def load(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.resize),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            torchvision.transforms.Lambda(lambda x: torch.flatten(x)) if self.flatten else torchvision.transforms.Lambda(lambda x: x)
        ])

        trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
        testset  = torchvision.datasets.STL10(root='./data', split='test',  download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch, shuffle=True,  num_workers=2, pin_memory=True)
        test_loader  = torch.utils.data.DataLoader(testset,  batch_size=self.test_batch, shuffle=False, num_workers=2, pin_memory=True)

        return train_loader, test_loader