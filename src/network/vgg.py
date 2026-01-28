# File adapted from https://github.com/kuangliu/pytorch-cifar
# Copyright (c) 2017 liukuang
# Copyright (c) 2026 Mateusz Jaworski
# Licensed under the MIT License

'''VGG11/13/16/19 in Pytorch.'''
from typing import List, Union
import src.network.detail
import torch

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(src.network.detail.NetworkInterface):

    def __init__(self, vgg_name: str, num_classes: int, sparsity: float, method: str = 'uniform'):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = src.network.detail.MaskedLayer(torch.nn.Linear(512, num_classes))
        src.network.detail.initialize_masks(self, method=method, sparsity=sparsity)

    def _make_layers(self, cfg: List[Union[str, int]]) -> torch.nn.Sequential:
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [src.network.detail.MaskedLayer(torch.nn.Conv2d(in_channels, x, kernel_size=3, padding=1)),
                           torch.nn.BatchNorm2d(x),
                           src.network.detail.ActivationFunctionWrapper(torch.nn.ReLU(inplace=True))]
                in_channels = x
        layers += [torch.nn.AvgPool2d(kernel_size=1, stride=1)]
        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

def vgg11(**kwargs) -> src.network.detail.NetworkInterface:
    return VGG(vgg_name='VGG11', **kwargs)

def vgg13(**kwargs) -> src.network.detail.NetworkInterface:
    return VGG(vgg_name='VGG13', **kwargs)

def vgg16(**kwargs) -> src.network.detail.NetworkInterface:
    return VGG(vgg_name='VGG16', **kwargs)

def vgg19(**kwargs) -> src.network.detail.NetworkInterface:
    return VGG(vgg_name='VGG19', **kwargs)
