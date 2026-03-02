# File adapted from https://github.com/kuangliu/pytorch-cifar
# Copyright (c) 2017 liukuang
# Copyright (c) 2026 Mateusz Jaworski
# Licensed under the MIT License

'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from typing import List
import src.network.detail
import torch

class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int):
        super(BasicBlock, self).__init__()

        self.conv1 = src.network.detail.MaskedConv2D(torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = src.network.detail.MaskedConv2D(torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = torch.nn.Sequential(
                src.network.detail.MaskedConv2D(torch.nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)),
                torch.nn.BatchNorm2d(self.expansion*planes)
            )
        self.relu2 = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out

class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int):
        super(Bottleneck, self).__init__()
        self.conv1 = src.network.detail.MaskedConv2D(torch.nn.Conv2d(in_planes, planes, kernel_size=1, bias=False))
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = src.network.detail.MaskedConv2D(torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv3 = src.network.detail.MaskedConv2D(torch.nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False))
        self.bn3 = torch.nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = torch.nn.Sequential(
                src.network.detail.MaskedConv2D(torch.nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)),
                torch.nn.BatchNorm2d(self.expansion*planes)
            )
        self.relu3 = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


class ResNet(src.network.detail.NetworkInterface):

    def __init__(self, block: torch.nn.Module, num_blocks: List[int], num_classes: int, sparsity: float, method: str = 'uniform'):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = src.network.detail.MaskedConv2D(torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, sparsity=sparsity)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, sparsity=sparsity)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, sparsity=sparsity)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, sparsity=sparsity)
        self.linear = src.network.detail.MaskedLinear(torch.nn.Linear(512*block.expansion, num_classes))
        src.network.detail.initialize_masks(self, method=method, sparsity=sparsity)

    def _make_layer(self, block: torch.nn.Module, planes: int, num_blocks: int, stride: int, sparsity: float) -> torch.nn.Sequential:
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def compute_reachabilities(self) -> None:
        pass # TODO

def resnet18(**kwargs) -> src.network.detail.NetworkInterface:
    return ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], **kwargs)

def resnet34(**kwargs) -> src.network.detail.NetworkInterface:
    return ResNet(block=BasicBlock, num_blocks=[3, 4, 6, 3], **kwargs)

def resnet50(**kwargs) -> src.network.detail.NetworkInterface:
    return ResNet(block=Bottleneck, num_blocks=[3, 4, 6, 3], **kwargs)

def resnet101(**kwargs) -> src.network.detail.NetworkInterface:
    return ResNet(block=Bottleneck, num_blocks=[3, 4, 23, 3], **kwargs)

def resnet152(**kwargs) -> src.network.detail.NetworkInterface:
    return ResNet(block=Bottleneck, num_blocks=[3, 8, 36, 3], **kwargs)
