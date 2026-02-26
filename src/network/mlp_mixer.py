# taken and adjusted from https://github.com/jaketae/mlp-mixer

# MIT License
#
# Copyright (c) 2021 Jake Tae
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from torch.nn import functional
import torch

import src.network.detail

class MLP(torch.nn.Module, src.network.detail.Graph):

    def __init__(self, num_features: int, hidden_size: int):
        super().__init__()
        self.fc1 = src.network.detail.MaskedLinear(torch.nn.Linear(num_features, hidden_size))
        self.act1 = torch.nn.GELU()
        self.norm1 = torch.nn.LayerNorm(hidden_size)
        self.fc2 = src.network.detail.MaskedLinear(torch.nn.Linear(hidden_size, num_features))

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.norm1(x)
        x = self.fc2(x)
        return x

    def forward_pass(self, input_connectivity: torch.Tensor) -> torch.Tensor:
        input_connectivity = self.fc1.forward_pass(input_connectivity)
        input_connectivity = self.fc2.forward_pass(input_connectivity)
        return input_connectivity

    def backward_pass(self, output_connectivity: torch.Tensor) -> torch.Tensor:
        output_connectivity = self.fc2.backward_pass(output_connectivity)
        output_connectivity = self.fc1.backward_pass(output_connectivity)
        return output_connectivity


class TokenMixer(torch.nn.Module, src.network.detail.Graph):

    def __init__(self, num_features: int, num_patches: int, hidden_size: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(num_features)
        self.mlp = MLP(num_patches, hidden_size)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        out = x + residual
        return out

    def forward_pass(self, input_connectivity: torch.Tensor) -> torch.Tensor:
        residual = input_connectivity
        x = input_connectivity.transpose(0, 1)
        x = self.mlp.forward_pass(x)
        x = x.transpose(0, 1)
        x = x | residual
        return x

    def backward_pass(self, output_connectivity: torch.Tensor) -> torch.Tensor:
        residual = output_connectivity
        x = output_connectivity.transpose(0, 1)
        x = self.mlp.backward_pass(x)
        x = x.transpose(0, 1)
        x = x | residual
        return x


class ChannelMixer(torch.nn.Module, src.network.detail.Graph):

    def __init__(self, num_features: int, hidden_size: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(num_features)
        self.mlp = MLP(num_features, hidden_size)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        out = x + residual
        return out

    def forward_pass(self, input_connectivity: torch.Tensor) -> torch.Tensor:
        residual = input_connectivity
        x = self.mlp.forward_pass(input_connectivity)
        x = x | residual
        return x

    def backward_pass(self, output_connectivity: torch.Tensor) -> torch.Tensor:
        residual = output_connectivity
        x = self.mlp.backward_pass(output_connectivity)
        x = x | residual
        return x


class MixerLayer(torch.nn.Module, src.network.detail.Graph):

    def __init__(self, num_features: int, num_patches: int, token_hidden: int, mixer_hidden: int):
        super().__init__()
        self.token_mixer = TokenMixer(num_features, num_patches, token_hidden)
        self.channel_mixer = ChannelMixer(num_features, mixer_hidden)

    def forward(self, x):
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        return x

    def forward_pass(self, input_connectivity: torch.Tensor) -> torch.Tensor:
        x = self.token_mixer.forward_pass(input_connectivity)
        x = self.channel_mixer.forward_pass(x)
        return x

    def backward_pass(self, output_connectivity: torch.Tensor) -> torch.Tensor:
        x = self.channel_mixer.backward_pass(output_connectivity)
        x = self.token_mixer.backward_pass(x)
        return x

def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
    num_patches = sqrt_num_patches ** 2
    return num_patches

class MLPMixer(src.network.detail.NetworkInterface, src.network.detail.Graph):
    def __init__(self, image_size: int = 256, patch_size: int = 16, in_channels: int = 3, num_features: int = 128, token_hidden: int = 256, mixer_hidden: int = 128, num_layers: int = 8, num_classes: int = 10, sparsity: float = 0.0, method: str = 'uniform'):
        super().__init__()
        num_patches = check_sizes(image_size, patch_size)
        print(num_patches)

        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.patcher = src.network.detail.MaskedConv2D(torch.nn.Conv2d(in_channels, num_features, kernel_size=patch_size, stride=patch_size))
        self.mixers = torch.nn.ModuleList([MixerLayer(num_features, num_patches, token_hidden, mixer_hidden) for _ in range(num_layers)])
        self.classifier = src.network.detail.MaskedLinear(torch.nn.Linear(num_features, num_classes))
        src.network.detail.initialize_masks(self, method=method, sparsity=sparsity)

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_features, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_features)

        embedding = patches
        for mixer in self.mixers:
            embedding = mixer(embedding)

        embedding = embedding.mean(dim=1)
        logits = self.classifier(embedding)
        return logits

    def forward_pass(self, input_connectivity: torch.Tensor) -> torch.Tensor:
        x = self.patcher.forward_pass(input_connectivity)
        num_features, h, w = x.shape
        x = x.permute(1, 2, 0).contiguous().view(-1, num_features)
        for mixer in self.mixers:
            x = mixer.forward_pass(x)
        self.x_shape = x.shape
        x = x.any(dim=0).expand_as(x)
        x = self.classifier.forward_pass(x)
        return x

    def backward_pass(self, output_connectivity: torch.Tensor) -> torch.Tensor:
        x = self.classifier.backward_pass(output_connectivity)
        x = x.unsqueeze(0).expand(self.x_shape)
        for mixer in reversed(self.mixers):
            x = mixer.backward_pass(x)
        num_patches, num_features = x.shape
        side = int(num_patches ** 0.5)
        x = x.view(side, side, num_features).permute(2, 0, 1).contiguous()
        x = self.patcher.backward_pass(x)
        return x

    def compute_reachabilities(self):
        device = next(self.parameters()).device

        forward = torch.ones((self.in_channels, self.image_size, self.image_size), dtype=torch.bool).to(device)
        backward = torch.ones(self.num_classes, dtype=torch.bool).to(device)

        self.forward_pass(forward)
        self.backward_pass(backward)

# convenience constructors
def mixer_s32(image_size: int = 224, num_classes: int = 200, sparsity: float = 0.0, method: str = 'uniform'):
    return MLPMixer(image_size=image_size, patch_size=32, num_layers=8, num_features=512, token_hidden=256, mixer_hidden=2048, num_classes=num_classes, sparsity=sparsity, method=method)

def mixer_s16(image_size: int = 224, num_classes: int = 200, sparsity: float = 0.0, method: str = 'uniform'):
    return MLPMixer(image_size=image_size, patch_size=16, num_layers=8, num_features=512, token_hidden=256, mixer_hidden=2048, num_classes=num_classes, sparsity=sparsity, method=method)

def mixer_s32a(image_size: int = 224, num_classes: int = 200, sparsity: float = 0.0, method: str = 'uniform'):
    return MLPMixer(image_size=image_size, patch_size=32, num_layers=9, num_features=512, token_hidden=256, mixer_hidden=2048, num_classes=num_classes, sparsity=sparsity, method=method)

def mixer_b32(image_size: int = 224, num_classes: int = 200, sparsity: float = 0.0, method: str = 'uniform'):
    return MLPMixer(image_size=image_size, patch_size=32, num_layers=12, num_features=768, token_hidden=384, mixer_hidden=3072, num_classes=num_classes, sparsity=sparsity, method=method)

def mixer_b16(image_size: int = 224, num_classes: int = 200, sparsity: float = 0.0, method: str = 'uniform'):
    return MLPMixer(image_size=image_size, patch_size=16, num_layers=12, num_features=768, token_hidden=384, mixer_hidden=3072, num_classes=num_classes, sparsity=sparsity, method=method)

def mixer_l32(image_size: int = 224, num_classes: int = 200, sparsity: float = 0.0, method: str = 'uniform'):
    return MLPMixer(image_size=image_size, patch_size=32, num_layers=24, num_features=1024, token_hidden=512, mixer_hidden=4096, num_classes=num_classes, sparsity=sparsity, method=method)

def mixer_l16(image_size: int = 224, num_classes: int = 200, sparsity: float = 0.0, method: str = 'uniform'):
    return MLPMixer(image_size=image_size, patch_size=16, num_layers=24, num_features=1024, token_hidden=512, mixer_hidden=4096, num_classes=num_classes, sparsity=sparsity, method=method)

def mixer_h14(image_size: int = 224, num_classes: int = 200, sparsity: float = 0.0, method: str = 'uniform'):
    return MLPMixer(image_size=image_size, patch_size=14, num_layers=32, num_features=1280, token_hidden=640, mixer_hidden=5120, num_classes=num_classes,sparsity=sparsity, method=method)
