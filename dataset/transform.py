import torch
import random

class RandomCorruption:

    def __init__(self, mean_passes: float = 2.0, corruption_prob: float = 0.85):
        self.mean_passes = mean_passes
        self.corruption_prob = corruption_prob

    def __call__(self, x):
        if random.random() < self.corruption_prob:
            num_passes = max(1, int(random.expovariate(1 / self.mean_passes)))
            for _ in range(num_passes):
                x = self.apply_random_corruption(x)
        return x

    def apply_random_corruption(self, x: torch.Tensor) -> torch.Tensor:
        corruption_type = random.choice(['gaussian', 'salt_pepper', 'blur', 'brightness', 'fft'])

        if corruption_type == 'gaussian':
            noise = torch.randn_like(x) * 0.1
            x = torch.clamp(x + noise, 0, 1)

        elif corruption_type == 'salt_pepper':
            mask = torch.rand_like(x)
            x[mask < 0.05] = 0
            x[mask > 0.95] = 1

        elif corruption_type == 'blur':
            x = x.unsqueeze(0)
            x = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            x = x.squeeze(0)

        elif corruption_type == 'brightness':
            factor = random.uniform(0.5, 1.5)
            x = torch.clamp(x * factor, 0, 1)

        elif corruption_type == 'fft':
            fft_x = torch.fft.fft2(x)
            mask = torch.rand_like(x) < 0.05
            fft_x = fft_x * (~mask)
            x = torch.fft.ifft2(fft_x).real
            x = torch.clamp(x, 0, 1)

        return x

class NullCorruption:

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x
