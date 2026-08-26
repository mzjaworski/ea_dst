import dataset.i_dataset

from pathlib import Path
from typing import Tuple
import csv
import os

import torch
import torch.utils.data
import torchvision
import torchvision.transforms.v2
from torch.utils.data import default_collate

# LLM generated against supplied template
class ImageNet(dataset.i_dataset.IDataset):

    def __init__(self, train_batch: int, test_batch: int, resize: Tuple[float, float], train_samples_per_class: int | None = None, num_classes: int | None = None, seed: int = 0, device: torch.device = torch.device("cpu")):
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.resize = resize
        self.train_samples_per_class = train_samples_per_class
        self.num_classes = num_classes
        self.seed = seed
        self.device = device

    def load(self) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
        root = (Path(__file__).resolve().parent / ".." / "data" / "imagenet-object-localization-challenge").resolve()

        if not root.is_dir():
            raise FileNotFoundError(
                f"ImageNet dataset not found at: {root}\n"
                "Download and extract it from: https://www.kaggle.com/competitions/imagenet-object-localization-challenge"
            )

        train_dir = root / "ILSVRC" / "Data" / "CLS-LOC" / "train"
        val_dir = root / "ILSVRC" / "Data" / "CLS-LOC" / "val"
        val_solution = root / "LOC_val_solution.csv"

        for path in [train_dir, val_dir, val_solution]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required ImageNet path not found: {path}\n"
                    "Download and extract the complete dataset from: https://www.kaggle.com/competitions/imagenet-object-localization-challenge"
                )

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(self.resize),
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

        val_imagefolder_dir = root / "val_imagefolder"
        val_imagefolder_dir.mkdir(parents=True, exist_ok=True)

        expected_targets = set()

        with val_solution.open("r", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                image_id = row["ImageId"]
                synset = row["PredictionString"].split()[0]

                source = val_dir / f"{image_id}.JPEG"
                class_dir = val_imagefolder_dir / synset
                target = class_dir / f"{image_id}.JPEG"

                if not source.is_file():
                    raise FileNotFoundError(f"ImageNet validation image not found: {source}")

                class_dir.mkdir(parents=True, exist_ok=True)
                expected_targets.add(target)

                if target.is_symlink():
                    current_source = target.resolve(strict=False)
                    expected_source = source.resolve()

                    if current_source != expected_source:
                        target.unlink()
                elif target.exists():
                    raise RuntimeError(f"Expected validation symlink but found regular filesystem entry: {target}")

                if not target.is_symlink():
                    relative_source = Path(os.path.relpath(source, start=target.parent))
                    target.symlink_to(relative_source)

        for path in val_imagefolder_dir.rglob("*"):
            if path.is_dir() or path in expected_targets:
                continue

            if path.is_symlink():
                path.unlink()
            else:
                raise RuntimeError(f"Unexpected regular file in ImageNet validation cache: {path}")

        for class_dir in val_imagefolder_dir.iterdir():
            if class_dir.is_dir() and not any(class_dir.iterdir()):
                class_dir.rmdir()

        trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transform)
        testset = torchvision.datasets.ImageFolder(root=val_imagefolder_dir, transform=test_transform)

        if len(trainset.classes) != 1000:
            raise RuntimeError(f"Expected 1000 ImageNet classes, found {len(trainset.classes)}")

        if len(testset) != 50000:
            raise RuntimeError(f"Expected 50000 ImageNet validation images, found {len(testset)}")

        if trainset.class_to_idx != testset.class_to_idx:
            raise RuntimeError("ImageNet training and validation class mappings do not match")

        num_classes = self.num_classes if self.num_classes is not None else 1000

        if num_classes <= 0 or num_classes > 1000:
            raise ValueError("num_classes must be between 1 and 1000")

        if self.train_samples_per_class is not None and self.train_samples_per_class <= 0:
            raise ValueError("train_samples_per_class must be greater than 0")

        generator = torch.Generator().manual_seed(self.seed)

        if num_classes < 1000:
            selected_classes = torch.randperm(1000, generator=generator)[:num_classes].tolist()
        else:
            selected_classes = list(range(1000))

        selected_classes = sorted(selected_classes)
        selected_class_set = set(selected_classes)
        class_remap = {old_class: new_class for new_class, old_class in enumerate(selected_classes)}

        train_indices_by_class = {class_index: [] for class_index in selected_classes}

        for index, target in enumerate(trainset.targets):
            if target in selected_class_set:
                train_indices_by_class[target].append(index)

        train_indices = []

        for class_index in selected_classes:
            indices = train_indices_by_class[class_index]

            if self.train_samples_per_class is not None:
                if len(indices) < self.train_samples_per_class:
                    raise RuntimeError(f"Requested {self.train_samples_per_class} samples for class {class_index}, but only {len(indices)} are available")

                permutation = torch.randperm(len(indices), generator=generator)
                indices = [indices[i] for i in permutation[:self.train_samples_per_class]]

            train_indices.extend(indices)

        test_indices = [index for index, target in enumerate(testset.targets) if target in selected_class_set]

        trainset.samples = [(path, class_remap[target] if target in selected_class_set else target) for path, target in trainset.samples]
        trainset.targets = [class_remap[target] if target in selected_class_set else target for target in trainset.targets]

        testset.samples = [(path, class_remap[target] if target in selected_class_set else target) for path, target in testset.samples]
        testset.targets = [class_remap[target] if target in selected_class_set else target for target in testset.targets]

        trainset = torch.utils.data.Subset(trainset, train_indices)
        testset = torch.utils.data.Subset(testset, test_indices)

        cutmix = torchvision.transforms.v2.CutMix(num_classes=num_classes, alpha=0.5)
        mixup = torchvision.transforms.v2.MixUp(num_classes=num_classes, alpha=0.5)
        cutmix_or_mixup = torchvision.transforms.v2.RandomChoice([cutmix, mixup])

        def collate_fn(batch):
            return cutmix_or_mixup(*default_collate(batch))

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch, shuffle=True, num_workers=6, pin_memory=False, drop_last=True, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.test_batch, shuffle=False, num_workers=6, pin_memory=False, drop_last=False)

        return train_loader, test_loader
