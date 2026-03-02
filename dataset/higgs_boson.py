from typing import Tuple
import dataset.i_dataset
import torch.utils.data
import torchvision
import torch
import utilities
import random
import numpy

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import default_collate
import urllib.request
import gzip
import os

class HiggsDataset(Dataset):
    URL = "https://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz"
    FILENAME = "HIGGS.csv.gz"

    def __init__(self, root: str = "./data", download: bool = True, subset: float = 1.0):
        self.root = root
        self.compressed_path = os.path.join(root, self.FILENAME)
        self.uncompressed_path = os.path.join(root, "HIGGS.csv")

        os.makedirs(root, exist_ok=True)

        # download
        if download and not os.path.exists(self.compressed_path):
            print("downloading higgs...")
            urllib.request.urlretrieve(self.URL, self.compressed_path)
            print("done")

        # extract
        if not os.path.exists(self.uncompressed_path):
            print("extracting higgs...")
            with gzip.open(self.compressed_path, "rt") as f_in, open(self.uncompressed_path, "w") as f_out:
                for line in f_in:
                    f_out.write(line)
            print("done")

        # TODO normalize

        # build line offsets for fast seek - llm
        self.offsets = []
        with open(self.uncompressed_path, "r") as f:
            offset = 0
            for line in f:
                self.offsets.append(offset)
                offset += len(line.encode("utf-8"))  # bytes to skip

        # select random subset
        if subset < 1.0:
            self.offsets = random.sample(self.offsets, int(len(self.offsets) * subset))

        self.length = len(self.offsets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: # llm
        offset = self.offsets[idx]
        with open(self.uncompressed_path, "r") as f:
            f.seek(offset)
            line = f.readline()
            parts = list(map(float, line.strip().split(",")))
            label = torch.tensor(parts[0], dtype=torch.long)
            features = torch.tensor(parts[1:], dtype=torch.float32)
            return features, label

class HiggsBoson(dataset.i_dataset.IDataset):

    def __init__(self, train_batch: int, test_batch: int, train_split: float = 0.8, subset: float = 1.0, device: torch.device = torch.device("cpu")):
        assert 1.0 > train_split > 0.0

        self.train_batch = train_batch
        self.test_batch = test_batch
        self.train_split = train_split
        self.subset = subset
        self.device = device

    def load(self):
        dataset = HiggsDataset(root='./data', download=True, subset=self.subset)

        train_size = int(self.train_split * len(dataset))
        test_size = len(dataset) - train_size

        train_ds, test_ds = random_split(dataset,[train_size, test_size], generator=torch.Generator().manual_seed(int(numpy.random.randint(0, 2 ** 32, dtype=numpy.uint32))))

        train_loader = DataLoader(train_ds, batch_size=self.train_batch, shuffle=True, num_workers=6, pin_memory=False, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=self.test_batch, shuffle=False, num_workers=6, pin_memory=False, drop_last=True)

        return train_loader, test_loader
