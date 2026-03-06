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
from sklearn.preprocessing import StandardScaler
import urllib.request
import gzip
import os

class HiggsDataset(Dataset):
    URL = "https://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz"
    FILENAME = "HIGGS.csv.gz"

    def __init__(self, root: str = "./data", download: bool = True, subset: float = 1.0, subset_seed: int = None):
        self.root = root
        self.subset_seed = subset_seed
        self.compressed_path = os.path.join(root, self.FILENAME)
        self.uncompressed_path = os.path.join(root, "HIGGS.csv")
        self.processed_path = os.path.join(root, f"HIGGS_{subset}_{subset_seed}.csv")

        os.makedirs(root, exist_ok=True)

        if not os.path.exists(self.processed_path):

            # download
            if download and not os.path.exists(self.compressed_path):
                print("downloading...")
                urllib.request.urlretrieve(self.URL, self.compressed_path)

            # extract
            if not os.path.exists(self.uncompressed_path):
                print("extracting...")
                with gzip.open(self.compressed_path, "rt") as f_in, open(self.uncompressed_path, "w") as f_out:
                    for line in f_in:
                        f_out.write(line)

            # build line offsets for fast seek - llm
            print("building offsets...")
            self.offsets = []
            with open(self.uncompressed_path, "r") as f:
                offset = 0
                for line in f:
                    self.offsets.append(offset)
                    offset += len(line.encode("utf-8"))  # bytes to skip

            # select random subset
            print("subsampling...")
            if subset < 1.0:
                rng = numpy.random.default_rng(seed=self.subset_seed)
                self.offsets = rng.choice(self.offsets, size=int(len(self.offsets) * subset), replace=False).tolist()

            # load random subset
            print("normalizing...")
            rows = [None for _ in range(len(self.offsets))]
            with open(self.uncompressed_path, "r") as f:
                for index, off in enumerate(self.offsets):
                    f.seek(off)
                    rows[index] = f.readline().strip()
            data = numpy.array([[float(x) for x in r.split(",")] for r in rows])
            y = data[:, 0]
            X = data[:, 1:]

            # normalize
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            processed = numpy.concatenate([y[:, None], X], axis=1)

            # serialize to file
            with open(self.processed_path, "w") as f:
                for row in processed:
                    f.write(",".join(map(str, row)) + "\n")

            print("done")

        # recover offsets from processed file
        self.offsets = []
        with open(self.processed_path, "r") as f:
            offset = 0
            for line in f:
                self.offsets.append(offset)
                offset += len(line.encode("utf-8"))

        self.length = len(self.offsets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        offset = self.offsets[idx]
        with open(self.processed_path, "r") as f:
            f.seek(offset)
            line = f.readline()
        parts = numpy.fromstring(line, sep=",", dtype=numpy.float32)
        label = torch.tensor(int(parts[0]), dtype=torch.long)
        features = torch.from_numpy(parts[1:])
        return features, label

class HiggsBoson(dataset.i_dataset.IDataset):

    def __init__(self, train_batch: int, test_batch: int, train_split: float = 0.8, subset: float = 1.0, subset_seed: int = None, device: torch.device = torch.device("cpu")):
        assert 1.0 > train_split > 0.0

        self.train_batch = train_batch
        self.test_batch = test_batch
        self.train_split = train_split
        self.subset = subset
        self.subset_seed = subset_seed
        self.device = device

    def load(self):
        dataset = HiggsDataset(root='./data', download=True, subset=self.subset, subset_seed=self.subset_seed)

        train_size = int(self.train_split * len(dataset))
        test_size = len(dataset) - train_size

        train_ds, test_ds = random_split(dataset,[train_size, test_size], generator=torch.Generator().manual_seed(int(numpy.random.randint(0, 2 ** 32, dtype=numpy.uint32))))

        train_loader = DataLoader(train_ds, batch_size=self.train_batch, shuffle=True, num_workers=6, pin_memory=False, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=self.test_batch, shuffle=False, num_workers=6, pin_memory=False, drop_last=True)

        return train_loader, test_loader
