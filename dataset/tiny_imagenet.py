# --------------- taken and adjusted from https://gist.github.com/z-a-f/b862013c0dc2b540cf96a123a6766e54 -------------- #

import imageio
import numpy as np
import os

from collections import defaultdict
from torch.utils.data import Dataset

from tqdm.autonotebook import tqdm
import PIL

import urllib.request
import zipfile

dir_structure_help = r"""
TinyImageNetPath
├── test
│   └── images
│       ├── test_0.JPEG
│       ├── t...
│       └── ...
├── train
│   ├── n01443537
│   │   ├── images
│   │   │   ├── n01443537_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01443537_boxes.txt
│   ├── n01629819
│   │   ├── images
│   │   │   ├── n01629819_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01629819_boxes.txt
│   ├── n...
│   │   ├── images
│   │   │   ├── ...
│   │   │   └── ...
├── val
│   ├── images
│   │   ├── val_0.JPEG
│   │   ├── v...
│   │   └── ...
│   └── val_annotations.txt
├── wnids.txt
└── words.txt
"""

def download_and_unzip(URL, root_dir):
    os.makedirs(root_dir, exist_ok=True)
    zip_path = os.path.join(root_dir, os.path.basename(URL))

    if os.path.exists(zip_path):
        print(f"dataset found under: {zip_path}")
        return

    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, zip_path)

    print(f"extracting to {root_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(root_dir)

def _add_channels(img, total_channels=3):
    while len(img.shape) < 3:  # third axis is the channels
        img = np.expand_dims(img, axis=-1)
    while(img.shape[-1]) < 3:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img

"""Creates a paths datastructure for the tiny imagenet.
Args:
  root_dir: Where the data is located
  download: Download if the data is not there
Members:
  label_id:
  ids:
  nit_to_words:
  data_dict:
"""
class TinyImageNetPaths:
    def __init__(self, root_dir, download=False):
        if download:
            download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip', root_dir)
        train_path = os.path.join(root_dir, 'tiny-imagenet-200', 'train')
        val_path = os.path.join(root_dir, 'tiny-imagenet-200', 'val')
        test_path = os.path.join(root_dir, 'tiny-imagenet-200', 'test')

        wnids_path = os.path.join(root_dir, 'tiny-imagenet-200', 'wnids.txt')
        words_path = os.path.join(root_dir, 'tiny-imagenet-200', 'words.txt')

        self._make_paths(train_path, val_path, test_path, wnids_path, words_path)

    def _make_paths(self, train_path, val_path, test_path, wnids_path, words_path):
        self.ids = []
        with open(wnids_path, 'r') as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                labels = list(map(lambda x: x.strip(), labels.split(',')))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
          'train': [],  # [img_path, id, nid, box]
          'val': [],  # [img_path, id, nid, box]
          'test': []  # img_path
        }

        # Get the test paths
        self.paths['test'] = list(map(lambda x: os.path.join(test_path, x), os.listdir(test_path)))

        # Get the validation paths and labels
        with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, 'images', fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths['val'].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
            imgs_path = os.path.join(train_path, nid, 'images')
            label_id = self.ids.index(nid)
            with open(anno_path, 'r') as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths['train'].append((fname, label_id, nid, bbox))

"""Datastructure for the tiny image dataset.
Args:
  root_dir: Root directory for the data
  mode: One of "train", "test", or "val"
  preload: Preload into memory
  load_transform: Transformation to use at the preload time
  transform: Transformation to use at the retrieval time
  download: Download the dataset
Members:
  tinp: Instance of the TinyImageNetPaths
  img_data: Image data
  label_data: Label data
"""
class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, download=False):
        tinp = TinyImageNetPaths(root_dir, download)
        self.mode = mode
        self.label_idx = 1  # from [image, id, nid, box]
        self.transform = transform
        self.transform_results = dict()

        self.samples = tinp.paths[mode]
        self.samples_num = len(self.samples)

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]
            img = _add_channels(imageio.imread(sample[0]))
            img = PIL.Image.fromarray(img, mode="RGB")
            label = None if self.mode == 'test' else torch.tensor(sample[self.label_idx])

            if self.transform:
                img = self.transform(img)

            return (img, label)
        except Exception as e:
            print(e)
            print(self.samples[idx])
            raise e


# ---------------------------------------------------- custom code ---------------------------------------------------- #

from typing import Tuple
import dataset.i_dataset
import torch.utils.data
import torchvision
import torch
import utilities

class TinyImageNet(dataset.i_dataset.IDataset):

    def __init__(self, train_batch: int, test_batch: int, resize: Tuple[float, float], flatten: bool, device: torch.device = torch.device("cpu")):
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.resize = resize
        self.flatten = flatten
        self.device = device

    def load(self) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.resize),
            torchvision.transforms.RandomResizedCrop(self.resize),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            torchvision.transforms.Lambda(lambda x: torch.flatten(x)) if self.flatten else torchvision.transforms.Lambda(lambda x: x)  # needed for CNNs
        ])

        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.resize),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            torchvision.transforms.Lambda(lambda x: torch.flatten(x)) if self.flatten else torchvision.transforms.Lambda(lambda x: x) # needed for CNNs
        ])

        trainset = TinyImageNetDataset(root_dir='./data', mode='train', transform=train_transform, download=True)
        testset = TinyImageNetDataset(root_dir='./data', mode='val', transform=test_transform, download=True) # test has no labels

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch, shuffle=True, num_workers=6, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.test_batch, shuffle=False, num_workers=6, pin_memory=True)

        return train_loader, test_loader
