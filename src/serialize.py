from abc import ABC, abstractmethod
from typing import List

import secrets
import torch
import time
import os
import hashlib

class Serialization(ABC):

    @abstractmethod
    def serialize(self, models: List[torch.nn.Module]) -> None:
        pass

class NullSerialization(Serialization):

    def serialize(self, models: List[torch.nn.Module]) -> None:
        pass

class FileSystemSerialization(Serialization):

    def __init__(self, path: str) -> None:
        self.path = os.path.splitext(path)[0] + ".pt"
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def serialize(self, models: List[torch.nn.Module]) -> None:
        for model in models:
            torch.save(model.state_dict(), self.path)
