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
        print(path)
        self.path = os.path.dirname(path)
        print(self.path)
        os.makedirs(self.path, exist_ok=True)

    def __next_file_name(self) -> str:
        data = f"{time.time_ns()}:{os.getpid()}".encode()
        h = hashlib.blake2b(data, digest_size=6).hexdigest()
        return f"{h}.pt"

    def serialize(self, models: List[torch.nn.Module]) -> None:
        for model in models:
            filename = self.__next_file_name()
            filepath = os.path.join(self.path, filename)
            torch.save(model.state_dict(), filepath)
