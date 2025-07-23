from abc import ABC, abstractmethod
from typing import Final
import logging

class ILogger(ABC):
    DEFAULT_FMT: Final[str] = "[%(asctime)s] %(message)s"
    DEFAULT_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"

    @staticmethod
    def get_level() -> int:
        return logging.INFO

    @staticmethod
    def format_message(data: dict) -> str:
        message = ""

        for key, value in data.items():
            message += f" [{key}::{value}]"

        return message

    @classmethod
    def get_formatter(cls) -> logging.Formatter:
        return logging.Formatter(cls.DEFAULT_FMT, datefmt=cls.DEFAULT_DATEFMT)

    @abstractmethod
    def log(self, data: dict) -> None:
        pass
