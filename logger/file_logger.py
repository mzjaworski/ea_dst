import logger.i_logger
import logging
import uuid
import os

class FileLogger(logger.i_logger.ILogger):

    def __init__(self, filename: str = "./data/system_log.log"):
        logger_name = f"FileLogger:{filename}:{uuid.uuid4()}"
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = False
        self.logger.handlers.clear()

        # create dir if absent
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True) # can fail when running multiple experiments

        self._file_handler = logging.FileHandler(filename)
        self._file_handler.setFormatter(self.get_formatter())
        self.logger.addHandler(self._file_handler)
        self.logger.setLevel(self.get_level())

    def log(self, data: dict) -> None:
        self.logger.info(self.format_message(data))
