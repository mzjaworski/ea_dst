import logger.i_logger
import logging, sys
import uuid

class StdoutLogger(logger.i_logger.ILogger):

    def __init__(self):
        logger_name = f"StdoutLogger:{uuid.uuid4()}"
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = False
        self.logger.handlers.clear()
        self._stream = sys.stdout

        handler = logging.StreamHandler(self._stream)
        handler.setFormatter(self.get_formatter())

        self.logger.addHandler(handler)
        self.logger.setLevel(self.get_level())

    def log(self, data: dict) -> None:
        self.logger.info(self.format_message(data))
