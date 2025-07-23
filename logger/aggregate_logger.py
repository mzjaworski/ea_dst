from typing import List
import logger.i_logger

class AggregateLogger(logger.i_logger.ILogger):

    def __init__(self, loggers: List[logger.i_logger.ILogger] | List[dict]):
        self.loggers = []
        for logger in loggers:
            self.loggers.append(logger)

    def log(self, data: dict) -> None:
        for logger in self.loggers:
            logger.log(data)
