from typing import List
import logger.i_logger

class AggregateLogger(logger.i_logger.ILogger):

    def __init__(self, loggers: List[logger.i_logger.ILogger] | List[dict]):
        self.loggers = []
        self.stack_level = 2 # capture callers scope

        for logger in loggers:
            self.loggers.append(logger)

        # recursively update stack levels
        self.set_stack_level(self.stack_level)

    def set_stack_level(self, stack_level: int = 2):
        self.stack_level = stack_level
        for logger in self.loggers:
            logger.set_stack_level(stack_level + 1)

    def log(self, data: dict) -> None:
        for logger in self.loggers:
            logger.log(data)
