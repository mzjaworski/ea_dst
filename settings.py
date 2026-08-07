import logger.aggregate_logger
import logger.stdout_logger
import logger.csv_logger

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super(SingletonMeta, cls).__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class Settings(metaclass=SingletonMeta):

    def __init__(self, configuration):
        # if set we will initialize file logging
        self.log = configuration["log"] if "log" in configuration else None

        # store current experiment settings
        self.dataset = configuration["dataset"]
        self.model = configuration["model"]
        self.optimizer = configuration["optimizer"]
        self.lr_decay = configuration["lr_decay"]
        self.criterion = configuration["criterion"]
        self.warmup = configuration["warmup"]
        self.pruning = configuration["pruning"]
        self.regrowth = configuration["regrowth"]
        self.decay = configuration["decay"]
        self.mode = configuration["mode"]
        self.serialize = configuration["serialize"] if "serialize" in configuration else None

        self._logger = None

    @property
    def logger(self):
        if self._logger is None:
            loggers = [logger.stdout_logger.StdoutLogger()] + ([logger.csv_logger.CSVLogger(filename=self.log)] if self.log is not None else [])
            self._logger = logger.aggregate_logger.AggregateLogger(loggers)
        return self._logger
