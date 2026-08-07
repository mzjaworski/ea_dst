import logger.i_logger
import csv
import os

class CSVLogger(logger.i_logger.ILogger):
    def __init__(self, filename: str = "./data/system_log.csv"):
        self.filename = filename
        self.fieldnames = None

        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def log(self, data: dict) -> None:
        file_exists = os.path.exists(self.filename)
        file_has_content = file_exists and os.path.getsize(self.filename) > 0

        if self.fieldnames is None:
            self.fieldnames = list(data.keys())

        with open(self.filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")

            if not file_has_content:
                writer.writeheader()

            writer.writerow(data)
