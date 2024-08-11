import logging
from pprint import pformat
from dataclasses import dataclass


@dataclass(frozen=True)
class Color:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"


class Logger:
    def __init__(
        self,
        name: str = __name__,
        width: int = 100,
    ):
        self.logger = logging.getLogger(name)
        self.width = width

    @staticmethod
    def log(message: str | dict | list | object, level: str = "info"):
        if level == "info":
            print(Color.GREEN + pformat(message, width=100) + Color.RESET)
        elif level == "warning":
            print(Color.YELLOW + pformat(message, width=100) + Color.RESET)
        elif level == "error":
            print(Color.RED + pformat(message, width=100) + Color.RESET)

    def info(self, message: str | dict | list | object):
        self.logger.info(Color.GREEN + pformat(message, width=self.width) + Color.RESET)

    def warning(self, message: str | dict | list | object):
        self.logger.warning(
            Color.YELLOW + pformat(message, width=self.width) + Color.RESET
        )

    def error(self, message: str | dict | list | object):
        self.logger.error(Color.RED + pformat(message, width=self.width) + Color.RESET)
