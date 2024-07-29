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
    ):
        self.logger = logging.getLogger(name)

    def info(self, message: str | dict | list | object):
        self.logger.info(Color.GREEN + pformat(message) + Color.RESET)

    def warning(self, message: str | dict | list | object):
        self.logger.warning(Color.YELLOW + pformat(message) + Color.RESET)

    def error(self, message: str | dict | list | object):
        self.logger.error(Color.RED + pformat(message) + Color.RESET)
