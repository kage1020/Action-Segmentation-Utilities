import logging
from dataclasses import dataclass


@dataclass(frozen=True)
class Color:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"


class Logger(logging.Logger):
    def __init__(self):
        super(Logger, self).__init__(name=__name__)
        self.logger = logging.getLogger(__name__)

    def info(self, message: str | dict | list | object):
        self.logger.info(Color.GREEN + self.__parse(message) + Color.RESET)

    def warn(self, message: str | dict | list | object):
        self.logger.warning(Color.YELLOW + self.__parse(message) + Color.RESET)

    def error(self, message: str | dict | list | object):
        self.logger.error(Color.RED + self.__parse(message) + Color.RESET)

    def __parse(self, message: str | dict | list | object):
        if isinstance(message, dict):
            return "\n".join([f"{k}: {v}" for k, v in message.items()])
        if isinstance(message, list):
            return "\n".join([str(m) for m in message])
        if isinstance(message, object):
            return str(message)
        return message
