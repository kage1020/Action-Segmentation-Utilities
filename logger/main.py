import logging

class Color:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'


class Logger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def info(self, message: str | dict | list):
        self.logger.info(Color.GREEN + self.__parse(message) + Color.RESET)

    def warn(self, message: str | dict | list):
        self.logger.warning(Color.YELLOW + self.__parse(message) + Color.RESET)

    def error(self, message: str | dict | list):
        self.logger.error(Color.RED + self.__parse(message) + Color.RESET)

    def __parse(self, message: str | dict | list):
        if isinstance(message, dict):
            return "\n".join([f"{k}: {v}" for k, v in message.items()])
        if isinstance(message, list):
            return "\n".join([str(m) for m in message])
        return message
