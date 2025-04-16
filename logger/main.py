from dataclasses import dataclass
from logging import INFO, getLogger
from pprint import pformat
from sys import stdout


@dataclass(frozen=True)
class Color:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"


def log(
    message: str | dict | list | object,
    level: str = "info",
    prefix: str = "",
    end: str = "\n",
):
    if level == "info":
        print(
            prefix + Color.GREEN + pformat(message, width=100)[1:-1] + Color.RESET,
            end=end,
        )
    elif level == "warning":
        print(
            prefix + Color.YELLOW + pformat(message, width=100)[1:-1] + Color.RESET,
            end=end,
        )
    elif level == "error":
        print(
            prefix + Color.RED + pformat(message, width=100)[1:-1] + Color.RESET,
            end=end,
        )


def clear():
    stdout.write("\033[K")
    stdout.flush()


class Logger:
    def __init__(
        self,
        name: str = __name__,
        width: int = 100,
    ):
        self.logger = getLogger(name)
        self.logger.setLevel(INFO)
        self.width = width

    def info(
        self, message: str | dict | list | object, prefix: str = "", end: str = "\n"
    ):
        self.logger.info(
            pformat(message, width=self.width)[1:-1],
            extra={"prefix": prefix, "suffix": end},
        )

    def warning(
        self, message: str | dict | list | object, prefix: str = "", end: str = "\n"
    ):
        self.logger.warning(
            pformat(message, width=self.width)[1:-1],
            extra={"prefix": prefix, "suffix": end},
        )

    def error(
        self, message: str | dict | list | object, prefix: str = "", end: str = "\n"
    ):
        self.logger.error(
            pformat(message, width=self.width)[1:-1],
            extra={"prefix": prefix, "suffix": end},
        )

    def clear(self):
        stdout.write("\033[K")
        stdout.flush()
