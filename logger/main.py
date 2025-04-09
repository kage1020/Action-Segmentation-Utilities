from dataclasses import dataclass
from logging import ERROR, INFO, WARNING, Formatter, StreamHandler, getLogger
from pprint import pformat
from sys import stdout


@dataclass(frozen=True)
class Color:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"


class ColorFormatter(Formatter):
    def format(self, record):
        if record.levelno == INFO:
            record.msg = Color.GREEN + record.msg + Color.RESET
        elif record.levelno == WARNING:
            record.msg = Color.YELLOW + record.msg + Color.RESET
        elif record.levelno == ERROR:
            record.msg = Color.RED + record.msg + Color.RESET
        prefix = getattr(record, "prefix", "")
        if prefix:
            message = f"{prefix}{record.msg}"
        else:
            message = super().format(record)
        suffix = getattr(record, "suffix", "")
        return message + suffix


class ColorStreamHandler(StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        self.setLevel(INFO)
        self.terminator = ""
        formatter = ColorFormatter(
            "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
        )
        self.setFormatter(formatter)


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
        handler = ColorStreamHandler(stdout)
        self.logger.addHandler(handler)
        self.logger.root.handlers = [
            h
            for h in self.logger.root.handlers
            if not getattr(h, "stream", None) == stdout
        ]
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
