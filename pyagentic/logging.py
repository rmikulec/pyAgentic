import logging
from logging.config import dictConfig

LOG_LEVEL = "WARNING"

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "colored": {
            "()": "colorlog.ColoredFormatter",
            "format": (
                "%(log_color)s[%(asctime)s]%(reset)s "
                "%(log_color)s%(levelname)-8s%(reset)s - "
                "%(name)s - %(message)s"
            ),
            "log_colors": {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
            "reset": True,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "colored",
            "level": LOG_LEVEL,
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": LOG_LEVEL,
    },
}

_configured = False


def configure_logging() -> None:
    global _configured
    if not _configured:
        dictConfig(LOGGING_CONFIG)
        _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
