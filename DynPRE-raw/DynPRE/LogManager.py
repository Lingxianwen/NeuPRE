# -*- coding: utf-8 -*
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

import Constants


def init_logger(level=logging.DEBUG):

    formatter = logging.Formatter(Constants.LOG_FORMAT)

    logger = logging.getLogger()

    streamHandler = logging.StreamHandler(stream=sys.stdout)
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

    logger.setLevel(level)


class LogManager:
    def __init__(
        self,
        log_dir: str,
        tmp_file_name: str,
        level: int = logging.DEBUG,
        enable_stdout: bool = False,
        max_bytes: Optional[int] = None,
    ) -> None:
        """
        Using RotatingFileHandler if max_bytes is assigned with value
        """

        self.log_dir = log_dir
        self.level = level
        self.tmp_file_name = tmp_file_name

        self.formatter = logging.Formatter(Constants.LOG_FORMAT)

        logger = logging.getLogger()
        if max_bytes is None:
            fileHandler = logging.FileHandler(
                os.path.join(log_dir, self.tmp_file_name), mode="w"
            )
        else:
            fileHandler = RotatingFileHandler(
                os.path.join(log_dir, self.tmp_file_name),
                mode="a",
                maxBytes=max_bytes,
                backupCount=0,
            )
        fileHandler.setFormatter(self.formatter)
        logger.addHandler(fileHandler)

        if enable_stdout:
            streamHandler = logging.StreamHandler(stream=sys.stdout)
            streamHandler.setFormatter(self.formatter)
            logger.addHandler(streamHandler)

        logger.setLevel(level)

    def reset_fileHandler(self) -> None:
        logger = logging.getLogger()
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)

        fileHandler = logging.FileHandler(
            os.path.join(self.log_dir, self.tmp_file_name), mode="w"
        )
        fileHandler.setFormatter(self.formatter)
        logger.addHandler(fileHandler)
        logger.setLevel(self.level)
