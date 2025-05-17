"""
This module sets up global logging configurations for the application.

It reads the logging configuration from a YAML file, sets up the logging
directory, and provides a function to create and configure individual loggers.

Functions:
    create_logger: Create and configure a logger using global settings.

Usage Example:

.. code-block:: Python

    from logger_setup import create_logger

    logger = create_logger(__name__)
    logger.info("This is an info message.")
"""
import logging
import pathlib
import os
import sys
from typing import Tuple, Optional, Union
import errno
from reemission.utils import get_package_file
from reemission import registry

# Move this code inside create_logger

APP_CONFIG = registry.main_config.get("app_config")
# Set global logging settings from logging configuration
try:
    # logging.getLogger('test').setLevel(APP_CONFIG['logging']['level'])
    LOGGING_LEVEL = APP_CONFIG['logging']['level']
except (ValueError, TypeError):
    LOGGING_LEVEL = logging.DEBUG
# Create logging path
if APP_CONFIG['logging']['log_dir']:
    logging_path = get_package_file(APP_CONFIG['logging']['log_dir'])
else:
    logging_path = pathlib.Path.joinpath(get_package_file(), 'logs')
# Make logging path folder structure if not present
try:
    os.makedirs(logging_path)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

global_formatter = logging.Formatter(
    '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
global_filehandler = logging.FileHandler(
    pathlib.Path.joinpath(logging_path, APP_CONFIG['logging']['log_filename']),
    mode=APP_CONFIG['logging']['mode'])
global_streamhandler = logging.StreamHandler(sys.stdout)


def create_logger(
        logger_name: str,
        formatter: logging.Formatter = global_formatter,
        handlers: Tuple[logging.Handler, ...] = (
            global_filehandler, global_streamhandler),
        logging_level: Optional[Union[str, int]] = None) -> logging.Logger:
    """
    Create and setup a logger using global settings.

    Args:
        logger_name (str): Name of the logger, usually file name given in
            variable `__name__`.
        formatter (logging.Formatter): The logging formatter to use. Defaults to global_formatter.
        handlers (Tuple[logging.Handler, ...]): The logging handlers to use. Defaults to global_filehandler and global_streamhandler.
        logging_level (Optional[Union[str, int]]): The logging level to set. If None, uses the global logging level.

    Returns:
        logging.Logger: Initialized logger object.
    """
    log = logging.getLogger(logger_name)
    # Get a global logging level
    log.setLevel(LOGGING_LEVEL)
    # Set logging level to the value given in the argument
    if logging_level is not None:
        try:
            log.setLevel(logging_level)
        except (ValueError, TypeError):
            pass
    for handler in handlers:
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


if __name__ == "__main__":
    """Main entry point for the module."""
