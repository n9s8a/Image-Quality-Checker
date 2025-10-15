import logging
import os

def get_logger(name, log_file=None, level=logging.INFO):
    """
    Creates a logger for a module/function.

    Args:
        name (str): Logger name, typically __name__.
        log_file (str, optional): File to save logs.
        level (int): Logging level (default INFO).

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger
