import logging

from logging import Logger


def get_configured_logger(module_name: str) -> Logger:
    handler = logging.FileHandler('traversing_commoncrawl.log')
    formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')
    handler.setFormatter(formatter)
    logging.basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', )
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger
