import logging
import os.path

from logging import Logger


def get_configured_logger(module_name: str) -> Logger:
    utils_dir = os.path.abspath(os.path.dirname(__file__))
    if not os.path.exists(log_dir := os.path.join(utils_dir, '..', 'logs')):
        os.makedirs(log_dir)
    path = os.path.join(log_dir, 'traversing_commoncrawl.log')
    handler = logging.FileHandler(path)
    formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')
    handler.setFormatter(formatter)

    logging.basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', )
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    logger.addHandler(handler)

    return logger
