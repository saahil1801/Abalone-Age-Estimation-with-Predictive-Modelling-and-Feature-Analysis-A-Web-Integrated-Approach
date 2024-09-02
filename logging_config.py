# logging_config.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a handler for writing log messages to a file
    handler = RotatingFileHandler('app.log', maxBytes=1024*1024*10, backupCount=5)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Add the handler to the root logger
    logger.addHandler(handler)
