import logging
import logging.handlers
import os
import sys

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO # Change to logging.DEBUG for more verbose logs
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
LOG_BACKUP_COUNT = 3             # Keep 3 backup files (e.g., fakeseeker.log.1, .2, .3)

def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "fakeseeker.log")

    # Create rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8' # Specify encoding
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    # Console handler (no change needed)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    # Get the root logger and add handlers
    # Avoid basicConfig if adding handlers manually like this
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    # Clear existing handlers (important if this function is called multiple times)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info("Logging initialized with rotation.")

def get_logger(name):
    """Gets a logger instance."""
    return logging.getLogger(name)