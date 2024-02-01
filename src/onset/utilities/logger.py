import logging
import os
import sys
import traceback

# Redirect sys.stdout to the logger
class LoggerWriter:
    def __init__(self, logger, level=logging.DEBUG):
        self.logger = logger
        self.level = level

    def write(self, message):
        # Only log the message if it's not an empty string
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass

class StdErrHandler(logging.Handler):
    def __init__(self, error_log_file):
        super().__init__()
        self.error_log_file = error_log_file

    def emit(self, record):
        try:
            msg = self.format(record)
            with open(self.error_log_file, 'a') as f:
                f.write(msg + '\n')
        except Exception:
            self.handleError(record)

if 0:
    class NewLogger():
        def __init__(self):
            pid = os.getpid()
            log_file = f"logs/process_{pid}_log.txt"
            error_log_file = f"logs/process_{pid}_error.txt"

            # Configure the logger
            logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(levelname)s - %(message)s")
            # Create formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(levelname)s - %(message)s"
            )

            # Create logger with a dynamic name based on the process ID
            logger = logging.getLogger(f"onset-proc-{pid}")

            # Create console handler and set its level to debug
            # ch = logging.StreamHandler()
            # ch.setLevel(logging.DEBUG)

            # Create a handler for the logger to write to a file
            file_handler = logging.FileHandler(log_file)

            # Replace sys.stdout with the custom LoggerWriter
            # sys.stdout = LoggerWriter(logger)

            # error_handler = StdErrHandler(error_log_file)
            # error_handler.setLevel(logging.ERROR)
            # logger.addHandler(error_handler)

            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(formatter)

            logger.addHandler(consoleHandler)
            logger.info(f"Log file: {log_file}.")
            # logger.info(f"error file: {error_log_file}")
            # sys.stderr = error_handler
            self.logger = logger

        def get_logger(self):
            return self.logger
        
    logger = NewLogger().get_logger

pid = os.getpid()
os.makedirs("logs", exist_ok=True)
log_file = f"logs/process_{pid}_log.txt"
error_log_file = f"logs/process_{pid}_error.txt"

# Configure the logger
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(levelname)s - %(message)s")
# Create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(levelname)s - %(message)s"
)

# Create logger with a dynamic name based on the process ID
logger = logging.getLogger(f"onset-proc-{pid}")
# Create console handler and set its level to debug
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)

# Create a handler for the logger to write to a file
file_handler = logging.FileHandler(log_file)

# Replace sys.stdout with the custom LoggerWriter
# sys.stdout = LoggerWriter(logger)

# error_handler = StdErrHandler(error_log_file)
# error_handler.setLevel(logging.ERROR)
# logger.addHandler(error_handler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)

logger.addHandler(consoleHandler)
logger.info(f"Log file: {log_file}.")
# logger.info(f"error file: {error_log_file}")
# sys.stderr = error_handler
