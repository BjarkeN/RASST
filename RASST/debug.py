import logging, sys

# ======================
# Setup
level = "INFO"
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
# ======================

# ======================
# Possible values:
# CRITICAL
# ERROR
# WARNING
# INFO
# DEBUG
# NOTSET
# ======================

# ======================
# Setup functions
def debug(s):
    logging.debug(s)
def info(s):
    logging.info(s)
def warning(s):
    logging.warning(s)
def error(s):
    logging.warning(s)
def critical(s):
    logging.critical(s)
# ======================

# ======================
# Pass information to user
info("{level} level set".format(level=level))
# ======================