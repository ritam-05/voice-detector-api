import logging

def setup_logging(level=logging.INFO):
    fmt = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    logging.basicConfig(level=level, format=fmt)
    return logging.getLogger("voice_detector")
