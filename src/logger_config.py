import logging
import sys

def setup_logger(name="policy_simulator", level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        try:
            fh = logging.FileHandler('simulator.log')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception:
            pass
    return logger

log = setup_logger()
