"""
Shared utilities: logging setup and reproducibility seeding.
"""

import logging
import os
import random
import sys
from typing import Optional

import numpy as np


_LOGGING_CONFIGURED = False


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """Configure root logger once; return the 'battery' logger."""
    global _LOGGING_CONFIGURED
    if not _LOGGING_CONFIGURED:
        handlers = [logging.StreamHandler(sys.stdout)]
        if log_file:
            handlers.append(logging.FileHandler(log_file))
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=handlers,
            force=True,
        )
        _LOGGING_CONFIGURED = True
    return logging.getLogger('battery')


def set_global_seed(seed: int) -> None:
    """Seed python, numpy, and tensorflow for reproducibility.

    PYTHONHASHSEED must be set in the environment before the Python process
    starts to guarantee hash determinism, but setting it here still covers
    most downstream code paths.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def sanitize_for_json(value):
    """Replace inf/nan with None so json.dump doesn't emit non-standard tokens."""
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    if isinstance(value, (np.integer,)):
        return int(value)
    return value
