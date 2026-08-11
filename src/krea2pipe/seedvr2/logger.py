"""Logging shims for the ported SeedVR code (``common.logger`` / ``common.decorators``)."""

import logging
from functools import wraps


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or "krea2pipe.seedvr2")


def log_on_entry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        get_logger(func.__module__).debug("entering %s", func.__qualname__)
        return func(*args, **kwargs)

    return wrapper


def log_runtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time

        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        get_logger(func.__module__).debug(
            "%s took %.2fs", func.__qualname__, time.perf_counter() - t0
        )
        return result

    return wrapper
