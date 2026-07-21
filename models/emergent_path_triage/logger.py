"""Dedicated logging namespace for E-PATH-CO-REASON."""

from __future__ import annotations

import logging

# Establish the dedicated logging namespace
LOGGER_NAME = "models.emergent_path_triage"


def get_logger() -> logging.Logger:
    """Retrieve the E-PATH-CO-REASON structured logger instance."""
    logger = logging.getLogger(LOGGER_NAME)
    # Ensure it inherits configurations from parent loggers if set
    if not logger.handlers and logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    return logger
