"""
Omnigent — Structured Logging

JSON logging to file with rotation.
Helps debug production issues without cluttering user output.
"""

import json
import logging
import logging.handlers
import sys
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for machine parsing."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for attr in ("tool_name", "provider", "cost"):
            if hasattr(record, attr):
                log_data[attr] = getattr(record, attr)

        return json.dumps(log_data)


def setup_logging(verbose: bool = False, logger_name: str = "omnigent"):
    """Setup structured logging for Omnigent.

    Args:
        verbose: If True, also log to console
        logger_name: Root logger name (override for domain implementations)

    Returns:
        Logger instance
    """
    log_dir = Path.home() / ".omnigent" / "logs"

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        root_logger = logging.getLogger(logger_name)
        root_logger.setLevel(logging.WARNING if not verbose else logging.INFO)

        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        root_logger.addHandler(console_handler)

        if verbose:
            root_logger.warning(f"Could not create log directory: {e}")

        return root_logger

    log_file = log_dir / f"{logger_name}.log"

    root_logger = logging.getLogger(logger_name)
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
    except (OSError, PermissionError) as e:
        if verbose:
            print(f"Warning: Could not create log file: {e}", file=sys.stderr)

    if verbose:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "[%(levelname)s] %(name)s: %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    root_logger.info(f"{logger_name} logging initialized", extra={"verbose": verbose})

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(f"omnigent.{name}")
