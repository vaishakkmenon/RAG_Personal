"""
Structured JSON Logging Configuration for Production.

This module provides:
- JSON-formatted log output for easy parsing by log aggregators
- Sensitive data redaction (API keys, passwords, tokens)
- Environment-aware configuration (JSON in production, human-readable in dev)
- Noise reduction from chatty libraries

Usage:
    from app.logging_config import setup_logging
    setup_logging()  # Call once at application startup
"""

import logging
import sys
import os
import json
import re
from datetime import datetime, timezone
from typing import Any


# ==============================================================================
# Sensitive Data Patterns for Redaction
# ==============================================================================

SENSITIVE_PATTERNS = [
    # API keys and tokens
    (
        re.compile(r'(api[_-]?key\s*[=:]\s*)["\']?[\w-]{20,}["\']?', re.IGNORECASE),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r'(token\s*[=:]\s*)["\']?[\w-]{20,}["\']?', re.IGNORECASE),
        r"\1[REDACTED]",
    ),
    (re.compile(r"(bearer\s+)[\w-]{20,}", re.IGNORECASE), r"\1[REDACTED]"),
    # Passwords
    (
        re.compile(r'(password\s*[=:]\s*)["\']?[^\s"\']+["\']?', re.IGNORECASE),
        r"\1[REDACTED]",
    ),
    (re.compile(r"(redis://:[^@]+@)", re.IGNORECASE), r"redis://:[REDACTED]@"),
    # Groq API key specific
    (re.compile(r"(gsk_)[a-zA-Z0-9]{20,}", re.IGNORECASE), r"\1[REDACTED]"),
    # Generic secrets
    (
        re.compile(r'(secret\s*[=:]\s*)["\']?[\w-]{10,}["\']?', re.IGNORECASE),
        r"\1[REDACTED]",
    ),
    # Session IDs (partial redaction - keep first 8 chars)
    (
        re.compile(
            r'(session[_-]?id\s*[=:]\s*)["\']?([a-f0-9]{8})[a-f0-9-]+["\']?',
            re.IGNORECASE,
        ),
        r"\1\2[...]",
    ),
]


def redact_sensitive_data(message: str) -> str:
    """
    Redact sensitive information from log messages.

    Args:
        message: The log message to redact

    Returns:
        Message with sensitive data replaced with [REDACTED]
    """
    if not isinstance(message, str):
        return str(message)

    result = message
    for pattern, replacement in SENSITIVE_PATTERNS:
        result = pattern.sub(replacement, result)

    return result


# ==============================================================================
# JSON Log Formatter
# ==============================================================================


class JSONFormatter(logging.Formatter):
    """
    Format log records as JSON for structured logging.

    Output includes:
    - timestamp: ISO 8601 format with timezone
    - level: Log level name
    - logger: Logger name
    - message: Log message (with sensitive data redacted)
    - module: Source module
    - function: Source function
    - line: Source line number
    - exception: Exception info if present
    - extra: Any additional fields passed to the log call
    """

    # Fields to exclude from 'extra' (standard LogRecord attributes)
    RESERVED_ATTRS = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        # Build the base log structure
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_sensitive_data(record.getMessage()),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = redact_sensitive_data(
                self.formatException(record.exc_info)
            )

        # Add any extra fields (from extra={} parameter)
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS:
                # Redact sensitive values
                if isinstance(value, str):
                    value = redact_sensitive_data(value)
                log_data[key] = value

        return json.dumps(log_data, default=str)


# ==============================================================================
# Human-Readable Formatter (for development)
# ==============================================================================


class ColoredFormatter(logging.Formatter):
    """
    Human-readable colored formatter for development.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        # Get color for level
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET if color else ""

        # Redact sensitive data from message
        record.msg = redact_sensitive_data(str(record.msg))

        # Format with color
        formatted = super().format(record)
        return f"{color}{formatted}{reset}"


# ==============================================================================
# Setup Function
# ==============================================================================


def setup_logging() -> None:
    """
    Configure application logging based on environment.

    - Production (ENV=production): JSON format to stdout
    - Development: Colored human-readable format

    Also configures:
    - Log level from LOG_LEVEL env var (default: INFO)
    - Noise reduction for chatty libraries
    - Sensitive data redaction in all output
    """
    env = os.getenv("ENV", "development")
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Validate log level
    if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        log_level = "INFO"

    # Clear any existing handlers (important for testing)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Choose formatter based on environment
    if env == "production":
        formatter = JSONFormatter()
    else:
        formatter = ColoredFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Reduce noise from chatty libraries
    noisy_loggers = [
        "uvicorn.access",
        "uvicorn.error",
        "httpx",
        "httpcore",
        "chromadb",
        "sentence_transformers",
        "transformers",
        "huggingface_hub",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured: level={log_level}, format={'JSON' if env == 'production' else 'colored'}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Use this instead of logging.getLogger() for consistency.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
