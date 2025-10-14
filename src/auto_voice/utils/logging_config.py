"""Structured logging configuration for AutoVoice."""

import logging
import logging.handlers
import os
import sys
import json
from typing import Dict, Any, Optional
from pathlib import Path
import socket
import threading


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self):
        super().__init__()
        self.hostname = socket.gethostname()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "hostname": self.hostname,
            "process": record.process,
            "thread": record.thread,
            "thread_name": record.threadName,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "created", "filename", "funcName",
                          "levelname", "levelno", "lineno", "module", "msecs",
                          "message", "pathname", "process", "processName",
                          "relativeCreated", "thread", "threadName", "exc_info",
                          "exc_text", "stack_info"]:
                log_data[key] = value

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output in development."""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m'  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive information from logs."""

    SENSITIVE_KEYS = {'password', 'token', 'api_key', 'secret', 'credential'}

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter sensitive data from log record."""
        # Redact sensitive data from message
        if isinstance(record.msg, dict):
            record.msg = self._redact_dict(record.msg)

        # Redact from args
        if record.args:
            record.args = tuple(
                self._redact_dict(arg) if isinstance(arg, dict) else arg
                for arg in record.args
            )

        return True

    def _redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive keys from dictionary."""
        return {
            key: '***REDACTED***' if any(s in key.lower() for s in self.SENSITIVE_KEYS) else value
            for key, value in data.items()
        }


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Set up structured logging configuration.

    Args:
        config: Optional logging configuration dictionary
    """
    # Get configuration from environment or use defaults
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_format = os.getenv('LOG_FORMAT', 'json').lower()
    log_dir = os.getenv('LOG_DIR', 'logs')

    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))

    if log_format == 'json':
        console_handler.setFormatter(JSONFormatter())
    else:
        # Use colored formatter for development
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

    console_handler.addFilter(SensitiveDataFilter())
    root_logger.addHandler(console_handler)

    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, 'autovoice.log'),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    file_handler.addFilter(SensitiveDataFilter())
    root_logger.addHandler(file_handler)

    # Separate error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, 'error.log'),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(error_handler)

    # Configure specific loggers
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('socketio').setLevel(logging.INFO)
    logging.getLogger('engineio').setLevel(logging.INFO)

    # Set up exception hook to log uncaught exceptions
    def exception_hook(exc_type, exc_value, exc_traceback):
        """Log uncaught exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        root_logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = exception_hook

    root_logger.info(
        "Logging configured",
        extra={"log_level": log_level, "log_format": log_format, "log_dir": log_dir}
    )


class LogContext:
    """Context manager for adding contextual information to logs."""

    _context = threading.local()

    def __init__(self, **kwargs):
        self.context = kwargs
        self.old_factory = None

    def __enter__(self):
        """Enter context manager."""
        # Store current context
        if not hasattr(self._context, 'data'):
            self._context.data = {}

        self._context.data.update(self.context)

        # Inject context into log records
        self.old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self._context.data.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        # Restore old factory
        if self.old_factory:
            logging.setLogRecordFactory(self.old_factory)

        # Clean up context
        for key in self.context:
            if hasattr(self._context, 'data'):
                self._context.data.pop(key, None)


def log_execution_time(operation_name: str):
    """Decorator to log function execution time."""
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                logger.info(
                    f"{operation_name} completed",
                    extra={
                        "operation": operation_name,
                        "execution_time": execution_time,
                        "function": func.__name__
                    }
                )

                # Log slow operations as warnings
                if execution_time > 5.0:
                    logger.warning(
                        f"{operation_name} took longer than expected",
                        extra={
                            "operation": operation_name,
                            "execution_time": execution_time,
                            "threshold": 5.0
                        }
                    )

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"{operation_name} failed",
                    extra={
                        "operation": operation_name,
                        "execution_time": execution_time,
                        "error": str(e)
                    },
                    exc_info=True
                )
                raise

        return wrapper
    return decorator
