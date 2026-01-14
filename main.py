#!/usr/bin/env python3
"""Main entry point for AutoVoice application."""

import os
import sys
import signal
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from auto_voice.web.app import create_app
from auto_voice.utils.config_loader import load_config
from auto_voice.utils.logging_config import setup_logging

# Initialize structured logging before any other imports
setup_logging()

import logging
logger = logging.getLogger(__name__)


def initialize_system(config_path=None, log_level='INFO', log_format='json'):
    """Initialize the AutoVoice system.

    Args:
        config_path: Path to configuration file (default: 'config/gpu_config.yaml')
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Logging format (json, text)

    Returns:
        Tuple of (app, socketio, config)
    """
    try:
        # Use default config path if not provided
        config_path = config_path or 'config/gpu_config.yaml'

        # Update environment variables for logging config
        os.environ['LOG_LEVEL'] = log_level
        os.environ['LOG_FORMAT'] = log_format

        # Setup logging with new settings
        setup_logging()

        # Load configuration
        config = load_config(config_path)

        # Create Flask app and SocketIO
        app, socketio = create_app(config_path)

        logger.info(
            "AutoVoice system initialized successfully",
            extra={
                "config_path": config_path,
                "log_level": log_level,
                "log_format": log_format
            }
        )

        return app, socketio, config

    except Exception as e:
        logger.error(
            "Failed to initialize AutoVoice system",
            extra={"error": str(e)},
            exc_info=True
        )
        raise


def run_app(app, socketio, config=None, host=None, port=None, debug=None):
    """Run the AutoVoice application.

    Args:
        app: Flask application instance
        socketio: SocketIO instance
        config: Configuration dictionary (optional, used to derive defaults)
        host: Host to bind to (default: from config or '0.0.0.0')
        port: Port to bind to (default: from config or 5000)
        debug: Enable debug mode (default: from config or False)
    """
    # Derive defaults from config if not explicitly provided
    if config is not None:
        host = host if host is not None else config.get('server', {}).get('host', '0.0.0.0')
        port = port if port is not None else config.get('server', {}).get('port', 5000)
        debug = debug if debug is not None else config.get('server', {}).get('debug', False)
    else:
        # Use hardcoded defaults if no config provided
        host = host if host is not None else '0.0.0.0'
        port = port if port is not None else 5000
        debug = debug if debug is not None else False

    # Setup graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Log startup information
    logger.info(
        "Starting AutoVoice server",
        extra={
            "host": host,
            "port": port,
            "debug": debug
        }
    )

    try:
        # Run the application
        socketio.run(
            app,
            host=host,
            port=port,
            debug=debug,
            allow_unsafe_werkzeug=True  # Required for Flask-SocketIO 5.x
        )
    except KeyboardInterrupt:
        logger.info("Shutdown initiated by user")
    except Exception as e:
        logger.error(
            "Server error",
            extra={"error": str(e)},
            exc_info=True
        )
        raise
    finally:
        logger.info("AutoVoice server shutdown complete")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description='AutoVoice GPU-accelerated voice synthesis')
    parser.add_argument(
        '--config',
        type=str,
        default='config/gpu_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to bind to'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default=os.getenv('LOG_LEVEL', 'INFO'),
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set logging level'
    )
    parser.add_argument(
        '--log-format',
        type=str,
        default=os.getenv('LOG_FORMAT', 'json'),
        choices=['json', 'text'],
        help='Set logging format'
    )
    args = parser.parse_args()

    try:
        # Initialize system
        app, socketio, config = initialize_system(
            config_path=args.config,
            log_level=args.log_level,
            log_format=args.log_format
        )

        # Override config with command line arguments
        host = args.host if args.host else config['server']['host']
        port = args.port if args.port else config['server']['port']
        debug = args.debug if args.debug else config['server'].get('debug', False)

        # Log startup information
        logger.info(
            "Starting AutoVoice application",
            extra={
                "version": "1.0.0",
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "host": host,
                "port": port,
                "debug": debug
            }
        )

        # Run the application
        run_app(app, socketio, config, host=host, port=port, debug=debug)

    except Exception as e:
        logger.error(
            "Failed to start application",
            extra={"error": str(e)},
            exc_info=True
        )
        sys.exit(1)


if __name__ == '__main__':
    main()