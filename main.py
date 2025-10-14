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

    # Update environment variables for logging config
    os.environ['LOG_LEVEL'] = args.log_level
    os.environ['LOG_FORMAT'] = args.log_format

    # Re-initialize logging with new settings
    setup_logging()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    if args.host:
        config['server']['host'] = args.host
    if args.port:
        config['server']['port'] = args.port
    if args.debug:
        config['server']['debug'] = args.debug

    # Log startup information
    logger.info(
        "Starting AutoVoice application",
        extra={
            "version": "1.0.0",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "host": config['server']['host'],
            "port": config['server']['port'],
            "debug": config['server'].get('debug', False)
        }
    )

    # Setup graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Create Flask app
        app, socketio = create_app(args.config)

        logger.info("Application initialized successfully")

        # Run the application
        socketio.run(
            app,
            host=config['server']['host'],
            port=config['server']['port'],
            debug=config['server']['debug']
        )
    except KeyboardInterrupt:
        logger.info("Shutdown initiated by user")
    except Exception as e:
        logger.error(
            "Failed to start application",
            extra={"error": str(e)},
            exc_info=True
        )
        sys.exit(1)
    finally:
        logger.info("AutoVoice application shutdown complete")


if __name__ == '__main__':
    main()