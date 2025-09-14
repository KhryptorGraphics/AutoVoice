#!/usr/bin/env python3
"""Main entry point for AutoVoice application."""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from auto_voice.web.app import create_app
from auto_voice.utils.config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    if args.host:
        config['server']['host'] = args.host
    if args.port:
        config['server']['port'] = args.port
    if args.debug:
        config['server']['debug'] = args.debug

    logger.info("Starting AutoVoice application...")
    logger.info(f"Configuration: {config['server']}")

    try:
        # Create Flask app
        app, socketio = create_app(args.config)

        # Run the application
        socketio.run(
            app,
            host=config['server']['host'],
            port=config['server']['port'],
            debug=config['server']['debug']
        )
    except KeyboardInterrupt:
        logger.info("Shutting down AutoVoice...")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()