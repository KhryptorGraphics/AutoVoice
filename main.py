"""AutoVoice server entry point."""
import argparse
import logging
import signal
import sys
import time

from auto_voice.web.app import create_app

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='AutoVoice Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--config', default='config/gpu_config.yaml', help='Config file path')
    return parser.parse_args()


def setup_signal_handlers(app):
    """Setup graceful shutdown handlers for SIGTERM and SIGINT."""
    def shutdown_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")

        job_manager = getattr(app, 'job_manager', None)
        if job_manager:
            logger.info("Stopping job manager...")
            try:
                job_manager.stop_cleanup_thread()
            except Exception as e:
                logger.warning(f"Error stopping job manager: {e}")

        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Clearing GPU memory...")
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Error clearing GPU memory: {e}")

        logger.info("Shutdown complete")
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s'
    )

    config = {}
    try:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    except (FileNotFoundError, ImportError):
        logger.warning(f"Config file {args.config} not found, using defaults")

    config['DEBUG'] = args.debug

    app, socketio = create_app(config=config)

    setup_signal_handlers(app)

    logger.info(f"Starting AutoVoice on {args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
