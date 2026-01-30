"""AutoVoice server entry point."""
import argparse
import logging
import sys

from auto_voice.web.app import create_app


def parse_args():
    parser = argparse.ArgumentParser(description='AutoVoice Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--config', default='config/gpu_config.yaml', help='Config file path')
    return parser.parse_args()


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
        logging.warning(f"Config file {args.config} not found, using defaults")

    config['DEBUG'] = args.debug

    app, socketio = create_app(config=config)

    logging.info(f"Starting AutoVoice on {args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
