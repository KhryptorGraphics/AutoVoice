"""
Flask web application factory for AutoVoice
"""
from flask import Flask
from flask_socketio import SocketIO
from typing import Tuple

from ..utils.config_loader import load_config


def create_app(config_path: str = None) -> Tuple[Flask, SocketIO]:
    """
    Create Flask application with SocketIO support.

    Args:
        config_path: Path to configuration file (optional)

    Returns:
        Tuple containing (Flask app, SocketIO instance)
    """
    app = Flask(__name__)

    # Load configuration if provided
    if config_path:
        config = load_config(config_path)
        app.config.update(config.get('web', {}))

    # Initialize SocketIO
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode='threading'
    )

    @app.route('/')
    def index():
        return {"message": "AutoVoice API", "status": "running"}

    @app.route('/health')
    def health():
        return {"status": "healthy"}

    @socketio.on('connect')
    def handle_connect():
        print("Client connected")

    @socketio.on('disconnect')
    def handle_disconnect():
        print("Client disconnected")

    return app, socketio