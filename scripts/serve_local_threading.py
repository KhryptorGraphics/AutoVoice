#!/usr/bin/env python3
"""Local threading-mode AutoVoice server.

Bypasses the eventlet async_mode used by ``scripts/start_autovoice.sh`` /
``main.py`` for boxes where eventlet is unavailable (e.g. the Jetson Thor
py3.13 base env raises "Invalid async_mode specified"). ``create_app`` forces
Socket.IO threading mode whenever ``TESTING`` is set, while still initializing
the real ML components (KaraokeManager / VoiceCloner / JobManager) because the
``testing`` PARAMETER is intentionally left unset.

This module is import-safe: the server only starts under ``__main__``, so it is
safe as the entrypoint of a process whose Trainer uses spawn-based DataLoader
workers (the workers re-import this module under ``__mp_main__`` and must not
start a second server).

For local training verification only; production serving uses the eventlet
path in ``scripts/start_autovoice.sh``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10601)
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    args = ap.parse_args()

    from werkzeug.serving import make_server
    from auto_voice.web.app import create_app

    app, _socketio = create_app(config={
        "TESTING": True,  # threading Socket.IO mode (no eventlet)
        "DATA_DIR": str(Path(args.data_dir).resolve()),
        "singing_conversion_enabled": True,
        "voice_cloning_enabled": True,
    })
    # threaded=True: the web UI opens concurrent polls and Socket.IO falls back
    # to long-polling on werkzeug; a single-threaded server would starve every
    # other request behind one ~25s poll.
    server = make_server(args.host, args.port, app, threaded=True)
    print(f"serve_local_threading: {args.host}:{args.port} "
          f"data_dir={Path(args.data_dir).resolve()}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
