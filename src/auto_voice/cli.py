"""Canonical CLI entrypoint for running the AutoVoice web server."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from auto_voice.config.loader import ConfigLoader, ConfigLoadError
from auto_voice.swarm.runner import (
    execute_manifest,
    load_manifest,
    prepare_resume_run,
    print_status,
    request_cancel,
    task_specs,
    topological_order,
)
from auto_voice.web.app import create_app

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/gpu_config.yaml")
DEFAULT_DATA_DIR = Path("data")


def _serve_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("serve", help="Start the AutoVoice web server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory for durable application data",
    )
    parser.add_argument(
        "--ssl-cert",
        default=None,
        help="Path to a TLS certificate for local HTTPS serving",
    )
    parser.add_argument(
        "--ssl-key",
        default=None,
        help="Path to the TLS private key for local HTTPS serving",
    )


def _swarm_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("swarm", help="Run DAG-based swarm manifests")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory for durable swarm run state",
    )
    swarm_subparsers = parser.add_subparsers(dest="swarm_command", required=True)

    run_parser = swarm_subparsers.add_parser("run", help="Execute a swarm manifest")
    run_parser.add_argument("--manifest", required=True, help="Path to the swarm manifest")
    run_parser.add_argument("--run-id", default=f"run-{int(time.time())}", help="Explicit run identifier")
    run_parser.add_argument("--dry-run", action="store_true", help="Write run state without executing tasks")

    validate_parser = swarm_subparsers.add_parser("validate", help="Validate a swarm manifest")
    validate_parser.add_argument("--manifest", required=True, help="Path to the swarm manifest")

    status_parser = swarm_subparsers.add_parser("status", help="Print status for an existing swarm run")
    status_parser.add_argument("--run-id", required=True, help="Run identifier to inspect")

    cancel_parser = swarm_subparsers.add_parser("cancel", help="Request cancellation for an active swarm run")
    cancel_parser.add_argument("--run-id", required=True, help="Run identifier to cancel")
    cancel_parser.add_argument("--reason", default="cancel_requested", help="Cancellation reason")

    resume_parser = swarm_subparsers.add_parser("resume", help="Resume unfinished work from a prior swarm run")
    resume_parser.add_argument("--source-run-id", required=True, help="Existing run identifier to resume from")
    resume_parser.add_argument("--run-id", required=True, help="New run identifier for the resumed run")

    retry_parser = swarm_subparsers.add_parser("retry", help="Retry failed or skipped work from a prior swarm run")
    retry_parser.add_argument("--source-run-id", required=True, help="Existing run identifier to retry from")
    retry_parser.add_argument("--run-id", required=True, help="New run identifier for the retry run")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    raw_args = list(argv if argv is not None else sys.argv[1:])
    if not raw_args or raw_args[0].startswith("-"):
        raw_args = ["serve", *raw_args]

    parser = argparse.ArgumentParser(prog="autovoice")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _serve_parser(subparsers)
    _swarm_parser(subparsers)
    return parser.parse_args(raw_args)


def setup_signal_handlers(app, socketio) -> None:
    """Install graceful shutdown handlers."""

    def shutdown_handler(signum, _frame):
        logger.info("Received signal %s, initiating graceful shutdown", signum)

        wait_method = getattr(app, "wait_for_requests", None)
        if wait_method:
            completed = wait_method(timeout=30.0)
            if not completed:
                logger.warning("Some requests did not complete before shutdown timeout")

        job_manager = getattr(app, "job_manager", None)
        if job_manager:
            try:
                job_manager.stop_cleanup_thread()
            except Exception as exc:  # pragma: no cover - defensive shutdown path
                logger.warning("Error stopping job manager: %s", exc)

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - optional dependency path
            logger.warning("Error clearing GPU memory: %s", exc)

        if socketio:
            try:
                socketio.stop()
            except Exception as exc:  # pragma: no cover - defensive shutdown path
                logger.warning("Error stopping SocketIO: %s", exc)

        logger.info("Shutdown complete")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)


def _load_config(args: argparse.Namespace) -> Dict[str, Any]:
    loader = ConfigLoader()
    defaults: Dict[str, Any] = {}

    yaml_config: Dict[str, Any] = {}
    if args.config:
        try:
            yaml_config = loader.load_from_file(Path(args.config))
        except ConfigLoadError as exc:
            raise SystemExit(str(exc)) from exc

    config = loader.load_with_merge(defaults, yaml_config=yaml_config)

    server_config = config.get("server", {})
    config["DEBUG"] = bool(
        args.debug or config.get("debug", server_config.get("debug", False))
    )
    config["TESTING"] = bool(config.get("TESTING", config.get("testing", False)))

    data_dir = args.data_dir or config.get("DATA_DIR") or os.environ.get("DATA_DIR")
    if data_dir:
        config["DATA_DIR"] = data_dir

    secret_key = config.get("SECRET_KEY") or os.environ.get("SECRET_KEY")
    if secret_key:
        config["SECRET_KEY"] = secret_key

    return config


def _resolve_ssl_context(args: argparse.Namespace) -> tuple[str, str] | None:
    """Resolve an optional Flask-SocketIO SSL context from CLI args or env."""
    ssl_cert = args.ssl_cert or os.environ.get("AUTOVOICE_SSL_CERT")
    ssl_key = args.ssl_key or os.environ.get("AUTOVOICE_SSL_KEY")

    if bool(ssl_cert) != bool(ssl_key):
        raise SystemExit("Both --ssl-cert and --ssl-key are required to enable HTTPS")

    if not ssl_cert or not ssl_key:
        return None

    cert_path = Path(ssl_cert).expanduser()
    key_path = Path(ssl_key).expanduser()
    missing = [str(path) for path in (cert_path, key_path) if not path.exists()]
    if missing:
        raise SystemExit(f"TLS file not found: {', '.join(missing)}")

    return str(cert_path), str(key_path)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    if args.command == "swarm":
        run_root = Path(args.data_dir) / "swarm_runs"
        if args.swarm_command == "validate":
            manifest = load_manifest(Path(args.manifest))
            specs = task_specs(manifest)
            topological_order(specs)
            print(
                json.dumps(
                    {
                        "manifest": str(Path(args.manifest)),
                        "task_count": len(specs),
                        "status": "valid",
                    },
                    indent=2,
                )
            )
            return 0
        if args.swarm_command == "status":
            return print_status(args.run_id, run_root=run_root)
        if args.swarm_command == "cancel":
            print(json.dumps(request_cancel(run_id=args.run_id, run_root=run_root, reason=args.reason), indent=2))
            return 0
        if args.swarm_command == "run":
            return execute_manifest(
                Path(args.manifest),
                run_id=args.run_id,
                dry_run=bool(args.dry_run),
                run_root=run_root,
                project_root=Path.cwd(),
                parent_run_id=os.environ.get("AUTOVOICE_SWARM_RUN_ID"),
            )
        if args.swarm_command in {"resume", "retry"}:
            prepared = prepare_resume_run(
                source_run_id=args.source_run_id,
                new_run_id=args.run_id,
                run_root=run_root,
                mode=args.swarm_command,
            )
            return execute_manifest(
                Path(prepared["manifest_path"]),
                run_id=args.run_id,
                dry_run=False,
                run_root=run_root,
                project_root=Path.cwd(),
                parent_run_id=str(prepared["parent_run_id"]),
            )
        raise SystemExit(f"Unsupported swarm command: {args.swarm_command}")

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.command != "serve":
        raise SystemExit(f"Unsupported command: {args.command}")

    config = _load_config(args)
    ssl_context = _resolve_ssl_context(args)
    app, socketio = create_app(config=config)
    setup_signal_handlers(app, socketio)

    logger.info(
        "Starting AutoVoice on %s:%s%s",
        args.host,
        args.port,
        " with HTTPS" if ssl_context else "",
    )
    socketio.run(
        app,
        host=args.host,
        port=args.port,
        debug=bool(config.get("DEBUG")),
        ssl_context=ssl_context,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI module entrypoint
    raise SystemExit(main())
