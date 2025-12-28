#!/usr/bin/env python3
"""Web server for real-time IMU visualization.

Launches the sensor fusion process as a subprocess and
broadcasts orientation data to connected WebSocket clients.
"""

import argparse
import json
import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from flask import Flask, render_template
from flask_socketio import SocketIO

from core import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = "imu_visualization_secret"
socketio = SocketIO(app, cors_allowed_origins="*")

fusion_process: Optional[subprocess.Popen] = None


def start_fusion_process(config_path: Optional[str], use_mock: bool) -> None:
    """Start the sensor fusion subprocess.

    Args:
        config_path: Path to configuration file.
        use_mock: If True, use mock IMU.
    """
    global fusion_process

    script_dir = Path(__file__).parent
    main_script = script_dir / "main.py"

    cmd = [sys.executable, str(main_script)]

    if config_path:
        cmd.extend(["-c", config_path])
    if use_mock:
        cmd.append("--mock")

    logger.info("Starting fusion process: %s", " ".join(cmd))

    fusion_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True,
        cwd=str(script_dir),
    )

    stderr_thread = threading.Thread(
        target=_read_stderr,
        args=(fusion_process,),
        daemon=True,
    )
    stderr_thread.start()

    stdout_thread = threading.Thread(
        target=_read_stdout,
        args=(fusion_process,),
        daemon=True,
    )
    stdout_thread.start()

    logger.info("Fusion process started (PID: %d)", fusion_process.pid)


def _read_stderr(process: subprocess.Popen) -> None:
    """Read and log stderr from fusion process.

    Args:
        process: Fusion subprocess.
    """
    try:
        for line in process.stderr:
            line = line.rstrip()
            if line:
                logger.info("[FUSION] %s", line)
    except Exception as e:
        logger.error("Error reading stderr: %s", e)


def _read_stdout(process: subprocess.Popen) -> None:
    """Read stdout from fusion process and emit to WebSocket clients.

    Args:
        process: Fusion subprocess.
    """
    try:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                socketio.emit("orientation_update", data)
            except json.JSONDecodeError as e:
                logger.debug("Invalid JSON: %s - %s", e, line[:100])

    except Exception as e:
        logger.error("Error reading stdout: %s", e)

    logger.info("Fusion process terminated")


def stop_fusion_process() -> None:
    """Stop the fusion subprocess."""
    global fusion_process

    if fusion_process is not None:
        logger.info("Stopping fusion process...")
        fusion_process.terminate()
        try:
            fusion_process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            fusion_process.kill()
        fusion_process = None


@app.route("/")
def index():
    """Serve the main visualization page."""
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    """Handle WebSocket client connection."""
    logger.info("WebSocket client connected")


@socketio.on("disconnect")
def handle_disconnect():
    """Handle WebSocket client disconnection."""
    logger.info("WebSocket client disconnected")


def main() -> int:
    """Application entry point.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Web server for IMU visualization"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock IMU for testing",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host address to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to listen on",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.warning("Could not load config: %s, using defaults", e)
        from core.config import Config
        config = Config()

    host = args.host or config.web.host
    port = args.port or config.web.port

    start_fusion_process(args.config, args.mock)

    try:
        logger.info("=" * 60)
        logger.info("Web server starting on http://%s:%d", host, port)
        logger.info("=" * 60)

        socketio.run(app, host=host, port=port, debug=False)

    except KeyboardInterrupt:
        logger.info("Server interrupted")

    finally:
        stop_fusion_process()

    return 0


if __name__ == "__main__":
    sys.exit(main())
