#!/usr/bin/env python3
"""
Serveur de visualisation IMU - WebSocket + subprocess main_fusion.py
Usage: python visu_server.py [--host HOST] [--port PORT] [--debug]
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import Flask, render_template_string
from flask_socketio import SocketIO

app = Flask(__name__)
app.config["SECRET_KEY"] = "imu_visu"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

fusion_process = None
DEBUG_MODE = False

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>IMU Visualisation</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: monospace;
            background: #1a1a2e;
            color: white;
            overflow: hidden;
        }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 8px;
            z-index: 100;
        }
        #info div { margin: 5px 0; }
        .value { color: #4ade80; }
        #status { color: #f87171; }
        #status.ok { color: #4ade80; }
    </style>
</head>
<body>
    <div id="info">
        <div>Roll: <span class="value" id="roll">0.00</span>°</div>
        <div>Pitch: <span class="value" id="pitch">0.00</span>°</div>
        <div>Yaw: <span class="value" id="yaw">0.00</span>°</div>
        <div>Hz: <span class="value" id="hz">0</span></div>
        <div id="latency-info" style="display:none; border-top: 1px solid #444; margin-top: 10px; padding-top: 10px;">
            <div>Latency: <span class="value" id="latency">-</span> ms</div>
            <div>IMU read: <span class="value" id="t_read">-</span> ms</div>
            <div>EKF: <span class="value" id="t_ekf">-</span> ms</div>
        </div>
        <div>Status: <span id="status">Disconnected</span></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script>
        // Three.js setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);

        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 3, 8);
        camera.lookAt(0, 0, 0);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        // Lights
        scene.add(new THREE.AmbientLight(0xffffff, 0.5));
        const light = new THREE.DirectionalLight(0xffffff, 0.8);
        light.position.set(5, 10, 5);
        scene.add(light);

        // Glider
        const glider = new THREE.Group();

        // Fuselage
        const fuseMat = new THREE.MeshPhongMaterial({ color: 0x4a90e2 });
        const fuse = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 3, 12), fuseMat);
        fuse.rotation.z = Math.PI / 2;
        glider.add(fuse);

        // Nose
        const noseMat = new THREE.MeshPhongMaterial({ color: 0xe74c3c });
        const nose = new THREE.Mesh(new THREE.SphereGeometry(0.2, 12, 12), noseMat);
        nose.position.x = 1.5;
        nose.scale.set(1.5, 1, 1);
        glider.add(nose);

        // Wings
        const wingMat = new THREE.MeshPhongMaterial({ color: 0x5aa3e8 });
        const wing = new THREE.Mesh(new THREE.BoxGeometry(1, 0.05, 6), wingMat);
        glider.add(wing);

        // Horizontal tail
        const tailH = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.05, 1.5), wingMat);
        tailH.position.x = -1.3;
        glider.add(tailH);

        // Vertical tail
        const tailV = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.8, 0.05), fuseMat);
        tailV.position.set(-1.3, 0.4, 0);
        glider.add(tailV);

        scene.add(glider);

        // Grid
        const grid = new THREE.GridHelper(20, 20, 0x4a5568, 0x2d3748);
        grid.position.y = -2;
        scene.add(grid);

        // Axes
        scene.add(new THREE.AxesHelper(2));

        // WebSocket
        const socket = io();
        let lastUpdate = Date.now();
        let updateCount = 0;

        socket.on('connect', () => {
            document.getElementById('status').textContent = 'Connected';
            document.getElementById('status').className = 'ok';
        });

        socket.on('disconnect', () => {
            document.getElementById('status').textContent = 'Disconnected';
            document.getElementById('status').className = '';
        });

        socket.on('orientation', (data) => {
            // Apply quaternion directly
            glider.quaternion.set(data.qx, data.qy, data.qz, data.qw);

            // Update display
            if (data.roll !== undefined) {
                document.getElementById('roll').textContent = data.roll.toFixed(1);
                document.getElementById('pitch').textContent = data.pitch.toFixed(1);
                document.getElementById('yaw').textContent = data.yaw.toFixed(1);
            }

            // Latency info (debug mode)
            if (data.t_send !== undefined) {
                document.getElementById('latency-info').style.display = 'block';
                const latency = Date.now() - data.t_send;
                document.getElementById('latency').textContent = latency.toFixed(0);
                document.getElementById('t_read').textContent = data.t_read_ms.toFixed(2);
                document.getElementById('t_ekf').textContent = data.t_ekf_ms.toFixed(2);

                // Color code latency
                const el = document.getElementById('latency');
                if (latency < 20) el.style.color = '#4ade80';
                else if (latency < 50) el.style.color = '#fbbf24';
                else el.style.color = '#f87171';
            }

            // Hz counter
            updateCount++;
            const now = Date.now();
            if (now - lastUpdate >= 1000) {
                document.getElementById('hz').textContent = updateCount;
                updateCount = 0;
                lastUpdate = now;
            }
        });

        // Render loop
        function animate() {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }
        animate();

        // Resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


def read_fusion_stdout(process):
    """Lit stdout de main_fusion.py et envoie via WebSocket"""
    try:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                socketio.emit("orientation", data)
            except json.JSONDecodeError:
                pass
    except Exception as e:
        print(f"Stdout reader error: {e}")


def read_fusion_stderr(process):
    """Lit stderr pour les logs"""
    try:
        for line in process.stderr:
            line = line.rstrip()
            if line:
                print(f"[FUSION] {line}")
    except Exception:
        pass


def start_fusion():
    """Démarre main_fusion.py en subprocess"""
    global fusion_process, DEBUG_MODE

    script_dir = Path(__file__).parent
    cmd = [sys.executable, str(script_dir / "main_fusion.py")]

    if DEBUG_MODE:
        cmd.append("--debug")

    print(f"Starting: {' '.join(cmd)}")

    fusion_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(script_dir)
    )

    threading.Thread(target=read_fusion_stdout, args=(fusion_process,), daemon=True).start()
    threading.Thread(target=read_fusion_stderr, args=(fusion_process,), daemon=True).start()


def stop_fusion():
    """Arrête le subprocess"""
    global fusion_process
    if fusion_process:
        fusion_process.terminate()
        try:
            fusion_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            fusion_process.kill()


def main():
    global DEBUG_MODE

    parser = argparse.ArgumentParser(description="IMU Visualization Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable latency debugging")
    args = parser.parse_args()

    DEBUG_MODE = args.debug
    if DEBUG_MODE:
        print("DEBUG MODE: Latency timestamps enabled")

    start_fusion()

    try:
        print(f"Server: http://{args.host}:{args.port}")
        socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_fusion()


if __name__ == "__main__":
    main()
