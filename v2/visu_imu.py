#!/usr/bin/env python3
"""Fusion IMU avec visualisation 3D via WebSocket"""

import time
import json
import threading
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler
import asyncio
import websockets

import imu_reader
from ekf import EKF

# Port WebSocket et HTTP
WS_PORT = 8765
HTTP_PORT = 8080

# Page HTML de visualisation
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <title>IMU Visualisation</title>
    <style>
        body { margin: 0; background: #1a1a2e; }
        #info { position: absolute; top: 10px; left: 10px; color: #fff; font-family: monospace; }
    </style>
</head>
<body>
    <div id="info">Connexion...</div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Scene
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setClearColor(0x1a1a2e);
        document.body.appendChild(renderer.domElement);

        // Lumières
        scene.add(new THREE.AmbientLight(0x404040));
        const light = new THREE.DirectionalLight(0xffffff, 1);
        light.position.set(5, 5, 5);
        scene.add(light);

        // Objet 3D (boîte représentant l'IMU)
        const geometry = new THREE.BoxGeometry(2, 0.5, 3);
        const materials = [
            new THREE.MeshPhongMaterial({ color: 0xff0000 }), // +X rouge
            new THREE.MeshPhongMaterial({ color: 0x800000 }), // -X
            new THREE.MeshPhongMaterial({ color: 0x00ff00 }), // +Y vert
            new THREE.MeshPhongMaterial({ color: 0x008000 }), // -Y
            new THREE.MeshPhongMaterial({ color: 0x0000ff }), // +Z bleu
            new THREE.MeshPhongMaterial({ color: 0x000080 })  // -Z
        ];
        const cube = new THREE.Mesh(geometry, materials);
        scene.add(cube);

        // Axes
        scene.add(new THREE.AxesHelper(3));

        // Grille
        const grid = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
        scene.add(grid);

        camera.position.set(4, 3, 4);
        camera.lookAt(0, 0, 0);

        // WebSocket
        const info = document.getElementById('info');
        let ws;

        function connect() {
            ws = new WebSocket('ws://' + window.location.hostname + ':8765');
            
            ws.onopen = () => {
                info.textContent = 'Connecté';
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                // Quaternion [w, x, y, z]
                cube.quaternion.set(data.x, data.y, data.z, data.w);
                info.textContent = `q=[${data.w.toFixed(3)}, ${data.x.toFixed(3)}, ${data.y.toFixed(3)}, ${data.z.toFixed(3)}]`;
            };
            
            ws.onclose = () => {
                info.textContent = 'Déconnecté - reconnexion...';
                setTimeout(connect, 1000);
            };
        }
        connect();

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
</html>"""


class QuaternionBroadcaster:
    def __init__(self):
        self.clients = set()
        self.latest_quat = {"w": 1, "x": 0, "y": 0, "z": 0}
        self.lock = threading.Lock()

    async def handler(self, websocket, path):
        self.clients.add(websocket)
        try:
            async for _ in websocket:
                pass
        finally:
            self.clients.discard(websocket)

    async def broadcast(self):
        while True:
            if self.clients:
                with self.lock:
                    msg = json.dumps(self.latest_quat)
                await asyncio.gather(
                    *[client.send(msg) for client in self.clients],
                    return_exceptions=True
                )
            await asyncio.sleep(0.02)  # 50 Hz

    def update(self, w, x, y, z):
        with self.lock:
            self.latest_quat = {"w": float(w), "x": float(x), "y": float(y), "z": float(z)}


broadcaster = QuaternionBroadcaster()


def imu_thread():
    """Thread de lecture IMU et fusion EKF"""
    with imu_reader.ImuReader() as imu:
        m = imu.read()
        while m is None:
            m = imu.read()

        accel = np.array([m["ax"], m["ay"], m["az"]])
        ekf = EKF(accel_data=accel)

        last_time = time.time()

        while True:
            m = imu.read()
            if m is None:
                continue

            now = time.time()
            dt = now - last_time
            last_time = now

            gyro = np.array([m["gx"], m["gy"], m["gz"]])
            accel = np.array([m["ax"], m["ay"], m["az"]])

            ekf.predict(gyro, dt)
            ekf.update(accel)

            q = ekf.state[0:4]
            broadcaster.update(q[0], q[1], q[2], q[3])


class HTMLHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode())


def http_thread():
    """Thread du serveur HTTP"""
    server = HTTPServer(('0.0.0.0', HTTP_PORT), HTMLHandler)
    print(f"HTTP: http://localhost:{HTTP_PORT}")
    server.serve_forever()


async def main():
    # Démarrer le serveur WebSocket
    server = await websockets.serve(broadcaster.handler, "0.0.0.0", WS_PORT)
    print(f"WebSocket: ws://localhost:{WS_PORT}")

    # Tâche de broadcast
    asyncio.create_task(broadcaster.broadcast())

    await asyncio.Future()  # Run forever


if __name__ == "__main__":
    # Lancer thread IMU
    t_imu = threading.Thread(target=imu_thread, daemon=True)
    t_imu.start()

    # Lancer thread HTTP
    t_http = threading.Thread(target=http_thread, daemon=True)
    t_http.start()

    print("Visualisation démarrée")
    
    # Boucle principale asyncio (WebSocket)
    asyncio.run(main())