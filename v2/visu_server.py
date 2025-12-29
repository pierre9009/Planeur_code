#!/usr/bin/env python3
"""Serveur de visualisation IMU avec Flask-SocketIO"""

import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation
from flask import Flask, render_template_string
from flask_socketio import SocketIO

import imu_reader
from ekf import EKF

app = Flask(__name__)
app.config['SECRET_KEY'] = 'imu_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

HTML_PAGE = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Visualisation IMU 3D</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); overflow: hidden; color: white; }
        #container { width: 100vw; height: 100vh; display: flex; flex-direction: column; }
        #header { padding: 20px; background: rgba(0,0,0,0.3); backdrop-filter: blur(10px); }
        h1 { font-size: 28px; font-weight: 300; letter-spacing: 2px; }
        #canvas-container { flex: 1; position: relative; }
        #info-panel { position: absolute; top: 20px; left: 20px; background: rgba(0,0,0,0.7); backdrop-filter: blur(10px); padding: 20px; border-radius: 10px; min-width: 280px; }
        .data-row { display: flex; justify-content: space-between; margin: 10px 0; font-size: 16px; }
        .label { font-weight: 600; color: #a0aec0; }
        .value { font-family: 'Courier New', monospace; color: #4ade80; font-weight: bold; }
        .status { margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.2); font-size: 14px; }
        .connected { color: #4ade80; }
        .disconnected { color: #f87171; }
    </style>
</head>
<body>
    <div id="container">
        <div id="header"><h1>🛩️ VISUALISATION CENTRALE INERTIELLE</h1></div>
        <div id="canvas-container"></div>
        <div id="info-panel">
            <div class="data-row"><span class="label">Roll:</span><span class="value" id="roll-value">0.00°</span></div>
            <div class="data-row"><span class="label">Pitch:</span><span class="value" id="pitch-value">0.00°</span></div>
            <div class="data-row"><span class="label">Yaw:</span><span class="value" id="yaw-value">0.00°</span></div>
            <div class="data-row"><span class="label">Hz:</span><span class="value" id="hz-value">0</span></div>
            <div class="status">État: <span id="status" class="disconnected">Déconnecté</span></div>
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script>
        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.set(0, 3, 8);
        camera.lookAt(0, 0, 0);
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.shadowMap.enabled = true;
        container.appendChild(renderer.domElement);

        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(5, 10, 5);
        scene.add(dirLight);

        const planeGroup = new THREE.Group();
        
        // Fuselage
        const fuselage = new THREE.Mesh(
            new THREE.CylinderGeometry(0.3, 0.3, 3, 16),
            new THREE.MeshPhongMaterial({ color: 0x4a90e2, shininess: 100 })
        );
        fuselage.rotation.z = Math.PI / 2;
        planeGroup.add(fuselage);
        
        // Ailes
        const wings = new THREE.Mesh(
            new THREE.BoxGeometry(1, 0.1, 6),
            new THREE.MeshPhongMaterial({ color: 0x5aa3e8 })
        );
        planeGroup.add(wings);
        
        // Empennage vertical (tourné de 90°)
        const tailVert = new THREE.Mesh(
            new THREE.BoxGeometry(0.8, 1, 0.1),
            new THREE.MeshPhongMaterial({ color: 0x3a7bc8 })
        );
        tailVert.position.set(-1.3, 0.5, 0);
        planeGroup.add(tailVert);
        
        // Empennage horizontal (tourné de 90°)
        const tailHoriz = new THREE.Mesh(
            new THREE.BoxGeometry(0.6, 0.1, 2),
            new THREE.MeshPhongMaterial({ color: 0x3a7bc8 })
        );
        tailHoriz.position.set(-1.3, 0.8, 0);
        planeGroup.add(tailHoriz);
        
        // Nez
        const nose = new THREE.Mesh(
            new THREE.SphereGeometry(0.3, 16, 16),
            new THREE.MeshPhongMaterial({ color: 0xe74c3c })
        );
        nose.position.x = 1.5;
        nose.scale.set(1.2, 1, 1);
        planeGroup.add(nose);
        
        planeGroup.add(new THREE.AxesHelper(5));
        scene.add(planeGroup);
        
        scene.add(new THREE.GridHelper(20, 20, 0x4a5568, 0x2d3748));
        const axes = new THREE.AxesHelper(3);
        axes.position.y = -2;
        scene.add(axes);

        const socket = io();
        let lastUpdate = performance.now();
        let frameCount = 0;
        
        socket.on('connect', () => {
            document.getElementById('status').textContent = 'Connecté';
            document.getElementById('status').className = 'connected';
        });
        
        socket.on('disconnect', () => {
            document.getElementById('status').textContent = 'Déconnecté';
            document.getElementById('status').className = 'disconnected';
        });
        
        socket.on('orientation', (data) => {
            // Correction axes: pitch était affiché comme yaw
            // EKF: X=avant, Y=droite, Z=bas (NED)
            // Three.js: X=droite, Y=haut, Z=avant
            planeGroup.quaternion.set(data.qz, -data.qx, data.qy, data.qw);
            document.getElementById('roll-value').textContent = data.roll.toFixed(2) + '°';
            document.getElementById('pitch-value').textContent = data.pitch.toFixed(2) + '°';
            document.getElementById('yaw-value').textContent = data.yaw.toFixed(2) + '°';
            
            frameCount++;
            const now = performance.now();
            if (now - lastUpdate > 1000) {
                document.getElementById('hz-value').textContent = frameCount;
                frameCount = 0;
                lastUpdate = now;
            }
        });

        function animate() {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', () => {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });
    </script>
</body>
</html>"""


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
            m = imu.read(timeout=0.1)
            if m is None:
                continue

            now = time.time()
            dt = now - last_time
            last_time = now

            gyro = np.array([m["gx"], m["gy"], m["gz"]])
            accel = np.array([m["ax"], m["ay"], m["az"]])

            ekf.predict(gyro, dt)
            ekf.update(accel)

            q = ekf.state[0:4]  # [w, x, y, z]
            
            # Calcul angles d'Euler pour affichage
            rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy: [x,y,z,w]
            euler = rot.as_euler('xyz', degrees=True)
            
            # Envoi immédiat via SocketIO
            socketio.emit('orientation', {
                'qw': float(q[0]),
                'qx': float(q[1]),
                'qy': float(q[2]),
                'qz': float(q[3]),
                'roll': float(euler[0]),
                'pitch': float(euler[1]),
                'yaw': float(euler[2])
            })


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@socketio.on('connect')
def handle_connect():
    print('Client connecté')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client déconnecté')


if __name__ == '__main__':
    threading.Thread(target=imu_thread, daemon=True).start()
    print("\n" + "=" * 50)
    print("http://0.0.0.0:5000")
    print("=" * 50 + "\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)