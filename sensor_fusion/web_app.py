#!/usr/bin/env python3
import subprocess
import threading
import json
import time
from pathlib import Path

from flask import Flask, jsonify, Response

app = Flask(__name__)

# etat global mis a jour par le thread de lecture
latest_attitude = {
    "t": None,
    "roll_deg": 0.0,
    "pitch_deg": 0.0,
    "yaw_deg": 0.0,
    "roll_comp": 0.0,
    "pitch_comp": 0.0,
}

proc = None


def reader_thread():
    global proc, latest_attitude

    script_path = Path(__file__).parent / "sensor_fusion.py"

    proc = subprocess.Popen(
        ["python3", str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    while True:
        line = proc.stdout.readline()
        if not line:
            # processus termine ou pipe ferme
            time.sleep(0.1)
            if proc.poll() is not None:
                break
            continue

        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        # mise a jour du dernier etat
        for k in latest_attitude.keys():
            if k in data:
                latest_attitude[k] = data[k]


@app.route("/")
def index():
    html = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Planeur 3D - Fourati</title>
    <style>
        body {
            margin: 0;
            background: #111;
            color: #eee;
            font-family: sans-serif;
        }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            padding: 8px 12px;
            background: rgba(0, 0, 0, 0.6);
            border-radius: 4px;
            font-size: 13px;
            z-index: 10;
        }
        canvas {
            display: block;
        }
    </style>
</head>
<body>
<div id="info">
    <div>Planeur 3D - Fourati</div>
    <div id="angles">R=0 P=0 Y=0</div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r152/three.min.js"></script>
<script>
let scene, camera, renderer;
let glider;

function init() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const width = window.innerWidth;
    const height = window.innerHeight;

    camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.set(0, 2, 6);

    renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setSize(width, height);
    document.body.appendChild(renderer.domElement);

    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambient);

    const dir = new THREE.DirectionalLight(0xffffff, 0.6);
    dir.position.set(5, 10, 7);
    scene.add(dir);

    // planeur simple: un fuselage + voilure
    const fuselageGeom = new THREE.BoxGeometry(2.5, 0.2, 0.2);
    const wingGeom = new THREE.BoxGeometry(3.5, 0.05, 0.8);
    const tailGeom = new THREE.BoxGeometry(0.8, 0.05, 0.5);
    const material = new THREE.MeshStandardMaterial({color: 0x3399ff});

    glider = new THREE.Group();

    const fuselage = new THREE.Mesh(fuselageGeom, material);
    glider.add(fuselage);

    const wings = new THREE.Mesh(wingGeom, material);
    wings.position.set(0.0, 0.0, 0.0);
    glider.add(wings);

    const tail = new THREE.Mesh(tailGeom, material);
    tail.position.set(-1.2, 0.0, 0.0);
    glider.add(tail);

    scene.add(glider);

    window.addEventListener("resize", onWindowResize, false);

    animate();
    setInterval(fetchAttitude, 50);  // 20 Hz
}

function onWindowResize() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

function fetchAttitude() {
    fetch("/attitude")
        .then(response => response.json())
        .then(data => {
            const roll = data.roll_deg || 0.0;
            const pitch = data.pitch_deg || 0.0;
            const yaw = data.yaw_deg || 0.0;

            // conversion en radians
            const rr = roll * Math.PI / 180.0;
            const pr = pitch * Math.PI / 180.0;
            const yr = yaw * Math.PI / 180.0;

            // hypothese: roll X, pitch Y, yaw Z
            glider.rotation.set(rr, pr, yr);

            const info = document.getElementById("angles");
            info.textContent = 
                "R=" + roll.toFixed(1) + "  P=" + pitch.toFixed(1) + "  Y=" + yaw.toFixed(1);
        })
        .catch(err => {
            // ignore temporairement
        });
}

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

init();
</script>
</body>
</html>
    """
    return Response(html, mimetype="text/html")


@app.route("/attitude")
def attitude():
    return jsonify(latest_attitude)


def start_reader():
    t = threading.Thread(target=reader_thread, daemon=True)
    t.start()


if __name__ == "__main__":
    start_reader()
    # host 0.0.0.0 pour y acceder depuis un autre PC
    app.run(host="0.0.0.0", port=5000, debug=False)
