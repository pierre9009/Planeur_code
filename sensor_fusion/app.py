import serial
import time
import math
import numpy as np
import json
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading

from ahrs.filters import Fourati
from ahrs.common.orientation import acc2q
from ahrs.common.quaternion import Quaternion


app = Flask(__name__)
app.config['SECRET_KEY'] = 'imu_visualization_secret'
socketio = SocketIO(app, cors_allowed_origins="*")


def parse_packet(line: str):
    try:
        if not line.startswith("D,"):
            return None

        parts = line.strip().split(",")
        if len(parts) != 12:
            return None

        return {
            "roll_comp":  float(parts[1]),
            "pitch_comp": float(parts[2]),
            "gx_deg":     float(parts[3]),
            "gy_deg":     float(parts[4]),
            "gz_deg":     float(parts[5]),
            "ax":         float(parts[6]),
            "ay":         float(parts[7]),
            "az":         float(parts[8]),
            "mx":         float(parts[9]),
            "my":         float(parts[10]),
            "mz":         float(parts[11]),
        }
    except ValueError:
        return None


def imu_thread():
    """Thread qui lit les données série et met à jour le filtre Fourati"""
    
    ser = serial.Serial(
        port="/dev/serial0",
        baudrate=230400,
        timeout=1.0,
    )

    time.sleep(0.2)
    print("Port série ouvert")

    sample_freq = 50.0
    gyro_scale = 1.0
    magnetic_dip_deg = 64.0

    fourati = Fourati(
        frequency=sample_freq,
        magnetic_dip=magnetic_dip_deg,
    )

    q = None
    last_t = time.time()

    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            now = time.time()
            dt = now - last_t
            if dt <= 0.0:
                dt = 1.0 / sample_freq
            last_t = now

            data = parse_packet(line)
            if data is None:
                continue

            # Conversion gyroscope
            gx_rad = data["gx_deg"] * gyro_scale * math.pi / 180.0
            gy_rad = data["gy_deg"] * gyro_scale * math.pi / 180.0
            gz_rad = data["gz_deg"] * gyro_scale * math.pi / 180.0
            gyr = np.array([gx_rad, gy_rad, gz_rad], dtype=float)

            # Accéléromètre
            acc = np.array([data["ax"], data["ay"], data["az"]], dtype=float)

            # Magnétomètre µT -> mT
            mag = np.array([data["mx"], data["my"], data["mz"]], dtype=float) / 1000.0

            # Initialisation du quaternion
            if q is None:
                q = acc2q(acc)
                q = q / np.linalg.norm(q)
                print("Quaternion initial:", q)
                continue

            # Mise à jour Fourati
            q = fourati.update(
                q=q,
                gyr=gyr,
                acc=acc,
                mag=mag,
                dt=dt,
            )

            # Conversion en angles d'Euler
            angles = Quaternion(q).to_angles()
            roll_f, pitch_f, yaw_f = angles

            roll_f_deg  = math.degrees(roll_f)
            pitch_f_deg = math.degrees(pitch_f)
            yaw_f_deg   = math.degrees(yaw_f)

            # Envoi des données via WebSocket
            orientation_data = {
                'roll': roll_f_deg,
                'pitch': pitch_f_deg,
                'yaw': yaw_f_deg,
                'roll_comp': data["roll_comp"],
                'pitch_comp': data["pitch_comp"],
                'dt': dt * 1000.0
            }
            
            socketio.emit('orientation_update', orientation_data)

            print(
                f"COMP R={data['roll_comp']:7.2f} P={data['pitch_comp']:7.2f} | "
                f"FOURATI R={roll_f_deg:7.2f} P={pitch_f_deg:7.2f} Y={yaw_f_deg:7.2f}"
            )

        except Exception as e:
            print("Erreur IMU thread:", e)


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    print('Client connecté')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client déconnecté')


if __name__ == '__main__':
    # Démarrage du thread IMU
    imu_thread_instance = threading.Thread(target=imu_thread, daemon=True)
    imu_thread_instance.start()
    
    print("Serveur démarré sur http://0.0.0.0:5000")
    print("Ouvrez cette adresse dans votre navigateur")
    
    # Démarrage du serveur Flask
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)