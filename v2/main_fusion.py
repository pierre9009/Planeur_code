#!/usr/bin/env python3
"""Fusion IMU avec EKF - Sortie JSON sur stdout pour visualisation"""

import sys
import time
import json
import numpy as np
from scipy.spatial.transform import Rotation
import imu_reader
from ekf import EKF

# Config
TARGET_HZ = 100
DT = 1.0 / TARGET_HZ
DEBUG_LATENCY = "--debug" in sys.argv  # Ajouter timestamps pour diagnostic


def quat_to_euler(q):
    """Quaternion [w,x,y,z] -> [roll, pitch, yaw] en degrés"""
    # scipy utilise [x,y,z,w]
    r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    euler = r.as_euler('xyz', degrees=True)
    return euler[0], euler[1], euler[2]


def main():
    with imu_reader.ImuReader() as imu:
        # Première mesure pour initialiser l'EKF
        m = imu.read()
        while m is None:
            m = imu.read()

        accel = np.array([m["ax"], m["ay"], m["az"]])
        ekf = EKF(accel_data=accel)

        print("READY", file=sys.stderr)

        last_time = time.perf_counter()
        loop_count = 0
        last_hz_time = time.perf_counter()

        while True:
            t_start = time.perf_counter()

            # Lecture IMU (non-bloquante si possible)
            m = imu.read(timeout=0.05)  # Timeout court pour éviter blocage
            if m is None:
                continue

            t_after_read = time.perf_counter()

            # Calcul du dt réel
            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            # Données capteurs
            gyro = np.array([m["gx"], m["gy"], m["gz"]])
            accel = np.array([m["ax"], m["ay"], m["az"]])

            # EKF predict + update
            ekf.predict(gyro, dt)
            ekf.update(accel)

            t_after_ekf = time.perf_counter()

            # Quaternion [w, x, y, z]
            q = ekf.state[0:4]

            # Euler angles pour affichage
            roll, pitch, yaw = quat_to_euler(q)

            # Sortie JSON sur stdout (conversion NED -> Three.js Y-up)
            data = {
                "qw": float(q[0]),
                "qx": float(q[1]),
                "qy": float(-q[3]),  # -z_ned -> y_three
                "qz": float(-q[2]),  # -y_ned -> z_three
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
            }

            # Mode debug: ajouter timestamps pour diagnostic latence
            if DEBUG_LATENCY:
                data["t_send"] = time.time() * 1000  # ms epoch
                data["t_read_ms"] = (t_after_read - t_start) * 1000
                data["t_ekf_ms"] = (t_after_ekf - t_after_read) * 1000
                data["dt_ms"] = dt * 1000

            print(json.dumps(data), flush=True)

            # Stats Hz toutes les secondes
            loop_count += 1
            if now - last_hz_time >= 1.0:
                hz = loop_count / (now - last_hz_time)
                print(f"[{hz:.1f} Hz]", file=sys.stderr)
                loop_count = 0
                last_hz_time = now

            # PAS DE SLEEP - on veut la latence minimale
            # Le rate est limité par la lecture IMU (~100Hz)


if __name__ == "__main__":
    main()
