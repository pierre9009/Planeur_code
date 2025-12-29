#!/usr/bin/env python3
"""Fusion IMU avec EKF - Sortie JSON sur stdout pour visualisation"""

import sys
import time
import json
import numpy as np
import imu_reader
from ekf import EKF

DT = 0.01  # 100 Hz

with imu_reader.ImuReader() as imu:
    # Première mesure pour initialiser l'EKF
    m = imu.read()
    while m is None:
        m = imu.read()

    accel = np.array([m["ax"], m["ay"], m["az"]])
    ekf = EKF(accel_data=accel)

    print("READY", file=sys.stderr)

    last_time = time.time()

    while True:
        m = imu.read()
        if m is None:
            continue

        # Calcul du dt réel
        now = time.time()
        dt = now - last_time
        last_time = now

        # Données capteurs
        gyro = np.array([m["gx"], m["gy"], m["gz"]])
        accel = np.array([m["ax"], m["ay"], m["az"]])

        # EKF predict + update
        ekf.predict(gyro, dt)
        ekf.update(accel)

        # Quaternion [w, x, y, z]
        q = ekf.state[0:4]

        # Sortie JSON sur stdout (conversion NED -> Three.js Y-up)
        data = {
            "qw": float(q[0]),
            "qx": float(q[1]),
            "qy": float(-q[3]),  # -z_ned -> y_three
            "qz": float(-q[2])   # -y_ned -> z_three
        }
        print(json.dumps(data), flush=True)

        # Attente pour maintenir ~100Hz
        elapsed = time.time() - now
        if elapsed < DT:
            time.sleep(DT - elapsed)
