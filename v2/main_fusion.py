#!/usr/bin/env python3
"""Fusion IMU avec EKF - Estimation de quaternions"""

import time
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
    
    print("Fusion IMU démarrée (Ctrl+C pour arrêter)")
    print("Format: q=[w, x, y, z]")
    
    last_seq = m["seq"]
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
        
        # Quaternion
        q = ekf.state[0:4]
        
        print(f"q=[{q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f}, {q[3]:+.4f}]")