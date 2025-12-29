#!/usr/bin/env python3
"""Enregistre les données IMU dans un fichier .log"""

import imu_reader

with open("imu_data.log", "w") as f, imu_reader.ImuReader() as imu:
    print("Enregistrement dans imu_data.log (Ctrl+C pour arrêter)...")
    while True:
        m = imu.read()
        if m:
            f.write(f"{m['ax']} {m['ay']} {m['az']} {m['gx']} {m['gy']} {m['gz']} {m['mx']} {m['my']} {m['mz']} {m['seq']} {m['tempC']}\n")
            f.flush()
            print(f"{m['ax']} {m['ay']} {m['az']} {m['gx']} {m['gy']} {m['gz']}\n")