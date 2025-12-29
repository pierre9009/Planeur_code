#!/usr/bin/env python3
"""Serveur WebSocket pour visualisation 3D du planeur"""

import asyncio
import json
import time
import numpy as np
import websockets
import imu_reader
from ekf import EKF

DT = 0.01  # 100 Hz

async def fusion_loop(websocket):
    """Boucle de fusion IMU et envoi des quaternions via WebSocket"""

    with imu_reader.ImuReader() as imu:
        # Première mesure pour initialiser l'EKF
        m = imu.read()
        while m is None:
            m = imu.read()

        accel = np.array([m["ax"], m["ay"], m["az"]])
        ekf = EKF(accel_data=accel)

        print("Fusion IMU démarrée")
        last_time = time.time()

        while True:
            m = imu.read()
            if m is None:
                await asyncio.sleep(0.001)
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

            # Quaternion [w, x, y, z] depuis l'EKF
            q = ekf.state[0:4]

            # Envoyer au format JSON pour Three.js
            # Three.js: Quaternion(x, y, z, w) et convention Y-up
            # Notre EKF: [w, x, y, z] en NED (X=avant, Y=droite, Z=bas)
            # Conversion NED -> Three.js (Y-up, Z-arrière):
            # x_three = x_ned, y_three = -z_ned, z_three = -y_ned
            data = {
                "qw": float(q[0]),
                "qx": float(q[1]),
                "qy": float(-q[3]),  # -z_ned -> y_three
                "qz": float(-q[2])   # -y_ned -> z_three
            }

            try:
                await websocket.send(json.dumps(data))
            except websockets.ConnectionClosed:
                print("Client déconnecté")
                break

            await asyncio.sleep(DT)


async def handler(websocket, path=None):
    """Gestionnaire de connexion WebSocket"""
    print(f"Client connecté")
    try:
        await fusion_loop(websocket)
    except Exception as e:
        print(f"Erreur: {e}")
    finally:
        print("Connexion fermée")


async def main():
    print("Démarrage du serveur WebSocket sur ws://0.0.0.0:8765")
    async with websockets.serve(handler, "0.0.0.0", 8765):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
