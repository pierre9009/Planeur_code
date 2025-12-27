import time
import json
import sys
import numpy as np
from imu_api import ImuSoftUart

from ahrs.filters import EKF

from ahrs.common.orientation import acc2q


def run_imu_fusion():
    imu = ImuSoftUart(rx_gpio=24, baudrate=57600)
    imu.open()

    m = imu.read_measurement(timeout_s=1.0)

    acc0 = np.array([m["ax"], m["ay"], m["az"]], dtype=float)  # m/s^2
    mag0 = np.array([m["mx"], m["my"], m["mz"]], dtype=float)  # uT
    print(acc0.shape)
    print(mag0.shape)
    print(np.zeros((3,1)).shape)

    # Initialisation du filtre EKF
    ekf = EKF(gyr=np.zeros((1,3)), acc=acc0.reshape((1,3)), mag=mag0.reshape((1,3)), frequency=100, magnetic_ref=60.0)


    try:
        while True:
            m = imu.read_measurement(timeout_s=1.0)
            if m is None:
                continue

            # Calcul du dt avec timestamp réel
            current_time = time.time()
            if last_time is not None:
                dt = current_time - last_time
                # garde-fou si jitter/timeouts
                if dt <= 0.0 or dt > 0.2:
                    dt = 0.01
            else:
                dt = 0.01
            last_time = current_time

            # Données capteurs
            acc = np.array([m["ax"], m["ay"], m["az"]], dtype=float)  # m/s^2
            gyr = np.array([m["gx"], m["gy"], m["gz"]], dtype=float)  # rad/s
            mag = np.array([m["mx"], m["my"], m["mz"]], dtype=float)  # uT

            # Mise à jour EKF

            q= ekf.update(acc, gyr, mag, dt)
            

            # Debug console
            sys.stderr.write(
                f"\rEKF -> qw={q[0]:.4f}, qx={q[1]:.4f}, qy={q[2]:.4f}, qz={q[3]:.4f}   dt={dt:.4f}   "
            )
            sys.stderr.flush()

            # Envoi JSON vers le serveur Web
            print(json.dumps({
                "qw": float(q[0]), "qx": float(q[1]),
                "qy": float(q[2]), "qz": float(q[3]),
            }), flush=True)

    finally:
        imu.close()


if __name__ == "__main__":
    run_imu_fusion()
