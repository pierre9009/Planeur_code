import time
import json
import sys
import numpy as np
from imu_api import ImuSoftUart

from ahrs.filters import EKF

from ahrs.common.orientation import am2q


def run_imu_fusion():
    imu = ImuSoftUart(rx_gpio=24, baudrate=57600)
    imu.open()

    m = imu.read_measurement(timeout_s=1.0)

    acc0 = np.array([m["ax"], m["ay"], m["az"]], dtype=float)  # m/s^2
    mag0 = np.array([m["mx"], m["my"], m["mz"]], dtype=float)  # uT
    print(acc0.shape)
    print(mag0.shape)
    print(np.zeros((3,1)).shape)

    q0=am2q(acc0, mag0)
    q0=q0.flatten()

    # Initialisation du filtre EKF
    ekf = EKF(gyr=np.zeros((1,3)), acc=acc0.reshape((1,3)), mag=mag0.reshape((1,3)), frequency=100,q0=q0, magnetic_ref=60.0)

    last_time=None

    q=q0
    try:
        while True:
            m = imu.read_measurement(timeout_s=1.0)
            if m is None: continue

            current_time = time.time()
            dt = current_time - last_time if last_time else 0.01
            last_time = current_time

            acc = np.array([m["ax"], m["ay"], m["az"]], dtype=float)
            gyr = np.array([m["gx"], m["gy"], m["gz"]], dtype=float)
            mag = np.array([m["mx"], m["my"], m["mz"]], dtype=float)

            q = ekf.update(q, acc, gyr, mag, dt)

            # ENVOI COMPLET : Fusion + RAW
            print(json.dumps({
                "qw": float(q[0]), "qx": float(q[1]), "qy": float(q[2]), "qz": float(q[3]),
                "ax": m["ax"], "ay": m["ay"], "az": m["az"],
                "gx": m["gx"], "gy": m["gy"], "gz": m["gz"],
                "mx": m["mx"], "my": m["my"], "mz": m["mz"],
                "dt": dt * 1000 # en ms pour l'affichage
            }), flush=True)

    finally:
        imu.close()


if __name__ == "__main__":
    run_imu_fusion()
