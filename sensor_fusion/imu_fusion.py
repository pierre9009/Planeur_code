#!/usr/bin/env python3
import time
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

from ahrs.filters import EKF
from imu_api import ImuSoftUart

# ------------------------------------------------------------
# Quaternion helpers
# ------------------------------------------------------------
def q_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)

def q_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)

# ------------------------------------------------------------
# Initial attitude from ACC + MAG (tilt compensated yaw)
# ------------------------------------------------------------
def init_q0_from_acc_mag(acc, mag):
    ax, ay, az = acc
    mx, my, mz = mag

    # Roll / Pitch from accelerometer
    roll  = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay*ay + az*az))

    # Tilt compensated magnetometer
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)

    mx2 = mx * cp + mz * sp
    my2 = mx * sr * sp + my * cr - mz * sr * cp
    yaw = np.arctan2(-my2, mx2)

    rot = R.from_euler('zyx', [yaw, pitch, roll], degrees=False)
    return q_xyzw_to_wxyz(rot.as_quat())

# ------------------------------------------------------------
# Main fusion loop
# ------------------------------------------------------------
def run_imu_fusion():
    imu = ImuSoftUart(rx_gpio=24, baudrate=57600)
    imu.open()

    # -------------------------
    # Collect init samples
    # -------------------------
    samples = []
    t0 = time.time()
    while len(samples) < 40 and (time.time() - t0) < 5.0:
        m = imu.read_measurement(timeout_s=0.5)
        if m is not None:
            samples.append(m)

    if not samples:
        print("ERREUR: Pas de donnees IMU", file=sys.stderr)
        imu.close()
        return

    acc0 = np.mean([[s["ax"], s["ay"], s["az"]] for s in samples], axis=0)
    mag0 = np.mean([[s["mx"], s["my"], s["mz"]] for s in samples], axis=0) * 1000.0

    print(f"ACC init: {acc0}", file=sys.stderr)
    print(f"MAG init (nT): {mag0}", file=sys.stderr)

    q0 = init_q0_from_acc_mag(acc0, mag0)

    # -------------------------
    # EKF initialisation (FORCÉ EN 6D)
    # -------------------------
    ekf = EKF(
        gyr=np.zeros((1, 3)),
        acc=acc0.reshape(1, 3),
        mag=mag0.reshape(1, 3),
        frequency=100.0,
        frame="NED",
        q0=q0,
        magnetic_ref=60.0,     # dip angle OK pour prototype
        noises=[0.01, 0.1, 0.5]
    )

    q = q0
    last_t = time.time()

    print("EKF démarré", file=sys.stderr)

    try:
        while True:
            m = imu.read_measurement(timeout_s=1.0)
            if m is None:
                continue

            now = time.time()
            dt = now - last_t
            last_t = now
            if dt <= 0.0:
                dt = 0.01

            acc = np.array([m["ax"], m["ay"], m["az"]], dtype=float)
            gyr = np.array([m["gx"], m["gy"], m["gz"]], dtype=float)
            mag = np.array([m["mx"], m["my"], m["mz"]], dtype=float) * 1000.0

            q = ekf.update(q, gyr, acc, mag, dt=dt)

            rot = R.from_quat(q_wxyz_to_xyzw(q))
            yaw, pitch, roll = rot.as_euler('zyx', degrees=True)

            print(json.dumps({
                "roll": float(roll),
                "pitch": float(pitch),
                "yaw": float(yaw),
                "dt": float(dt * 1000.0),
            }), flush=True)

    finally:
        imu.close()

if __name__ == "__main__":
    run_imu_fusion()
