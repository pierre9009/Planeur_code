#!/usr/bin/env python3
import time
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from ahrs.common.orientation import am2q

from ahrs.filters import EKF
from imu_api import ImuSoftUart

# ------------------------------------------------------------
# Configuration magnétique locale
# ------------------------------------------------------------

# https://www.magnetic-declination.com/
MAG_DECLINATION = 2.7  # degrés (EAST positive)
MAG_INCLINATION = 60.2  # degrés (DIP angle)
MAG_INTENSITY = 30.76  # uT


def quaternion_to_euler_direct(q):
    q_scipy = [q[1], q[2], q[3], q[0]]
    r = R.from_quat(q_scipy)
    # On demande l'ordre aéronautique (Yaw, Pitch, Roll)
    return r.as_euler('zyx', degrees=False)

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
    print("Collecte des échantillons d'initialisation...", file=sys.stderr)
    
    while len(samples) < 40 and (time.time() - t0) < 5.0:
        m = imu.read_measurement(timeout_s=0.5)
        if m is not None:
            samples.append(m)
            if len(samples) % 10 == 0:
                print(f"  {len(samples)}/40 échantillons", file=sys.stderr)

    if not samples:
        print("ERREUR: Pas de données IMU", file=sys.stderr)
        imu.close()
        return

    acc0 = np.mean([[s["ax"], s["ay"], s["az"]] for s in samples], axis=0)
    mag0 = np.mean([[s["mx"], s["my"], s["mz"]] for s in samples], axis=0)

    print(f"\nInitialisation:", file=sys.stderr)
    print(f"  ACC: [{acc0[0]:.3f}, {acc0[1]:.3f}, {acc0[2]:.3f}] m/s²", file=sys.stderr)
    print(f"  MAG: [{mag0[0]:.2f}, {mag0[1]:.2f}, {mag0[2]:.2f}] µT", file=sys.stderr)
    
    mag_measured = np.linalg.norm(mag0)
    print(f"  Intensité magnétique mesurée: {mag_measured:.1f} uT", file=sys.stderr)
    print(f"  Intensité magnétique attendue: {MAG_INTENSITY:.1f} uT", file=sys.stderr)
    print(f"  Écart: {abs(mag_measured - MAG_INTENSITY):.1f} uT", file=sys.stderr)

    # -------------------------
    # Initial orientation
    # -------------------------
    q0 = am2q(acc0, mag0, frame='ENU')
    print(f"✓ Orientation initiale calculée: {q0}", file=sys.stderr)

    # Bruits 
    sigma_g = 0.05
    sigma_a = 0.2
    sigma_m = 0.5

    # Référence du champ magnétique en NED
    I = np.deg2rad(MAG_INCLINATION)
    D = np.deg2rad(MAG_DECLINATION)
    F = MAG_INTENSITY
    
    # En ENU: X=East, Y=North, Z=Up
    mag_ref_enu = np.array([
        F * np.cos(I) * np.sin(D),   # East
        F * np.cos(I) * np.cos(D),   # North
        F * np.sin(-I)               # Up (I est positif vers le bas, donc Up = -I)
    ])
    
    print(f"  Référence magnétique NED: [{mag_ref_enu[0]:.1f}, {mag_ref_enu[1]:.1f}, {mag_ref_enu[2]:.1f}] uT", file=sys.stderr)

    # -------------------------
    # EKF initialisation
    # -------------------------
    ekf = EKF(
        gyr=np.zeros((1, 3)),
        acc=acc0.reshape((1,3)),
        mag=mag0.reshape((1,3)),
        frequency=100.0,
        frame="ENU",
        magnetic_ref=mag_ref_enu,
        noises=[sigma_g, sigma_a, sigma_m] # Bruits ajustés pour plus de stabilité
    )

    q = q0
    last_t = time.time()

    print("\n✓ EKF démarré - Envoi des données...\n", file=sys.stderr)

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
            mag = np.array([m["mx"], m["my"], m["mz"]], dtype=float)

            q = ekf.update(q, gyr, acc, mag, dt=dt)
            
            yaw,pitch, roll, = quaternion_to_euler_direct(q)

            sys.stderr.write(
                f"\rEst: R:{np.rad2deg(roll):7.2f}° | P:{np.rad2deg(pitch):7.2f}° | Y:{np.rad2deg(yaw):7.2f}° | dt:{dt*1000:5.1f}ms"
            )
            sys.stderr.flush()

            print(json.dumps({
                "qx": float(q[1]), # x
                "qy": float(q[2]), # y
                "qz": float(q[3]), # z
                "qw": float(q[0]), # w (ahrs utilise w en premier)
                "roll": float(np.rad2deg(roll)),
                "pitch": float(np.rad2deg(pitch)),
                "yaw": float(np.rad2deg(yaw)),
                "dt": float(dt * 1000.0)
            }), flush=True)

    finally:
        imu.close()

if __name__ == "__main__":
    run_imu_fusion()