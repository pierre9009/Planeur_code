#!/usr/bin/env python3
import time
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

from ahrs.filters import EKF
from imu_api import ImuSoftUart

# ------------------------------------------------------------
# Configuration magnétique locale
# ------------------------------------------------------------

# https://www.magnetic-declination.com/
MAG_DECLINATION = 2.7  # degrés (EAST positive)
MAG_INCLINATION = 60.2  # degrés (DIP angle)
MAG_INTENSITY = 30.76  # uT
# ------------------------------------------------------------
# Quaternion helpers
# ------------------------------------------------------------
def q_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)

def q_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)

def quaternion_to_euler_direct(q):
    
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Roll (φ): rotation autour de X
    roll = np.arctan2(2.0 * (w*x + y*z), (w*w + z*z - x*x - y*y))
    
    # Pitch (θ): rotation autour de Y
    sin_pitch = 2.0 * (w*y - x*z)
    pitch = np.arcsin(sin_pitch)
    
    # Yaw (ψ): rotation autour de Z
    yaw = np.arctan2(2.0 * (w*z + x*y), (w*w + x*x - y*y - z*z))
    
    return roll, pitch, yaw


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
    q0 = np.array([1, 0, 0, 0], dtype=float) # w,x,y,z  on suppose que le capteur a plat

    # Bruits 
    sigma_g = 0.1**2
    sigma_a = 0.3**2
    sigma_m = 0.5**2

    # Référence du champ magnétique en NED
    I = np.deg2rad(MAG_INCLINATION)
    D = np.deg2rad(MAG_DECLINATION)
    F = MAG_INTENSITY
    
    mag_ref_ned = np.array([
        F * np.cos(I) * np.cos(D),   # North
        F * np.cos(I) * np.sin(D),   # East
        F * np.sin(I)                # Down
    ], dtype=float)
    
    print(f"  Référence magnétique NED: [{mag_ref_ned[0]:.1f}, {mag_ref_ned[1]:.1f}, {mag_ref_ned[2]:.1f}] uT", file=sys.stderr)

    # -------------------------
    # EKF initialisation
    # -------------------------
    ekf = EKF(
        gyr=np.zeros((1, 3)),
        acc=acc0.reshape((1,3)),
        mag=mag0.reshape((1,3)),  # En nT maintenant
        frequency=100.0,
        frame="ENU",
        magnetic_ref=mag_ref_ned,
        noises=[sigma_g, sigma_a, sigma_m]
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

            roll, pitch, yaw = quaternion_to_euler_direct(q)
            
            roll_deg = np.rad2deg(roll)
            pitch_deg = np.rad2deg(pitch)
            yaw_deg = np.rad2deg(yaw)
            
            print(json.dumps({
                "roll": float(roll_deg),
                "pitch": float(pitch_deg),
                "yaw": float(yaw_deg),           # Yaw géographique (nord vrai)
                "yaw_mag": float(yaw_deg - MAG_DECLINATION),  # Yaw magnétique
                "dt": float(dt * 1000.0),
            }), flush=True)

    finally:
        imu.close()

if __name__ == "__main__":
    run_imu_fusion()