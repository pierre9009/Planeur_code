#!/usr/bin/env python3
import time
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

from ahrs.filters import EKF
from imu_api import ImuSoftUart

# ------------------------------------------------------------
# Configuration magnétique locale (Faucon, France)
# ------------------------------------------------------------
MAG_DECLINATION = 2.7  # degrés (EAST positive)
MAG_INCLINATION = 60.18  # degrés (DIP angle)
MAG_INTENSITY = 47.1453  # µT (47145.3 nT)

# ------------------------------------------------------------
# Quaternion helpers
# ------------------------------------------------------------
def q_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)

def q_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)

def quaternion_to_euler_direct(q):
    """
    Extraction directe des angles d'Euler depuis un quaternion [w,x,y,z].
    Convention aéronautique ZYX (yaw-pitch-roll).
    
    Cette méthode évite les ambiguïtés de scipy.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Roll (φ): rotation autour de X
    roll = np.arctan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y))
    
    # Pitch (θ): rotation autour de Y
    sin_pitch = 2.0 * (w*y - z*x)
    # Clamp pour éviter les erreurs numériques
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    
    # Yaw (ψ): rotation autour de Z
    yaw = np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
    
    return roll, pitch, yaw

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

    # Séquence ZYX: rotation autour de Z (yaw), puis Y (pitch), puis X (roll)
    rot = R.from_euler('ZYX', [yaw, pitch, roll], degrees=False)
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
    print(f"  Intensité magnétique mesurée: {mag_measured:.2f} µT", file=sys.stderr)
    print(f"  Intensité magnétique attendue: {MAG_INTENSITY:.2f} µT", file=sys.stderr)
    print(f"  Écart: {abs(mag_measured - MAG_INTENSITY):.2f} µT", file=sys.stderr)

    q0 = init_q0_from_acc_mag(acc0, mag0)
    
    # Afficher l'orientation initiale
    rot0 = R.from_quat(q_wxyz_to_xyzw(q0))
    r0, p0, y0 = rot0.as_euler('ZYX', degrees=True)
    print(f"  Orientation initiale: Roll={r0:.1f}°, Pitch={p0:.1f}°, Yaw={y0:.1f}°", file=sys.stderr)

    # Bruits (écarts-types) cohérents avec ICM-20948 (ordre de grandeur)
    sigma_g = 8.3e-4   # rad/s  (ex: BW ~ 10 Hz)
    sigma_a = 7.1e-3   # m/s^2  (ex: BW ~ 10 Hz)
    sigma_m = 0.8*0.8  # default librarie

    # Référence du champ magnétique en NED (recommandé si l'API l'accepte)
    I = np.deg2rad(MAG_INCLINATION)
    D = np.deg2rad(MAG_DECLINATION)
    F = MAG_INTENSITY
    mag_ref_ned = np.array([
        F*np.cos(I)*np.cos(D),   # North
        F*np.cos(I)*np.sin(D),   # East
        F*np.sin(I)              # Down
    ], dtype=float)

    # -------------------------
    # EKF initialisation
    # -------------------------
    ekf = EKF(
        gyr=np.zeros((1, 3)),
        acc=acc0.reshape(1, 3),
        mag=mag0.reshape(1, 3),
        frequency=100.0,
        frame="NED",
        magnetic_ref=mag_ref_ned,  # Angle d'inclinaison magnétique (DIP)
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

            roll_deg = np.rad2deg(roll)
            pitch_deg = np.rad2deg(pitch)
            yaw_mag_deg = np.rad2deg(yaw)
            yaw_true_deg = yaw_mag_deg + MAG_DECLINATION

            print(json.dumps({
                "roll": float(roll_deg),
                "pitch": float(pitch_deg),
                "yaw": float(yaw_true_deg),      # Yaw géographique (nord vrai)
                "yaw_mag": float(yaw_mag_deg),   # Yaw magnétique
                "dt": float(dt * 1000.0),
            }), flush=True)

    finally:
        imu.close()

if __name__ == "__main__":
    run_imu_fusion()