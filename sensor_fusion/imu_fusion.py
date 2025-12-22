#!/usr/bin/env python3
import time
import math
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

from ahrs.filters import EKF
from ahrs.common.orientation import acc2q

from imu_api import ImuSoftUart


def run_imu_fusion():
    rx_gpio = 24
    baud = 57600
    imu = ImuSoftUart(rx_gpio=rx_gpio, baudrate=baud)
    imu.open()

    print("="*70, file=sys.stderr)
    print("INITIALISATION - Placez le capteur A PLAT", file=sys.stderr)
    print("="*70, file=sys.stderr)

    diag = []
    t_start = time.time()
    while len(diag) < 30 and (time.time() - t_start) < 5.0:
        m = imu.read_measurement(timeout_s=0.5)
        if m is None:
            continue
        diag.append(m)

    if not diag:
        print("ERREUR: Pas de donnees IMU", file=sys.stderr)
        imu.close()
        return

    avg_ax = float(np.mean([d["ax"] for d in diag]))
    avg_ay = float(np.mean([d["ay"] for d in diag]))
    avg_az = float(np.mean([d["az"] for d in diag]))

    print(f"Acc: X={avg_ax:.3f} Y={avg_ay:.3f} Z={avg_az:.3f} m/s^2", file=sys.stderr)
    z_sign = 1 if avg_az > 0 else -1
    print(f"Z pointe vers le {'HAUT' if z_sign > 0 else 'BAS'}", file=sys.stderr)
    print("="*70, file=sys.stderr)

    init_acc = np.array([avg_ax, avg_ay, avg_az], dtype=float)
    init_acc = init_acc / (np.linalg.norm(init_acc) + 1e-12) # unitaire
    init_g_body = -init_acc

    q0 = acc2q(init_g_body)

    ekf = EKF(
        frequency=100.0,
        frame="NED",
        noises=[0.3**2, 0.5**2, 0.8**2],
        q0=q0
    )
    q=q0

    print("Demarrage EKF...\n", file=sys.stderr)

    last_t = time.time()

    try:
        while True:
            m = imu.read_measurement(timeout_s=1.0)
            if m is None:
                continue

            now = time.time()
            dt = now - last_t
            if dt <= 0.0:
                dt = 0.01
            last_t = now

            acc = np.array([m["ax"], m["ay"], m["az"]], dtype=float)
            acc_u = acc / np.linalg.norm(acc) # unitaire
            g_body = -acc_u # forces qui s'appliquent sont l'oppose de ce qui est mesure

            gyr = np.array([m["gx"], m["gy"], m["gz"]], dtype=float)
            mag = np.array([m["mx"], m["my"], m["mz"]], dtype=float)
            mag = mag*1000 # conversion uT -> nT

            # EKF
            q = ekf.update(q, gyr, g_body, mag, dt=dt)

            rotation = R.from_quat([q[1], q[2], q[3], q[0]])
            euler_angles = rotation.as_euler('zyx', degrees=True)

            yaw_deg, pitch_deg, roll_deg = euler_angles

            roll_comp, pitch_comp = (0.0, 0.0)

            orientation_data = {
                "roll": roll_deg,
                "pitch": pitch_deg,
                "yaw": yaw_deg,
                "roll_comp": roll_comp,
                "pitch_comp": pitch_comp,
                "dt": dt * 1000.0,
            }
            print(json.dumps(orientation_data), flush=True)

            print(
                f"ACC  R={roll_comp:6.1f} P={pitch_comp:6.1f} | "
                f"EKF  R={roll_deg:6.1f} P={pitch_deg:6.1f} Y={yaw_deg:6.1f} | "
                f"{dt*1000:.0f}ms  crc_bad={imu.bad_crc}/{imu.total}",
                file=sys.stderr
            )

    except KeyboardInterrupt:
        print("\nArret demande", file=sys.stderr)
    finally:
        imu.close()

if __name__ == "__main__":
    run_imu_fusion()
