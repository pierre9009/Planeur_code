#!/usr/bin/env python3
import time
import math
import json
import sys
import numpy as np

from ahrs.filters import EKF
from ahrs.common.orientation import acc2q

from imu_api import ImuSoftUart

def quaternion_to_euler_zyx(q):
    w, x, y, z = q[0], q[1], q[2], q[3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def remap_imu(ax, ay, az, gx, gy, gz):
    # Rotation de 90° autour de Z (cas le plus courant quand roll/pitch sont croisés)
    # Ajuste si besoin (voir plus bas)
    ax_b = ay
    ay_b = -ax
    az_b = az

    gx_b = gy
    gy_b = -gx
    gz_b = gz
    return ax_b, ay_b, az_b, gx_b, gy_b, gz_b

def acc_to_roll_pitch_deg_body(ax, ay, az):
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))
    return math.degrees(roll), math.degrees(pitch)


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

    ekf = EKF(
        frequency=100.0,
        frame="NED",
        noises=[0.3**2, 0.5**2, 0.8**2],
    )

    init_acc = np.array([avg_ax, avg_ay, avg_az], dtype=float)
    ax_b, ay_b, az_b, *_ = remap_imu(avg_ax, avg_ay, avg_az, 0.0, 0.0, 0.0)

    init_acc = np.array([ax_b, ay_b, az_b], dtype=float)
    init_acc = init_acc / (np.linalg.norm(init_acc) + 1e-12)
    q = acc2q(init_acc)
    q = q / (np.linalg.norm(q) + 1e-12)

    r0, p0, y0 = quaternion_to_euler_zyx(q)
    print(
        f"Orientation initiale: R={math.degrees(r0):.1f} P={math.degrees(p0):.1f} Y={math.degrees(y0):.1f}",
        file=sys.stderr
    )
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
            acc_norm = np.linalg.norm(acc)
            if acc_norm < 1e-6:
                continue
            g_body = -acc / acc_norm

            gyr = np.array([m["gx"], m["gy"], m["gz"]], dtype=float)



            # EKF
            q = ekf.update(q, gyr, g_body)
            q = q / (np.linalg.norm(q) + 1e-12)

            roll, pitch, yaw = quaternion_to_euler_zyx(q)
            roll_deg = math.degrees(roll)
            pitch_deg = math.degrees(pitch)
            yaw_deg = math.degrees(yaw)

            roll_comp, pitch_comp = acc_to_roll_pitch_deg(g_body[0], g_body[1], g_body[2])

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
