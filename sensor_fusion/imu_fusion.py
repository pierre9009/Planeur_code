# ... imports ...

from ahrs.filters import EKF
from scipy.spatial.transform import Rotation as R
import numpy as np
import time, json, sys
from imu_api import ImuSoftUart

G = 9.80665

def q_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)

def q_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)

def init_q0_from_acc_only(ax, ay, az):
    # Roll/pitch depuis l'acc (yaw = 0)
    # Convention standard: acc mesure environ +g sur l’axe “down” quand c’est à plat
    roll  = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay*ay + az*az))
    rot = R.from_euler('yx', [pitch, roll], degrees=False)  # yaw=0
    q_xyzw = rot.as_quat()
    return q_xyzw_to_wxyz(q_xyzw)

def run_imu_fusion():
    imu = ImuSoftUart(rx_gpio=24, baudrate=57600)
    imu.open()

    diag = []
    t0 = time.time()
    while len(diag) < 30 and (time.time() - t0) < 5.0:
        m = imu.read_measurement(timeout_s=0.5)
        if m is not None:
            diag.append(m)

    if not diag:
        print("ERREUR: Pas de donnees IMU", file=sys.stderr)
        imu.close()
        return

    avg_ax = float(np.mean([d["ax"] for d in diag]))
    avg_ay = float(np.mean([d["ay"] for d in diag]))
    avg_az = float(np.mean([d["az"] for d in diag]))

    avg_mx = float(np.mean([d["mx"] for d in diag]))
    avg_my = float(np.mean([d["my"] for d in diag]))
    avg_mz = float(np.mean([d["mz"] for d in diag]))

    print(f"Acc: X={avg_ax:.3f} Y={avg_ay:.3f} Z={avg_az:.3f} m/s^2", file=sys.stderr)
    print(f"Mag: X={avg_mx:.2f} Y={avg_my:.2f} Z={avg_mz:.2f} uT", file=sys.stderr)

    q0 = init_q0_from_acc_only(avg_ax, avg_ay, avg_az)

    # IMPORTANT: forcer l’EKF en mode 6D en donnant mag à la construction
    mag0_nt = np.array([[avg_mx*1000.0, avg_my*1000.0, avg_mz*1000.0]], dtype=float)  # uT -> nT
    acc0 = np.array([[avg_ax, avg_ay, avg_az]], dtype=float)
    gyr0 = np.zeros((1, 3), dtype=float)

    ekf = EKF(
        gyr=gyr0,
        acc=acc0,
        mag=mag0_nt,
        frequency=100.0,
        frame="NED",
        q0=q0,
        noises=[0.3**2, 0.5**2, 0.8**2],
        mag_ref=60.0,
        magnetic_ref=60.0,
    )

    q = q0
    last_t = time.time()

    print("Demarrage EKF...", file=sys.stderr)

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
            gyr = np.array([m["gx"], m["gy"], m["gz"]], dtype=float)
            mag = np.array([m["mx"], m["my"], m["mz"]], dtype=float) * 1000.0  # uT -> nT

            # Ici: la doc demande acc en m/s^2, mag en nT, gyr en rad/s
            q = ekf.update(q, gyr, acc, mag, dt=dt)

            rot = R.from_quat(q_wxyz_to_xyzw(q))
            yaw_deg, pitch_deg, roll_deg = rot.as_euler('zyx', degrees=True)

            print(json.dumps({
                "roll": float(roll_deg),
                "pitch": float(pitch_deg),
                "yaw": float(yaw_deg),
                "dt": float(dt*1000.0),
            }), flush=True)

    finally:
        imu.close()

if __name__ == "__main__":
    run_imu_fusion()
