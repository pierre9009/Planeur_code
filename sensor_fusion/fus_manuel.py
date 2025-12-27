import time
import json
import sys
import numpy as np
from imu_api import ImuSoftUart
from ekf_attitude import AttitudeEKF


def run_imu_fusion():
    imu = ImuSoftUart(rx_gpio=24, baudrate=57600)
    imu.open()

    # Initialisation du filtre EKF
    ekf = AttitudeEKF()

    # Petite phase de "warmup" immobile (stabilise un peu l'EKF au démarrage)
    samples = []
    print("Initialisation (NE PAS BOUGER)...", file=sys.stderr)
    while len(samples) < 50:
        m = imu.read_measurement(timeout_s=0.5)
        if m:
            samples.append(m)

    acc_mean = np.mean([[s["ax"], s["ay"], s["az"]] for s in samples], axis=0)
    mag_mean = np.mean([[s["mx"], s["my"], s["mz"]] for s in samples], axis=0)

    # On "prime" l'EKF quelques itérations avec gyro=0 pour converger accel+mag
    # (pas aussi propre qu'un solveur ILSA, mais ça marche bien si tu es immobile)
    dt0 = 0.01
    for _ in range(200):
        q, b = ekf.update(np.zeros(3), acc_mean, mag_mean, dt0)

    print(f"✓ Quaternion initialisé: qw={q[0]:.4f}, qx={q[1]:.4f}, qy={q[2]:.4f}, qz={q[3]:.4f}", file=sys.stderr)

    last_time = None
    print("✓ Filtre EKF démarré.", file=sys.stderr)

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
            q, b = ekf.update(gyr, acc, mag, dt)

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
