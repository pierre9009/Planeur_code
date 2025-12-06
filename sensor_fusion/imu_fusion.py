"""
Fusion IMU avec Extended Kalman Filter (EKF) - Bibliothèque AHRS

Ce script lit les données IMU calibrées depuis l'Arduino (BMX160) via UART
et estime l'orientation 3D en utilisant un filtre EKF.

Format des données reçues (Arduino):
- D,roll_comp,pitch_comp,gx,gy,gz,ax,ay,az,mx,my,mz
- Gyroscope: deg/s (converti en rad/s ici)
- Accéléromètre: m/s² (déjà calibré)
- Magnétomètre: µT (converti en mT et normalisé)

Référence AHRS EKF:
- La méthode update() prend: update(q_prev, gyr, acc) pour mode 6 axes
- Pour mode 9 axes (MARG), l'EKF attend acc et mag concaténés en un seul vecteur
- Doc: https://ahrs.readthedocs.io/en/latest/filters/ekf.html
"""

import serial
import time
import math
import numpy as np
import json
import sys

from ahrs.filters import EKF
from ahrs.common.orientation import acc2q


def parse_packet(line: str):
    """Parse une ligne de données IMU au format CSV."""
    try:
        if not line.startswith("D,"):
            return None
        parts = line.strip().split(",")
        if len(parts) != 12:
            return None
        return {
            "roll_comp":  float(parts[1]),
            "pitch_comp": float(parts[2]),
            "gx_deg":     float(parts[3]),
            "gy_deg":     float(parts[4]),
            "gz_deg":     float(parts[5]),
            "ax":         float(parts[6]),
            "ay":         float(parts[7]),
            "az":         float(parts[8]),
            "mx":         float(parts[9]),
            "my":         float(parts[10]),
            "mz":         float(parts[11]),
        }
    except ValueError:
        return None


def quaternion_to_euler_zyx(q):
    """
    Convertit un quaternion [w, x, y, z] en angles d'Euler (roll, pitch, yaw).
    Convention ZYX (Yaw-Pitch-Roll) intrinsèque.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Roll (rotation autour de X)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (rotation autour de Y)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Gimbal lock
    else:
        pitch = math.asin(sinp)
    
    # Yaw (rotation autour de Z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def run_imu_fusion():
    """Processus principal de fusion IMU avec EKF."""
    
    # Ouverture port série
    ser = serial.Serial(port="/dev/serial0", baudrate=230400, timeout=1.0)
    time.sleep(0.2)
    
    print("="*70, file=sys.stderr)
    print("INITIALISATION - Placez le capteur À PLAT", file=sys.stderr)
    print("="*70, file=sys.stderr)

    # Nettoyer le buffer et collecter des échantillons
    time.sleep(1)
    for _ in range(10):
        ser.readline()
    
    diag_samples = []
    for _ in range(30):
        line = ser.readline().decode(errors="ignore").strip()
        data = parse_packet(line)
        if data:
            diag_samples.append(data)
    
    if not diag_samples:
        print("ERREUR: Pas de données reçues!", file=sys.stderr)
        return

    # Calcul moyennes pour déterminer l'orientation initiale
    avg_ax = np.mean([d["ax"] for d in diag_samples])
    avg_ay = np.mean([d["ay"] for d in diag_samples])
    avg_az = np.mean([d["az"] for d in diag_samples])
    
    print(f"Accéléromètre: X={avg_ax:.2f} Y={avg_ay:.2f} Z={avg_az:.2f} m/s²", file=sys.stderr)
    
    # Déterminer si Z pointe vers le haut ou le bas
    z_sign = 1 if avg_az > 0 else -1
    print(f"Z pointe vers le {'HAUT' if z_sign > 0 else 'BAS'}", file=sys.stderr)
    print("="*70, file=sys.stderr)

    # Configuration EKF
    # frame='NED' car Z pointe vers le bas dans la convention aéronautique
    # noises = [var_gyro, var_acc, var_mag] - variances du bruit de mesure
    ekf = EKF(
        frequency=50.0,
        frame='NED',
        noises=[0.3**2, 0.5**2, 0.8**2],  # Ajustez selon votre capteur
    )

    # Quaternion initial depuis l'accéléromètre
    init_acc = np.array([avg_ax, avg_ay, avg_az * z_sign])
    init_acc = init_acc / np.linalg.norm(init_acc)
    q = acc2q(init_acc)
    q = q / np.linalg.norm(q)
    
    roll, pitch, yaw = quaternion_to_euler_zyx(q)
    print(f"Orientation initiale: R={math.degrees(roll):.1f}° P={math.degrees(pitch):.1f}° Y={math.degrees(yaw):.1f}°", file=sys.stderr)
    print("Démarrage de l'EKF...\n", file=sys.stderr)

    last_t = time.time()

    # Boucle principale
    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            # Calcul dt
            now = time.time()
            dt = now - last_t
            if dt <= 0.0:
                dt = 0.02  # 50 Hz par défaut
            last_t = now

            data = parse_packet(line)
            if data is None:
                continue

            # Gyroscope: deg/s → rad/s
            gyr = np.array([
                data["gx_deg"] * math.pi / 180.0,
                data["gy_deg"] * math.pi / 180.0,
                data["gz_deg"] * math.pi / 180.0 * z_sign
            ], dtype=float)

            # Accéléromètre: m/s² (normalisé)
            acc = np.array([data["ax"], data["ay"], data["az"] * z_sign], dtype=float)
            acc_norm = np.linalg.norm(acc)
            if acc_norm < 0.1:
                continue
            acc = acc / acc_norm

            # Magnétomètre: µT → mT (normalisé)
            mag = np.array([data["mx"], data["my"], data["mz"] * z_sign], dtype=float) / 1000.0
            mag_norm = np.linalg.norm(mag)
            if mag_norm > 0.001:
                mag = mag / mag_norm
            else:
                mag = None

            # Mise à jour EKF
            # IMPORTANT: L'EKF de AHRS ne supporte PAS le magnétomètre en 3ème argument
            # Il faut utiliser mode 6 axes (gyro + acc uniquement)
            # Pour mode 9 axes, il faudrait concaténer acc et mag en un seul vecteur
            # Signature: update(q_prev, gyr, acc)
            q = ekf.update(q, gyr, acc)
            
            # Normalisation du quaternion
            q = q / np.linalg.norm(q)

            # Conversion en angles d'Euler
            roll, pitch, yaw = quaternion_to_euler_zyx(q)
            roll_deg  = math.degrees(roll)
            pitch_deg = math.degrees(pitch)
            yaw_deg   = math.degrees(yaw)

            # Envoi JSON sur stdout (pour le serveur web)
            orientation_data = {
                'roll': roll_deg,
                'pitch': pitch_deg,
                'yaw': yaw_deg,
                'roll_comp': data["roll_comp"],
                'pitch_comp': data["pitch_comp"],
                'dt': dt * 1000.0
            }
            print(json.dumps(orientation_data), flush=True)

            # Log sur stderr
            print(
                f"COMP R={data['roll_comp']:6.1f} P={data['pitch_comp']:6.1f} | "
                f"EKF R={roll_deg:6.1f} P={pitch_deg:6.1f} Y={yaw_deg:6.1f} | "
                f"{dt*1000:.0f}ms",
                file=sys.stderr
            )

        except KeyboardInterrupt:
            print("\nArrêt demandé", file=sys.stderr)
            break
        except Exception as e:
            print(f"Erreur: {e}", file=sys.stderr)


if __name__ == '__main__':
    run_imu_fusion()