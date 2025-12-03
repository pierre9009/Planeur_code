import serial
import time
import math

import numpy as np
from ahrs.filters import EKF
from ahrs.common.orientation import acc2q


def parse_packet(line):
    """
    Attend une ligne type:
    D,roll,pitch,gx,gy,gz,ax,ay,az

    Retourne un dict ou None si invalide.
    """
    try:
        if not line.startswith("D,"):
            return None

        parts = line.strip().split(",")
        if len(parts) != 9 + 1:  # "D" + 9 valeurs
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
        }
    except ValueError:
        return None


def quat_to_euler(q):
    """
    Convertit un quaternion [w, x, y, z] en angles Euler (rad)
    Convention type avion:
      roll  rotation autour de X
      pitch rotation autour de Y
      yaw   rotation autour de Z
    """
    qw, qx, qy, qz = q

    # roll (x)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y)
    sinp = 2.0 * (qw * qy - qz * qx)
    if sinp >= 1.0:
        pitch = math.pi / 2.0
    elif sinp <= -1.0:
        pitch = -math.pi / 2.0
    else:
        pitch = math.asin(sinp)

    # yaw (z)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def main():
    # UART du Pi, adapte si besoin
    ser = serial.Serial(
        port="/dev/serial0",
        baudrate=230400,
        timeout=1.0,
    )

    time.sleep(0.2)

    # Fréquence d’échantillonnage envoyée par l’Arduino
    sample_freq = 50.0

    # EKF de la lib "ahrs"
    ekf = EKF(frequency=sample_freq, frame="ENU")

    q = None
    print("Lecture UART + EKF en cours...")

    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            data = parse_packet(line)
            if data is None:
                continue

            roll_comp  = data["roll_comp"]
            pitch_comp = data["pitch_comp"]

            # Gyro en deg/s -> rad/s pour l’EKF
            gx_rad = data["gx_deg"] * math.pi / 180.0
            gy_rad = data["gy_deg"] * math.pi / 180.0
            gz_rad = data["gz_deg"] * math.pi / 180.0
            gyr = np.array([gx_rad, gy_rad, gz_rad], dtype=float)

            # Accélération en m/s^2
            acc = np.array([data["ax"], data["ay"], data["az"]], dtype=float)

            # Initialisation du quaternion avec l’accéléro seul
            if q is None:
                q = acc2q(acc)  # quaternion initial approximatif
                # Optionnel: normalisation explicite
                q = q / np.linalg.norm(q)
                print("Quaternion initial:", q)
                continue

            # Mise à jour EKF (sans mag pour l’instant)
            q = ekf.update(q, gyr, acc)

            # Conversion en Euler
            roll_ekf, pitch_ekf, yaw_ekf = quat_to_euler(q)

            roll_ekf_deg  = math.degrees(roll_ekf)
            pitch_ekf_deg = math.degrees(pitch_ekf)
            yaw_ekf_deg   = math.degrees(yaw_ekf)

            # Affichage comparatif
            print(
                f"COMP   R={roll_comp:7.2f}  P={pitch_comp:7.2f}  |  "
                f"EKF   R={roll_ekf_deg:7.2f}  P={pitch_ekf_deg:7.2f}  Y={yaw_ekf_deg:7.2f}"
            )

        except KeyboardInterrupt:
            print("\nStop.")
            ser.close()
            break
        except Exception as e:
            print("Erreur:", e)


if __name__ == "__main__":
    main()
