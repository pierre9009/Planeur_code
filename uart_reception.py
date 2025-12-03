import serial
import time
import math

import numpy as np
from ahrs.filters import EKF
from ahrs.common.orientation import acc2q


def parse_packet(line):
    """
    D,roll,pitch,gx,gy,gz,ax,ay,az,mx,my,mz
    """
    try:
        if not line.startswith("D,"):
            return None

        parts = line.strip().split(",")
        if len(parts) != 12:  # D + 11 valeurs
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


def quat_to_euler(q):
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
    ser = serial.Serial(
        port="/dev/serial0",
        baudrate=230400,
        timeout=1.0,
    )

    time.sleep(0.2)

    sample_freq = 50.0

    sigma_g  = 1e-3          # rad/s
    sigma_a  = 0.05          # m/s^2
    sigma_m  = 0.5e-6        # Tesla

    # bruit de processus sur le quaternion (ordre de grandeur)
    Q = (sigma_g**2) * np.eye(4)

    # bruit de mesure sur acc + mag
    R_acc = (sigma_a**2) * np.eye(3)
    R_mag = (sigma_m**2) * np.eye(3)
    
    ekf = EKF(
        frequency=sample_freq,
        frame="ENU",
        Q=Q,
        R_acc=R_acc,
        R_mag=R_mag,
    )

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

            gx_rad = data["gx_deg"] * math.pi / 180.0
            gy_rad = data["gy_deg"] * math.pi / 180.0
            gz_rad = data["gz_deg"] * math.pi / 180.0
            gyr = np.array([gx_rad, gy_rad, gz_rad], dtype=float)

            acc = np.array([data["ax"], data["ay"], data["az"]], dtype=float)
            mag = np.array([data["mx"], data["my"], data["mz"]], dtype=float)

            if q is None:
                q = acc2q(acc)   # init simple
                q = q / np.linalg.norm(q)
                print("Quaternion initial:", q)
                continue

            # EKF 9 axes
            q = ekf.update(q, gyr=gyr, acc=acc, mag=mag)
            # si erreur, essayer: q = ekf.update(gyr=gyr, acc=acc, q=q)

            roll_ekf, pitch_ekf, yaw_ekf = quat_to_euler(q)

            roll_ekf_deg  = math.degrees(roll_ekf)
            pitch_ekf_deg = math.degrees(pitch_ekf)
            yaw_ekf_deg   = math.degrees(yaw_ekf)

            print(
                f"COMP R={roll_comp:7.2f} P={pitch_comp:7.2f} | "
                f"EKF R={roll_ekf_deg:7.2f} P={pitch_ekf_deg:7.2f} Y={yaw_ekf_deg:7.2f}"
            )

        except KeyboardInterrupt:
            print("\nStop.")
            ser.close()
            break
        except Exception as e:
            print("Erreur:", e)


if __name__ == "__main__":
    main()
