import serial
import time
import math
import numpy as np

from ahrs.filters import EKF
from ahrs.common.orientation import acc2q


def parse_packet(line: str):
    """
    Trame Arduino:
    D,roll,pitch,gx,gy,gz,ax,ay,az,mx,my,mz
    """
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


def quat_to_euler(q):
    """Quaternion [w,x,y,z] -> (roll, pitch, yaw) en rad."""
    qw, qx, qy, qz = q

    # roll (x)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qz)
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
        port="/dev/serial0",   # adapte si besoin (/dev/ttyAMA0 etc.)
        baudrate=230400,
        timeout=1.0,
    )

    time.sleep(0.2)
    print("Port serie ouvert, test de lecture brute...")

    # petit apercu des lignes reçues
    for _ in range(3):
        raw = ser.readline().decode(errors="ignore").strip()
        if raw:
            print("RAW:", repr(raw))

    sample_freq = 50.0

    # d après la doc ahrs: usage streaming
    # ekf = EKF()
    # for t: q = ekf.update(q, gyr[t], acc[t])
    # on ajoute juste frequency et frame
    ekf = EKF(
        frequency=sample_freq,
        frame="ENU",
    )

    q = None
    print("Lecture UART + EKF gyro+acc+mag en cours...")

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

            # gyro deg/s -> rad/s
            gx_rad = data["gx_deg"] * math.pi / 180.0
            gy_rad = data["gy_deg"] * math.pi / 180.0
            gz_rad = data["gz_deg"] * math.pi / 180.0
            gyr = np.array([gx_rad, gy_rad, gz_rad], dtype=float)

            # accel m/s^2
            acc = np.array([data["ax"], data["ay"], data["az"]], dtype=float)

            # mag en uT
            mag = np.array([data["mx"], data["my"], data["mz"]], dtype=float)

            # init quaternion avec l'acceléromètre
            if q is None:
                q = acc2q(acc)
                q = q / np.linalg.norm(q)
                print("Quaternion initial:", q)
                continue

            # d après la lib: update(q, gyr, acc)
            # et dans les versions récentes, mag est supporté comme 4e argument
            try:
                q = ekf.update(q, gyr, acc, mag)
            except TypeError:
                # si jamais ta version ne prend que 3 args:
                # on met mag comme état interne et on appelle update(q, gyr, acc)
                ekf.mag = mag
                q = ekf.update(q, gyr, acc)

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
