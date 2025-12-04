import serial
import time
import math
import numpy as np

from ahrs.filters import Fourati
from ahrs.common.orientation import acc2q
from ahrs.common.quaternion import Quaternion


def parse_packet(line: str):
    try:
        if not line.startswith("D,"):
            return None

        parts = line.strip().split(",")
        if len(parts) != 12:
            return None

        return {
            "roll_comp":  float(parts[1]),
            "pitch_comp": float(parts[2]),
            "gx_deg":     float(parts[3]),   # ATTENTION: on suppose deg/s
            "gy_deg":     float(parts[4]),
            "gz_deg":     float(parts[5]),
            "ax":         float(parts[6]),   # m/s^2 (DFRobot)
            "ay":         float(parts[7]),
            "az":         float(parts[8]),
            "mx":         float(parts[9]),   # µT (DFRobot)
            "my":         float(parts[10]),
            "mz":         float(parts[11]),
        }
    except ValueError:
        return None


def main():
    ser = serial.Serial(
        port="/dev/serial0",
        baudrate=230400,
        timeout=1.0,
    )

    time.sleep(0.2)
    print("Port serie ouvert, test de lecture brute")

    for _ in range(3):
        raw = ser.readline().decode(errors="ignore").strip()
        if raw:
            print("RAW:", repr(raw))

    # Frequence nominale d envoi des trames Arduino
    sample_freq = 50.0

    # Coefficient de mise a l echelle du gyro
    #  - Commence a 1.0
    #  - Ajuste le apres ton test de verification cote Arduino
    gyro_scale = 1.0

    # Dip magnetique approximatif pour la France metropolitaine
    # En gros autour de 63 a 65 degres selon la zone
    # Tu peux affiner avec WMM via ahrs.utils.WMM plus tard.
    magnetic_dip_deg = 64.0

    fourati = Fourati(
        frequency=sample_freq,
        magnetic_dip=magnetic_dip_deg,
    )

    q = None
    print("Lecture UART avec Fourati en cours")

    # Mesure du temps pour dt
    last_t = time.time()

    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            now = time.time()
            dt = now - last_t
            # petite securite si jamais il y a un gros trou
            if dt <= 0.0:
                dt = 1.0 / sample_freq
            last_t = now

            data = parse_packet(line)
            if data is None:
                continue

            roll_comp  = data["roll_comp"]
            pitch_comp = data["pitch_comp"]

            # gyro deg/s -> rad/s, avec eventuel facteur d echelle
            gx_rad = data["gx_deg"] * gyro_scale * math.pi / 180.0
            gy_rad = data["gy_deg"] * gyro_scale * math.pi / 180.0
            gz_rad = data["gz_deg"] * gyro_scale * math.pi / 180.0
            gyr = np.array([gx_rad, gy_rad, gz_rad], dtype=float)

            # accel deja en m/s^2 dans la lib DFRobot
            acc = np.array([data["ax"], data["ay"], data["az"]], dtype=float)

            # mag en µT -> mT pour Fourati
            mag = np.array([data["mx"], data["my"], data["mz"]], dtype=float) / 1000.0

            # init quaternion avec la gravite
            if q is None:
                q = acc2q(acc)  # acc en repere corps, unites coherentes, pas critique
                q = q / np.linalg.norm(q)
                print("Quaternion initial", q)
                continue

            # mise a jour du filtre Fourati
            q = fourati.update(
                q=q,
                gyr=gyr,   # rad/s
                acc=acc,   # m/s^2
                mag=mag,   # mT
                dt=dt,     # seconde, mesuree en temps reel
            )

            angles = Quaternion(q).to_angles()
            roll_f, pitch_f, yaw_f = angles

            roll_f_deg  = math.degrees(roll_f)
            pitch_f_deg = math.degrees(pitch_f)
            yaw_f_deg   = math.degrees(yaw_f)

            print(
                f"COMP R={roll_comp:7.2f} P={pitch_comp:7.2f} | "
                f"FOURATI R={roll_f_deg:7.2f} P={pitch_f_deg:7.2f} Y={yaw_f_deg:7.2f} | "
                f"dt={dt*1000.0:6.2f} ms"
            )

        except KeyboardInterrupt:
            print("Stop")
            ser.close()
            break
        except Exception as e:
            print("Erreur:", e)


if __name__ == "__main__":
    main()
