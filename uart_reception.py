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

    sample_freq = 50.0
    dt = 1.0 / sample_freq

    fourati = Fourati(
        frequency=sample_freq,
        gain=0.1,
        magnetic_dip=None,
    )

    q = None
    print("Lecture UART avec Fourati en cours")

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

            gyr = np.array([gy_rad, gx_rad, -gz_rad], dtype=float)
            acc = np.array([ay, ax, -az], dtype=float)
            mag = np.array([data["my"], data["mx"], -data["mz"]], dtype=float)
            # Si ton magnetometre donne des microTesla, active
            mag = mag / 1000.0

            if q is None:
                q = acc2q(acc)
                q = q / np.linalg.norm(q)
                print("Quaternion initial", q)
                continue

            q = fourati.update(
                q=q,
                gyr=gyr,
                acc=acc,
                mag=mag,
                dt=dt,
            )

            angles = Quaternion(q).to_angles()
            roll_f, pitch_f, yaw_f = angles

            roll_f_deg  = math.degrees(roll_f)
            pitch_f_deg = math.degrees(pitch_f)
            yaw_f_deg   = math.degrees(yaw_f)

            print(
                f"COMP R={roll_comp:7.2f} P={pitch_comp:7.2f} | "
                f"FOURATI R={roll_f_deg:7.2f} P={pitch_f_deg:7.2f} Y={yaw_f_deg:7.2f}"
            )

        except KeyboardInterrupt:
            print("Stop")
            ser.close()
            break
        except Exception as e:
            print("Erreur:", e)


if __name__ == "__main__":
    main()
