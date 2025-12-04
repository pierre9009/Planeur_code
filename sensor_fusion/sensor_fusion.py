#!/usr/bin/env python3
import serial
import time
import math
import json
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
            "gx_deg":     float(parts[3]),   # deg/s confirme
            "gy_deg":     float(parts[4]),
            "gz_deg":     float(parts[5]),
            "ax":         float(parts[6]),   # m/s^2
            "ay":         float(parts[7]),
            "az":         float(parts[8]),
            "mx":         float(parts[9]),   # µT
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

    sample_freq = 50.0

    # facteur d echelle gyro (1.0 puisque tu as verifie deg/s)
    gyro_scale = 1.0

    # dip magnetique approximatif France metropolitaine
    magnetic_dip_deg = 64.0

    fourati = Fourati(
        frequency=sample_freq,
        magnetic_dip=magnetic_dip_deg,
    )

    q = None

    last_t = time.time()

    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            now = time.time()
            dt = now - last_t
            if dt <= 0.0 or dt > 0.2:  # securite si gros trou
                dt = 1.0 / sample_freq
            last_t = now

            data = parse_packet(line)
            if data is None:
                continue

            # gyro deg/s -> rad/s
            gx_rad = data["gx_deg"] * gyro_scale * math.pi / 180.0
            gy_rad = data["gy_deg"] * gyro_scale * math.pi / 180.0
            gz_rad = data["gz_deg"] * gyro_scale * math.pi / 180.0
            gyr = np.array([gx_rad, gy_rad, gz_rad], dtype=float)

            # accel m/s^2
            acc = np.array([data["ax"], data["ay"], data["az"]], dtype=float)

            # mag µT -> mT
            mag = np.array(
                [data["mx"], data["my"], data["mz"]],
                dtype=float
            ) / 1000.0

            # init quaternion avec gravite au premier paquet valide
            if q is None:
                q = acc2q(acc)
                q = q / np.linalg.norm(q)
                # on ne publie pas encore, on attend la prochaine boucle
                continue

            q = fourati.update(
                q=q,
                gyr=gyr,
                acc=acc,
                mag=mag,
                dt=dt,
            )

            if q is None:
                # si Fourati renvoie None pour une raison quelconque
                continue

            angles = Quaternion(q).to_angles()
            roll_f, pitch_f, yaw_f = angles

            roll_f_deg  = math.degrees(roll_f)
            pitch_f_deg = math.degrees(pitch_f)
            yaw_f_deg   = math.degrees(yaw_f)

            payload = {
                "t": now,
                "roll_deg":  roll_f_deg,
                "pitch_deg": pitch_f_deg,
                "yaw_deg":   yaw_f_deg,
                "roll_comp": data["roll_comp"],
                "pitch_comp": data["pitch_comp"],
            }

            print(json.dumps(payload), flush=True)

        except KeyboardInterrupt:
            break
        except Exception as e:
            # tu peux logger sur stderr si tu veux
            # mais pas de print normal pour ne pas casser le JSON
            continue


if __name__ == "__main__":
    main()
