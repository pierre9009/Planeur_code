import numpy as np
from imu_api import ImuSoftUart
import time
import json
import sys

def run_imu_fusion():
    imu = ImuSoftUart(rx_gpio=24, baudrate=57600)
    imu.open()

    try:
        while True:
            m = imu.read_measurement(timeout_s=1.0)
            if m is None:
                continue

            print(
                        f"seq={m["seq"]:6d} | "
                        f"A=({m["ax"]:+.3f}, {m["ay"]:+.3f}, {m["az"]:+.3f}) m/s² | "
                        f"G=({m["gx"]:+.3f}, {m["gy"]:+.3f}, {m["gz"]:+.3f}) rad/s | "
                        f"M=({m["mx"]:+.2f}, {m["my"]:+.2f}, {m["mz"]:+.2f}) uT | "
                        f"T={m["TempC"]:.2f} C"
                    )

    finally:
        imu.close()

if __name__ == "__main__":
    run_imu_fusion()