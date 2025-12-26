import numpy as np
from imu_api import ImuSoftUart
import time
import json
import sys
from ahrs.filters import EKF

def run_imu_fusion():
    imu = ImuSoftUart(rx_gpio=24, baudrate=57600)
    imu.open()


    sigma_g = 0.05
    sigma_a = 0.2
    sigma_m = 0.5
    
    ekf = EKF(
        gyr=np.zeros((1, 3)),
        acc=acc0.reshape((1,3)),
        mag=mag0.reshape((1,3)),
        frequency=100.0,
        frame="ENU",
        magnetic_ref=mag_ref_enu,
        noises=[sigma_g, sigma_a, sigma_m] # Bruits ajustés pour plus de stabilité
    )

    last_seq=0
    try:
        while True:
            m = imu.read_measurement(timeout_s=1.0)
            if m is None:
                continue
            #print(f"seq={m["seq"]:6d} | "f"A=({m["ax"]:+.3f}, {m["ay"]:+.3f}, {m["az"]:+.3f}) m/s² | "f"G=({m["gx"]:+.3f}, {m["gy"]:+.3f}, {m["gz"]:+.3f}) rad/s | "f"M=({m["mx"]:+.2f}, {m["my"]:+.2f}, {m["mz"]:+.2f}) uT | "f"T={m["tempC"]:.2f} C")
            acc = np.array([m["ax"], m["ay"], m["az"]], dtype=float)
            gyr = np.array([m["gx"], m["gy"], m["gz"]], dtype=float)
            mag = np.array([m["mx"], m["my"], m["mz"]], dtype=float)
            dt = (m["seq"] - last_seq)*0.010
            last_seq = m["seq"]



    finally:
        imu.close()

if __name__ == "__main__":
    run_imu_fusion()