#!/usr/bin/env python3
"""Minimal IMU fusion - stripped to essentials for debugging.

Only includes:
- UART communication
- EKF fusion
- JSON output
- Basic validation

No monitoring, no complex config, no abstractions.
"""

import json
import sys
import time
import signal
import numpy as np
from typing import Optional, Tuple

from imu_api import ImuHardwareUart
from ahrs.filters import EKF
from ahrs.common.orientation import q2euler


# Configuration - all in one place
CONFIG = {
    "uart_port": "/dev/ttyS0",
    "uart_baud": 115200,
    "sample_rate": 100,
    "init_samples": 100,
    "ekf_frame": "NED",
    "output_rate": 10,  # Hz, for JSON output
}

SHUTDOWN = False


def signal_handler(sig, frame):
    global SHUTDOWN
    SHUTDOWN = True


def init_quaternion(acc: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """Compute initial quaternion from acc/mag (NED frame)."""
    acc = acc / np.linalg.norm(acc)
    mag = mag / np.linalg.norm(mag)

    down = np.array([0.0, 0.0, 1.0])
    mag_horiz = mag - np.dot(mag, down) * down
    mag_horiz = mag_horiz / np.linalg.norm(mag_horiz)

    x = mag_horiz
    z = down
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    x = x / np.linalg.norm(x)

    R = np.column_stack([x, y, z])

    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w, x, y, z = 0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w, x, y, z = (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w, x, y, z = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w, x, y, z = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def validate_reading(m: dict) -> Tuple[bool, str]:
    """Basic validation of IMU reading."""
    # Check for NaN/Inf
    values = [m["ax"], m["ay"], m["az"], m["gx"], m["gy"], m["gz"], m["mx"], m["my"], m["mz"]]
    if any(not np.isfinite(v) for v in values):
        return False, "NaN/Inf in reading"

    # Check acc magnitude
    acc_mag = np.sqrt(m["ax"]**2 + m["ay"]**2 + m["az"]**2)
    if not (5.0 < acc_mag < 15.0):
        return False, f"acc_mag={acc_mag:.2f} out of range"

    # Check mag magnitude
    mag_mag = np.sqrt(m["mx"]**2 + m["my"]**2 + m["mz"]**2)
    if not (10.0 < mag_mag < 100.0):
        return False, f"mag_mag={mag_mag:.2f} out of range"

    return True, ""


def run_fusion():
    """Main fusion loop."""
    global SHUTDOWN

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Open UART
    imu = ImuHardwareUart(port=CONFIG["uart_port"], baudrate=CONFIG["uart_baud"])
    imu.open()

    print("Collecting initialization samples...", file=sys.stderr)

    # Collect init samples
    acc_samples = []
    mag_samples = []
    for _ in range(CONFIG["init_samples"]):
        m = imu.read_measurement(timeout_s=1.0)
        if m is None:
            continue
        valid, _ = validate_reading(m)
        if valid:
            acc_samples.append([m["ax"], m["ay"], m["az"]])
            mag_samples.append([m["mx"], m["my"], m["mz"]])

    if len(acc_samples) < 50:
        print("ERROR: Not enough valid init samples", file=sys.stderr)
        imu.close()
        return 1

    acc0 = np.mean(acc_samples, axis=0)
    mag0 = np.mean(mag_samples, axis=0)

    # Initialize quaternion
    q0 = init_quaternion(acc0, mag0)

    print(f"Init: acc={acc0}, mag={mag0}", file=sys.stderr)
    print(f"Init q: {q0}", file=sys.stderr)

    # Initialize EKF
    ekf = EKF(
        gyr=np.zeros((1, 3)),
        acc=acc0.reshape((1, 3)),
        mag=mag0.reshape((1, 3)),
        frequency=CONFIG["sample_rate"],
        q0=q0,
        frame=CONFIG["ekf_frame"],
    )

    q = q0.copy()
    last_time = time.time()
    last_output = time.time()
    output_interval = 1.0 / CONFIG["output_rate"]
    iteration = 0
    errors = 0

    print("Starting fusion loop...", file=sys.stderr)

    while not SHUTDOWN:
        m = imu.read_measurement(timeout_s=0.5)
        if m is None:
            continue

        valid, err = validate_reading(m)
        if not valid:
            errors += 1
            if errors % 100 == 0:
                print(f"Validation errors: {errors} ({err})", file=sys.stderr)
            continue

        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        # Skip unreasonable dt
        if dt <= 0 or dt > 0.1:
            continue

        acc = np.array([m["ax"], m["ay"], m["az"]])
        gyr = np.array([m["gx"], m["gy"], m["gz"]])
        mag = np.array([m["mx"], m["my"], m["mz"]])

        # EKF update
        q = ekf.update(q=q, gyr=gyr, acc=acc, mag=mag, dt=dt)

        # Normalize quaternion
        q_norm = np.linalg.norm(q)
        if abs(q_norm - 1.0) > 0.1:
            print(f"WARNING: q_norm={q_norm:.4f}, reinitializing", file=sys.stderr)
            q = q0.copy()
            continue
        q = q / q_norm

        # Euler angles
        roll, pitch, yaw = q2euler(q)

        iteration += 1

        # Output at configured rate
        if current_time - last_output >= output_interval:
            output = {
                "qw": float(q[0]),
                "qx": float(q[1]),
                "qy": float(q[2]),
                "qz": float(q[3]),
                "roll": float(np.rad2deg(roll)),
                "pitch": float(np.rad2deg(pitch)),
                "yaw": float(np.rad2deg(yaw)),
                "dt": float(dt * 1000),
                "q_norm": float(q_norm),
            }
            print(json.dumps(output), flush=True)
            last_output = current_time

    # Cleanup
    imu.close()
    stats = imu.get_stats()
    print(f"\nFinal stats:", file=sys.stderr)
    print(f"  Iterations: {iteration}", file=sys.stderr)
    print(f"  Packets: {stats['total_packets']}", file=sys.stderr)
    print(f"  CRC errors: {stats['bad_crc']}", file=sys.stderr)
    print(f"  Validation errors: {errors}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(run_fusion())
