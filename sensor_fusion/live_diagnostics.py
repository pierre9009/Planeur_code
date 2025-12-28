#!/usr/bin/env python3
"""Real-time diagnostic dashboard for IMU sensor fusion.

Displays live sensor health metrics and drift monitoring.
"""

import sys
import time
import signal
import numpy as np
from collections import deque
from typing import Optional

from imu_api import ImuHardwareUart
from ahrs.filters import EKF
from ahrs.common.orientation import q2euler


# ANSI colors
class C:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


SHUTDOWN = False


def signal_handler(sig, frame):
    global SHUTDOWN
    SHUTDOWN = True


def init_quaternion(acc: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """Initialize quaternion from acc/mag."""
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


class LiveDiagnostics:
    """Real-time diagnostic display."""

    def __init__(self):
        self.yaw_history = deque(maxlen=1000)
        self.roll_history = deque(maxlen=1000)
        self.pitch_history = deque(maxlen=1000)
        self.dt_history = deque(maxlen=1000)
        self.mag_history = deque(maxlen=1000)
        self.acc_history = deque(maxlen=1000)

        self.start_yaw: Optional[float] = None
        self.start_time: Optional[float] = None
        self.iteration = 0
        self.errors = 0

    def update(self, roll: float, pitch: float, yaw: float, dt: float,
               acc_mag: float, gyr_mag: float, mag_mag: float, q_norm: float):
        """Update with new data."""
        self.roll_history.append(roll)
        self.pitch_history.append(pitch)
        self.yaw_history.append(yaw)
        self.dt_history.append(dt)
        self.mag_history.append(mag_mag)
        self.acc_history.append(acc_mag)

        if self.start_yaw is None:
            self.start_yaw = yaw
            self.start_time = time.time()

        self.iteration += 1

    def display(self):
        """Display current status."""
        if len(self.yaw_history) < 10:
            return

        # Compute metrics
        yaw_array = np.array(self.yaw_history)
        yaw_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(yaw_array)))

        current_yaw = yaw_unwrapped[-1]
        yaw_drift = current_yaw - yaw_unwrapped[0]

        elapsed = time.time() - self.start_time if self.start_time else 1
        drift_rate = yaw_drift / elapsed * 60

        roll = self.roll_history[-1]
        pitch = self.pitch_history[-1]

        dt_mean = np.mean(self.dt_history) * 1000
        dt_std = np.std(self.dt_history) * 1000

        acc_mean = np.mean(self.acc_history)
        mag_mean = np.mean(self.mag_history)
        mag_std = np.std(self.mag_history)

        # Status colors
        def status_color(ok):
            return C.GREEN if ok else C.RED

        drift_ok = abs(drift_rate) < 30
        timing_ok = dt_std < 5
        acc_ok = 9.5 < acc_mean < 10.1
        mag_ok = 20 < mag_mean < 70 and mag_std < 3

        # Clear screen and display
        print("\033[2J\033[H", end="")  # Clear screen, move cursor to top

        print(f"{C.BOLD}{'='*70}{C.RESET}")
        print(f"{C.BOLD}  IMU LIVE DIAGNOSTICS  -  {elapsed:.1f}s elapsed{C.RESET}")
        print(f"{C.BOLD}{'='*70}{C.RESET}")

        print(f"\n{C.BOLD}ORIENTATION:{C.RESET}")
        print(f"  Roll:  {roll:+7.2f} deg")
        print(f"  Pitch: {pitch:+7.2f} deg")
        print(f"  Yaw:   {current_yaw:+7.2f} deg")

        print(f"\n{C.BOLD}YAW DRIFT:{C.RESET}")
        color = status_color(drift_ok)
        print(f"  {color}Total:  {yaw_drift:+7.2f} deg{C.RESET}")
        print(f"  {color}Rate:   {drift_rate:+7.1f} deg/min{C.RESET}")
        if drift_ok:
            print(f"  {C.GREEN}Status: OK{C.RESET}")
        else:
            print(f"  {C.RED}Status: EXCESSIVE DRIFT!{C.RESET}")

        print(f"\n{C.BOLD}TIMING:{C.RESET}")
        color = status_color(timing_ok)
        print(f"  {color}dt mean: {dt_mean:.2f} ms{C.RESET}")
        print(f"  {color}dt std:  {dt_std:.2f} ms (jitter){C.RESET}")
        print(f"  Rate:  {1000/dt_mean:.1f} Hz")

        print(f"\n{C.BOLD}SENSORS:{C.RESET}")
        color = status_color(acc_ok)
        print(f"  {color}||acc||: {acc_mean:.3f} m/s^2 (expect 9.81){C.RESET}")
        color = status_color(mag_ok)
        print(f"  {color}||mag||: {mag_mean:.1f} +/- {mag_std:.2f} uT{C.RESET}")

        print(f"\n{C.BOLD}STATISTICS:{C.RESET}")
        print(f"  Samples: {self.iteration}")
        print(f"  Errors:  {self.errors}")

        # Quick diagnosis
        print(f"\n{C.BOLD}{'='*70}{C.RESET}")
        if abs(drift_rate) < 10:
            print(f"{C.GREEN}DRIFT: Acceptable (<10 deg/min){C.RESET}")
        elif abs(drift_rate) < 30:
            print(f"{C.YELLOW}DRIFT: Moderate (10-30 deg/min){C.RESET}")
        else:
            print(f"{C.RED}DRIFT: EXCESSIVE (>30 deg/min){C.RESET}")

            # Suggest cause
            roll_range = np.max(self.roll_history) - np.min(self.roll_history)
            pitch_range = np.max(self.pitch_history) - np.min(self.pitch_history)

            if roll_range < 5 and pitch_range < 5:
                print(f"{C.RED}Likely cause: MAGNETOMETER (roll/pitch stable){C.RESET}")
            else:
                print(f"{C.RED}Likely cause: GYRO BIAS or EKF issue{C.RESET}")

        print(f"\n{C.BLUE}Press Ctrl+C to stop{C.RESET}")


def run_diagnostics():
    """Run live diagnostics."""
    global SHUTDOWN

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    imu = ImuHardwareUart(port="/dev/ttyS0", baudrate=115200)
    imu.open()

    print("Collecting initialization samples...")

    acc_samples = []
    mag_samples = []
    for _ in range(100):
        m = imu.read_measurement(timeout_s=1.0)
        if m:
            acc_samples.append([m["ax"], m["ay"], m["az"]])
            mag_samples.append([m["mx"], m["my"], m["mz"]])

    if len(acc_samples) < 50:
        print("ERROR: Not enough init samples")
        imu.close()
        return 1

    acc0 = np.mean(acc_samples, axis=0)
    mag0 = np.mean(mag_samples, axis=0)
    q0 = init_quaternion(acc0, mag0)

    ekf = EKF(
        gyr=np.zeros((1, 3)),
        acc=acc0.reshape((1, 3)),
        mag=mag0.reshape((1, 3)),
        frequency=100,
        q0=q0,
        frame='NED'
    )

    q = q0.copy()
    last_time = time.time()
    last_display = time.time()

    diag = LiveDiagnostics()

    while not SHUTDOWN:
        m = imu.read_measurement(timeout_s=0.5)
        if m is None:
            continue

        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        if dt <= 0 or dt > 0.1:
            continue

        acc = np.array([m["ax"], m["ay"], m["az"]])
        gyr = np.array([m["gx"], m["gy"], m["gz"]])
        mag = np.array([m["mx"], m["my"], m["mz"]])

        q = ekf.update(q=q, gyr=gyr, acc=acc, mag=mag, dt=dt)
        q_norm = np.linalg.norm(q)
        q = q / q_norm

        roll, pitch, yaw = q2euler(q)

        acc_mag = np.linalg.norm(acc)
        gyr_mag = np.linalg.norm(gyr)
        mag_mag = np.linalg.norm(mag)

        diag.update(
            np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw),
            dt, acc_mag, gyr_mag, mag_mag, q_norm
        )

        # Update display every 0.5s
        if current_time - last_display >= 0.5:
            diag.display()
            last_display = current_time

    imu.close()
    print("\nShutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(run_diagnostics())
