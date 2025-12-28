#!/usr/bin/env python3
"""Data recorder for capturing real IMU sensor data.

Records raw sensor readings along with computed orientation for
offline analysis and root cause diagnosis of yaw drift.
"""

import json
import time
import sys
import signal
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from imu_api import ImuHardwareUart


@dataclass
class RawDataSnapshot:
    """Single snapshot of all sensor data and computed values."""
    timestamp: float
    seq: int

    # Raw accelerometer (m/s^2)
    ax: float
    ay: float
    az: float

    # Raw gyroscope (rad/s)
    gx: float
    gy: float
    gz: float

    # Raw magnetometer (uT)
    mx: float
    my: float
    mz: float

    # Temperature
    temp_c: float

    # Timing
    dt: float

    # Computed magnitudes
    acc_mag: float
    gyr_mag: float
    mag_mag: float

    # Quaternion from EKF (w, x, y, z)
    qw: float
    qx: float
    qy: float
    qz: float

    # Euler angles (degrees)
    roll: float
    pitch: float
    yaw: float

    # Quaternion norm (should be 1.0)
    q_norm: float


@dataclass
class DataRecording:
    """Complete recording session."""
    start_time: str
    duration_s: float
    sample_count: int
    description: str
    snapshots: List[RawDataSnapshot]

    # Metadata
    uart_port: str
    baudrate: int
    ekf_frame: str

    def save(self, filepath: str) -> None:
        """Save recording to JSON file."""
        data = {
            "start_time": self.start_time,
            "duration_s": self.duration_s,
            "sample_count": self.sample_count,
            "description": self.description,
            "uart_port": self.uart_port,
            "baudrate": self.baudrate,
            "ekf_frame": self.ekf_frame,
            "snapshots": [asdict(s) for s in self.snapshots],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {self.sample_count} samples to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "DataRecording":
        """Load recording from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        snapshots = [RawDataSnapshot(**s) for s in data["snapshots"]]
        return cls(
            start_time=data["start_time"],
            duration_s=data["duration_s"],
            sample_count=data["sample_count"],
            description=data["description"],
            snapshots=snapshots,
            uart_port=data["uart_port"],
            baudrate=data["baudrate"],
            ekf_frame=data["ekf_frame"],
        )


class DataRecorder:
    """Records IMU data with EKF fusion for offline analysis."""

    def __init__(self, port: str = "/dev/ttyS0", baudrate: int = 115200):
        self._port = port
        self._baudrate = baudrate
        self._shutdown = False

    def record(
        self,
        duration_s: float,
        description: str = "",
        output_dir: str = "recordings",
    ) -> DataRecording:
        """Record sensor data for specified duration.

        Args:
            duration_s: Recording duration in seconds.
            description: Description of recording conditions.
            output_dir: Directory to save recording.

        Returns:
            DataRecording with all captured snapshots.
        """
        from ahrs.filters import EKF
        from ahrs.common.orientation import q2euler

        Path(output_dir).mkdir(exist_ok=True)

        imu = ImuHardwareUart(port=self._port, baudrate=self._baudrate)
        imu.open()

        snapshots: List[RawDataSnapshot] = []

        print(f"Collecting initialization samples...")

        # Collect init samples
        acc_init = []
        mag_init = []
        for _ in range(100):
            m = imu.read_measurement(timeout_s=1.0)
            if m:
                acc_init.append([m["ax"], m["ay"], m["az"]])
                mag_init.append([m["mx"], m["my"], m["mz"]])

        acc0 = np.mean(acc_init, axis=0)
        mag0 = np.mean(mag_init, axis=0)

        # Initialize quaternion from acc/mag
        q0 = self._init_quaternion(acc0, mag0)

        print(f"Initial quaternion: {q0}")
        print(f"Initial acc: {acc0}, mag: {mag0}")

        # Initialize EKF
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
        start_time = time.time()

        print(f"Recording for {duration_s} seconds...")
        print("Press Ctrl+C to stop early")

        signal.signal(signal.SIGINT, lambda s, f: setattr(self, '_shutdown', True))

        while not self._shutdown and (time.time() - start_time) < duration_s:
            m = imu.read_measurement(timeout_s=0.5)
            if m is None:
                continue

            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            acc = np.array([m["ax"], m["ay"], m["az"]])
            gyr = np.array([m["gx"], m["gy"], m["gz"]])
            mag = np.array([m["mx"], m["my"], m["mz"]])

            # EKF update
            q = ekf.update(q=q, gyr=gyr, acc=acc, mag=mag, dt=dt)

            # Euler angles
            roll, pitch, yaw = q2euler(q)

            snapshot = RawDataSnapshot(
                timestamp=current_time,
                seq=m["seq"],
                ax=m["ax"], ay=m["ay"], az=m["az"],
                gx=m["gx"], gy=m["gy"], gz=m["gz"],
                mx=m["mx"], my=m["my"], mz=m["mz"],
                temp_c=m["tempC"],
                dt=dt,
                acc_mag=float(np.linalg.norm(acc)),
                gyr_mag=float(np.linalg.norm(gyr)),
                mag_mag=float(np.linalg.norm(mag)),
                qw=float(q[0]), qx=float(q[1]),
                qy=float(q[2]), qz=float(q[3]),
                roll=float(np.rad2deg(roll)),
                pitch=float(np.rad2deg(pitch)),
                yaw=float(np.rad2deg(yaw)),
                q_norm=float(np.linalg.norm(q)),
            )
            snapshots.append(snapshot)

            # Progress
            if len(snapshots) % 100 == 0:
                elapsed = current_time - start_time
                print(f"  {elapsed:.1f}s: yaw={snapshot.yaw:.1f} roll={snapshot.roll:.1f} pitch={snapshot.pitch:.1f}")

        imu.close()

        end_time = time.time()
        recording = DataRecording(
            start_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            duration_s=end_time - start_time,
            sample_count=len(snapshots),
            description=description,
            snapshots=snapshots,
            uart_port=self._port,
            baudrate=self._baudrate,
            ekf_frame="NED",
        )

        # Auto-save
        filename = f"{output_dir}/recording_{int(start_time)}.json"
        recording.save(filename)

        return recording

    def _init_quaternion(
        self,
        acc: np.ndarray,
        mag: np.ndarray,
    ) -> np.ndarray:
        """Compute initial quaternion from accelerometer and magnetometer."""
        # Normalize
        acc_norm = acc / np.linalg.norm(acc)
        mag_norm = mag / np.linalg.norm(mag)

        # NED frame: gravity points +Z (down)
        down = np.array([0.0, 0.0, 1.0])

        # Project mag to horizontal plane
        mag_horiz = mag_norm - np.dot(mag_norm, down) * down
        mag_horiz = mag_horiz / np.linalg.norm(mag_horiz)

        # Build rotation matrix
        x_axis = mag_horiz  # North
        z_axis = down       # Down
        y_axis = np.cross(z_axis, x_axis)  # East

        # Normalize
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        R = np.column_stack([x_axis, y_axis, z_axis])

        # Rotation matrix to quaternion
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s

        q = np.array([w, x, y, z])
        return q / np.linalg.norm(q)


def main():
    """Command-line interface for data recording."""
    import argparse

    parser = argparse.ArgumentParser(description="Record IMU sensor data")
    parser.add_argument("-d", "--duration", type=float, default=30.0,
                        help="Recording duration in seconds")
    parser.add_argument("-o", "--output", type=str, default="recordings",
                        help="Output directory")
    parser.add_argument("-m", "--message", type=str, default="",
                        help="Description of recording conditions")
    parser.add_argument("-p", "--port", type=str, default="/dev/ttyS0",
                        help="UART port")
    args = parser.parse_args()

    recorder = DataRecorder(port=args.port)
    recording = recorder.record(
        duration_s=args.duration,
        description=args.message,
        output_dir=args.output,
    )

    # Quick summary
    if recording.snapshots:
        yaw_start = recording.snapshots[0].yaw
        yaw_end = recording.snapshots[-1].yaw
        drift = yaw_end - yaw_start
        drift_rate = drift / recording.duration_s * 60

        print(f"\nSummary:")
        print(f"  Samples: {recording.sample_count}")
        print(f"  Duration: {recording.duration_s:.1f}s")
        print(f"  Yaw drift: {drift:.1f} deg ({drift_rate:.1f} deg/min)")


if __name__ == "__main__":
    main()
