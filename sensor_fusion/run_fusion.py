#!/usr/bin/env python3
"""Single-file robust quaternion estimation from IMU.

Usage:
    python run_fusion.py                    # Run with default UART settings
    python run_fusion.py --port COM3        # Specify serial port
    python run_fusion.py --calibrate        # Run gyro calibration first
    python run_fusion.py --output json      # Output format (json/csv/minimal)

This script provides production-grade quaternion estimation with:
- Adaptive gyro bias estimation via EKF
- Temperature-dependent sensor compensation
- Magnetometer fault detection with automatic fallback
"""

import sys
import time
import json
import argparse
import signal
from typing import Optional

import numpy as np

# Add parent to path for imports
sys.path.insert(0, '.')

from core.types import ImuReading
from fusion.adaptive_ekf import AdaptiveBiasEKF
from fusion.temp_compensation import (
    TemperatureCalibration,
    TemperatureCompensator,
    create_default_calibration
)
from fusion.mag_health import MagnetometerHealthCheck
from core.quaternion import QuaternionOps
from core.types import Quaternion


class RobustFusionRunner:
    """Complete fusion system in a single class."""

    def __init__(
        self,
        temp_cal: Optional[TemperatureCalibration] = None,
        expected_mag_norm: float = 48.0
    ):
        """Initialize fusion runner.

        Args:
            temp_cal: Temperature calibration (uses defaults if None).
            expected_mag_norm: Expected local magnetic field in µT.
        """
        if temp_cal is None:
            temp_cal = create_default_calibration()

        self.temp_comp = TemperatureCompensator(temp_cal)
        self.mag_health = MagnetometerHealthCheck(expected_norm=expected_mag_norm)
        self.ekf: Optional[AdaptiveBiasEKF] = None

        self.is_initialized = False
        self.update_count = 0
        self.last_timestamp = 0.0

    def initialize_from_stationary(
        self,
        imu_source,
        duration_s: float = 3.0,
        verbose: bool = True
    ) -> bool:
        """Initialize from stationary sensor data.

        Args:
            imu_source: IMU source with read_measurement() method.
            duration_s: Calibration duration in seconds.
            verbose: Print progress messages.

        Returns:
            True if initialization successful.
        """
        if verbose:
            print(f"Initializing... keep sensor stationary for {duration_s}s")

        acc_samples = []
        gyro_samples = []
        mag_samples = []
        temp_samples = []

        start = time.time()
        while time.time() - start < duration_s:
            reading = imu_source.read_measurement(timeout_s=0.1)
            if reading is not None:
                acc_samples.append(reading.acc)
                gyro_samples.append(reading.gyr)
                mag_samples.append(reading.mag)
                temp_samples.append(reading.temperature)
            time.sleep(0.005)

        if len(acc_samples) < 10:
            if verbose:
                print("ERROR: Insufficient samples collected")
            return False

        # Compute means
        acc_mean = np.mean(acc_samples, axis=0)
        gyro_mean = np.mean(gyro_samples, axis=0)
        mag_mean = np.mean(mag_samples, axis=0)
        temp_mean = np.mean(temp_samples)

        # Initial quaternion from acc+mag
        q0 = QuaternionOps.from_acc_mag(acc_mean, mag_mean, frame="NED")

        # Gyro bias = mean reading when stationary
        initial_bias = gyro_mean

        # Set up magnetic reference
        mag_norm = np.linalg.norm(mag_mean)
        self.mag_health.update_expected_norm(mag_norm)

        # Compute magnetic reference in world frame
        mag_ref = self._compute_mag_reference(mag_mean, q0.to_array())

        # Initialize EKF
        self.ekf = AdaptiveBiasEKF(
            q0=q0.to_array(),
            initial_bias=initial_bias,
            mag_ref=mag_ref
        )

        self.is_initialized = True
        self.last_timestamp = time.time()

        if verbose:
            print(f"Initialized: {len(acc_samples)} samples")
            print(f"  Gyro bias: [{initial_bias[0]:.5f}, {initial_bias[1]:.5f}, {initial_bias[2]:.5f}] rad/s")
            print(f"  Mag norm: {mag_norm:.1f} µT")
            print(f"  Temperature: {temp_mean:.1f}°C")

        return True

    def update(self, reading: ImuReading) -> dict:
        """Process new IMU reading and return orientation.

        Args:
            reading: IMU reading with all sensor data.

        Returns:
            Dictionary with quaternion, euler angles, and status.
        """
        if not self.is_initialized or self.ekf is None:
            return {'error': 'Not initialized'}

        # Compute dt
        current_time = reading.timestamp
        if self.last_timestamp > 0:
            dt = current_time - self.last_timestamp
        else:
            dt = 0.01
        self.last_timestamp = current_time

        # Clamp dt to reasonable range
        dt = np.clip(dt, 0.001, 0.1)

        # Temperature compensation
        acc = self.temp_comp.compensate_accel(reading.acc, reading.temperature)
        gyro = self.temp_comp.compensate_gyro(reading.gyr, reading.temperature)

        # Magnetometer health check
        mag_status = self.mag_health.check(reading.mag)
        use_mag = mag_status.is_valid

        # EKF update
        self.ekf.predict(gyro, dt)
        self.ekf.update(acc, reading.mag, use_mag=use_mag)

        self.update_count += 1

        # Get results
        q = self.ekf.get_quaternion()
        quat = Quaternion.from_array(q)
        euler = QuaternionOps.to_euler(quat)
        bias = self.ekf.get_bias()

        return {
            'qw': float(q[0]),
            'qx': float(q[1]),
            'qy': float(q[2]),
            'qz': float(q[3]),
            'roll': float(np.rad2deg(euler.roll)),
            'pitch': float(np.rad2deg(euler.pitch)),
            'yaw': float(np.rad2deg(euler.yaw)),
            'bias_x': float(bias[0]),
            'bias_y': float(bias[1]),
            'bias_z': float(bias[2]),
            'mag_valid': use_mag,
            'temp': float(reading.temperature),
            'dt_ms': float(dt * 1000),
            'iteration': self.update_count
        }

    def _compute_mag_reference(
        self,
        mag: np.ndarray,
        q: np.ndarray
    ) -> np.ndarray:
        """Compute magnetic reference in world frame."""
        # Rotate mag to world frame using quaternion
        def quat_rotate(q, v):
            q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
            v_quat = np.array([0, v[0], v[1], v[2]])

            def qmul(a, b):
                return np.array([
                    a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
                    a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
                    a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
                    a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
                ])

            temp = qmul(q, v_quat)
            result = qmul(temp, q_conj)
            return result[1:4]

        mag_world = quat_rotate(q, mag)

        # Project to horizontal plane
        mag_h = np.array([mag_world[0], mag_world[1], 0.0])
        mag_h_norm = np.linalg.norm(mag_h)

        if mag_h_norm > 1e-6:
            return mag_h / mag_h_norm * np.linalg.norm(mag)
        return mag_world


def run_fusion(args):
    """Main fusion loop."""
    # Import UART interface
    try:
        from communication.uart import ImuUart
        from core.config import Config
    except ImportError:
        print("ERROR: Could not import UART module. Run from sensor_fusion directory.")
        sys.exit(1)

    # Load or create config
    try:
        config = Config.from_yaml('config/default.yaml')
        if args.port:
            config.uart.port = args.port
        if args.baudrate:
            config.uart.baudrate = args.baudrate
    except Exception:
        # Create minimal config
        from dataclasses import dataclass

        @dataclass
        class UartConfig:
            port: str = args.port or '/dev/ttyS00'
            baudrate: int = args.baudrate or 115200
            timeout_s: float = 0.1
            write_timeout_s: float = 0.1

        @dataclass
        class MinConfig:
            uart: UartConfig = None

        config = MinConfig(uart=UartConfig())

    # Load temperature calibration if available
    temp_cal = None
    if args.temp_cal:
        try:
            temp_cal = TemperatureCalibration.load(args.temp_cal)
            print(f"Loaded temperature calibration from {args.temp_cal}")
        except Exception as e:
            print(f"Warning: Could not load temp calibration: {e}")

    # Create fusion runner
    fusion = RobustFusionRunner(
        temp_cal=temp_cal,
        expected_mag_norm=args.mag_norm
    )

    # Handle Ctrl+C gracefully
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False
        print("\nShutting down...")

    signal.signal(signal.SIGINT, signal_handler)

    # Open UART and run
    print(f"Connecting to {config.uart.port} at {config.uart.baudrate} baud...")

    try:
        with ImuUart(config) as imu:
            print("Connected. Starting fusion...")

            # Initialize from stationary data
            if not fusion.initialize_from_stationary(imu, duration_s=args.init_time):
                print("Initialization failed. Exiting.")
                sys.exit(1)

            print("\nRunning fusion loop. Press Ctrl+C to stop.\n")

            output_interval = 1.0 / args.output_rate
            last_output = 0.0

            while running:
                reading = imu.read_measurement(timeout_s=0.1)
                if reading is None:
                    continue

                result = fusion.update(reading)

                # Output at specified rate
                now = time.time()
                if now - last_output >= output_interval:
                    last_output = now

                    if args.output == 'json':
                        print(json.dumps(result))
                    elif args.output == 'csv':
                        print(f"{result['roll']:.2f},{result['pitch']:.2f},{result['yaw']:.2f},"
                              f"{result['qw']:.4f},{result['qx']:.4f},{result['qy']:.4f},{result['qz']:.4f}")
                    else:  # minimal
                        mag_str = "MAG" if result['mag_valid'] else "---"
                        print(f"R:{result['roll']:7.2f} P:{result['pitch']:7.2f} Y:{result['yaw']:7.2f} "
                              f"[{mag_str}] dt:{result['dt_ms']:.1f}ms")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\nProcessed {fusion.update_count} samples.")


def main():
    parser = argparse.ArgumentParser(
        description='Robust quaternion estimation from IMU',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_fusion.py                         # Run with defaults
  python run_fusion.py --port /dev/ttyUSB0     # Specify serial port
  python run_fusion.py --output json           # JSON output for piping
  python run_fusion.py --mag-norm 52           # Set local magnetic field
        """
    )

    parser.add_argument('--port', '-p', type=str, default=None,
                        help='Serial port (default: from config)')
    parser.add_argument('--baudrate', '-b', type=int, default=None,
                        help='Baud rate (default: from config)')
    parser.add_argument('--output', '-o', type=str, default='minimal',
                        choices=['json', 'csv', 'minimal'],
                        help='Output format (default: minimal)')
    parser.add_argument('--output-rate', '-r', type=float, default=10.0,
                        help='Output rate in Hz (default: 10)')
    parser.add_argument('--init-time', '-i', type=float, default=3.0,
                        help='Initialization time in seconds (default: 3)')
    parser.add_argument('--mag-norm', '-m', type=float, default=48.0,
                        help='Expected magnetic field magnitude in µT (default: 48)')
    parser.add_argument('--temp-cal', '-t', type=str, default=None,
                        help='Path to temperature calibration JSON file')

    args = parser.parse_args()
    run_fusion(args)


if __name__ == '__main__':
    main()
