#!/usr/bin/env python3
"""
Main program for robust IMU fusion with temperature compensation.
Run this to get real-time quaternion estimation from ICM-20948.
"""

import sys
import time
import signal
import json
from pathlib import Path
from typing import Optional

import numpy as np

from communication.uart import ImuHardwareUart
from core import Config, load_config
from core.types import ImuReading
from sensor_fusion.fusion import RobustFusion
from sensor_fusion.temp_compensation import TemperatureCalibration
from sensor_fusion.gyro_calibration import startup_gyro_calibration


class FusionRunner:
    """Manages the complete fusion pipeline."""
    
    def __init__(self, config: Config, temp_cal_path: Optional[str] = None):
        self.config = config
        self.imu = None
        self.fusion = None
        self.running = False
        
        # Load or create temperature calibration
        if temp_cal_path and Path(temp_cal_path).exists():
            print(f"Loading temperature calibration: {temp_cal_path}")
            self.temp_cal = TemperatureCalibration.load(temp_cal_path)
        else:
            print("No temperature calibration found, using defaults")
            print("Run 'python calibration/temp_calibration.py' to calibrate")
            self.temp_cal = self._create_default_temp_cal()
    
    def _create_default_temp_cal(self) -> TemperatureCalibration:
        """Create default (uncalibrated) temperature model."""
        return TemperatureCalibration(
            gyro_bias_coeffs=np.zeros((3, 3)),  # No temp compensation
            accel_scale_coeffs=np.ones((3, 2)),  # No scaling
            calibration_temp_range=(20.0, 20.0)
        )
    
    def startup_calibration(self) -> np.ndarray:
        """Perform startup gyro bias calibration.
        
        Returns:
            Gyro bias [bx, by, bz] in rad/s
        """
        print("\n" + "="*60)
        print("STARTUP GYRO CALIBRATION")
        print("="*60)
        print("Keep sensor STATIONARY for 3 seconds...")
        
        bias = startup_gyro_calibration(
            self.imu,
            duration_s=3.0,
            motion_threshold=0.5
        )
        
        print(f"Gyro bias: [{bias[0]:.5f}, {bias[1]:.5f}, {bias[2]:.5f}] rad/s")
        
        return bias
    
    def initialize_fusion(self, num_samples: int = 100):
        """Initialize fusion from stationary samples.
        
        Args:
            num_samples: Number of samples to collect for initialization
        """
        print("\n" + "="*60)
        print("FUSION INITIALIZATION")
        print("="*60)
        print(f"Collecting {num_samples} samples (keep stationary)...")
        
        acc_samples = []
        mag_samples = []
        
        count = 0
        while count < num_samples:
            reading = self.imu.read_measurement()
            if reading is None:
                continue
            
            acc = np.array([reading.ax, reading.ay, reading.az])
            mag = np.array([reading.mx, reading.my, reading.mz])
            
            acc_samples.append(acc)
            mag_samples.append(mag)
            count += 1
            
            if count % 10 == 0:
                print(f"  {count}/{num_samples} samples collected", end='\r')
        
        print()
        
        # Get initial gyro bias from startup calibration
        gyro_bias = self.fusion._initial_bias  # Already set during startup
        
        # Initialize fusion
        self.fusion.initialize(acc_samples, mag_samples, gyro_bias)
        
        q0 = self.fusion.get_quaternion()
        euler = self.fusion.get_euler_deg()
        
        print(f"\nInitial orientation:")
        print(f"  Quaternion: [{q0[0]:.4f}, {q0[1]:.4f}, {q0[2]:.4f}, {q0[3]:.4f}]")
        print(f"  Roll:  {euler[0]:7.2f}°")
        print(f"  Pitch: {euler[1]:7.2f}°")
        print(f"  Yaw:   {euler[2]:7.2f}°")
    
    def run(self, output_mode: str = "compact"):
        """Run fusion loop.
        
        Args:
            output_mode: 'compact', 'detailed', or 'json'
        """
        print("\n" + "="*60)
        print("FUSION LOOP STARTED")
        print("="*60)
        print("Press Ctrl+C to stop\n")
        
        if output_mode == "compact":
            print("Time    | Roll    | Pitch   | Yaw     | Temp  | Mag  | Bias")
            print("-" * 70)
        
        last_time = None
        iteration = 0
        
        try:
            while self.running:
                reading = self.imu.read_measurement()
                if reading is None:
                    continue
                
                current_time = reading.timestamp
                
                # Calculate dt
                if last_time is None:
                    last_time = current_time
                    continue
                
                dt = current_time - last_time
                last_time = current_time
                
                # Update fusion
                q = self.fusion.update(reading, dt)
                
                # Output results
                if iteration % 10 == 0:  # Print every 10th iteration
                    self._output_state(reading, iteration, dt, output_mode)
                
                iteration += 1
                
        except KeyboardInterrupt:
            print("\n\nStopping fusion...")
    
    def _output_state(self, reading: ImuReading, iteration: int, dt: float, mode: str):
        """Output current state."""
        euler = self.fusion.get_euler_deg()
        bias = self.fusion.get_bias_estimate()
        mag_valid = self.fusion.mag_health.is_valid(
            np.array([reading.mx, reading.my, reading.mz])
        )
        
        if mode == "compact":
            print(f"{iteration*dt:7.1f}s | "
                  f"{euler[0]:7.2f}° | "
                  f"{euler[1]:7.2f}° | "
                  f"{euler[2]:7.2f}° | "
                  f"{reading.temperature:5.1f}C | "
                  f"{'OK' if mag_valid else 'FAIL':4s} | "
                  f"[{bias[2]:.3f}]")
        
        elif mode == "detailed":
            q = self.fusion.get_quaternion()
            print(f"\n{'='*60}")
            print(f"Iteration {iteration} (t={iteration*dt:.1f}s, dt={dt*1000:.1f}ms)")
            print(f"{'='*60}")
            print(f"Quaternion: [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
            print(f"Euler:      Roll={euler[0]:7.2f}°, Pitch={euler[1]:7.2f}°, Yaw={euler[2]:7.2f}°")
            print(f"Gyro bias:  [{bias[0]:.5f}, {bias[1]:.5f}, {bias[2]:.5f}] rad/s")
            print(f"Temperature: {reading.temperature:.2f}°C")
            print(f"Mag status: {'VALID' if mag_valid else 'INVALID'}")
        
        elif mode == "json":
            output = {
                "timestamp": iteration * dt,
                "quaternion": {
                    "w": float(self.fusion.get_quaternion()[0]),
                    "x": float(self.fusion.get_quaternion()[1]),
                    "y": float(self.fusion.get_quaternion()[2]),
                    "z": float(self.fusion.get_quaternion()[3])
                },
                "euler": {
                    "roll": float(euler[0]),
                    "pitch": float(euler[1]),
                    "yaw": float(euler[2])
                },
                "gyro_bias": {
                    "x": float(bias[0]),
                    "y": float(bias[1]),
                    "z": float(bias[2])
                },
                "temperature": float(reading.temperature),
                "mag_valid": bool(mag_valid)
            }
            print(json.dumps(output))
    
    def start(self, output_mode: str = "compact"):
        """Complete startup sequence and run fusion.
        
        Args:
            output_mode: Output format ('compact', 'detailed', 'json')
        """
        try:
            # Open UART
            print("\n" + "="*60)
            print("OPENING IMU COMMUNICATION")
            print("="*60)
            self.imu = ImuHardwareUart(self.config)
            self.imu.open()
            
            # Initialize fusion with temp calibration
            self.fusion = RobustFusion(self.temp_cal)
            
            # Startup gyro calibration
            gyro_bias = self.startup_calibration()
            self.fusion._initial_bias = gyro_bias  # Store for later
            
            # Initialize from samples
            self.initialize_fusion(num_samples=100)
            
            # Run fusion loop
            self.running = True
            self.run(output_mode=output_mode)
            
        finally:
            if self.imu:
                self.imu.close()
                print("IMU closed")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nShutdown signal received")
    sys.exit(0)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run robust IMU fusion with temperature compensation"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "-t", "--temp-cal",
        type=str,
        default="config/temp_calibration.json",
        help="Path to temperature calibration file"
    )
    parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=["compact", "detailed", "json"],
        default="compact",
        help="Output mode"
    )
    
    args = parser.parse_args()
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration")
        config = Config()
    
    # Run fusion
    runner = FusionRunner(config, temp_cal_path=args.temp_cal)
    runner.start(output_mode=args.mode)


if __name__ == "__main__":
    main()