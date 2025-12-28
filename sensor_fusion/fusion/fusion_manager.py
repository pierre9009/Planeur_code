"""Robust sensor fusion manager.

Main interface combining:
- AdaptiveBiasEKF for orientation estimation
- TemperatureCompensator for thermal drift correction
- MagnetometerHealthCheck for magnetic interference detection

Provides a single, simple API for quaternion estimation from IMU data.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .adaptive_ekf import AdaptiveBiasEKF
from .temp_compensation import (
    TemperatureCalibration,
    TemperatureCompensator,
    create_default_calibration
)
from .mag_health import MagnetometerHealthCheck
from core.types import ImuReading, Quaternion, EulerAngles
from core.quaternion import QuaternionOps


@dataclass
class FusionStatus:
    """Current status of the fusion algorithm."""
    quaternion: NDArray[np.float64]  # [w, x, y, z]
    euler_deg: NDArray[np.float64]   # [roll, pitch, yaw] in degrees
    gyro_bias: NDArray[np.float64]   # [bx, by, bz] in rad/s
    mag_valid: bool
    temperature: float
    update_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'qw': float(self.quaternion[0]),
            'qx': float(self.quaternion[1]),
            'qy': float(self.quaternion[2]),
            'qz': float(self.quaternion[3]),
            'roll_deg': float(self.euler_deg[0]),
            'pitch_deg': float(self.euler_deg[1]),
            'yaw_deg': float(self.euler_deg[2]),
            'gyro_bias_x': float(self.gyro_bias[0]),
            'gyro_bias_y': float(self.gyro_bias[1]),
            'gyro_bias_z': float(self.gyro_bias[2]),
            'mag_valid': self.mag_valid,
            'temperature': self.temperature,
            'update_count': self.update_count
        }


class RobustFusion:
    """Main fusion interface combining all components.

    Provides temperature-compensated, bias-corrected quaternion estimation
    with automatic magnetometer fallback.

    Usage:
        fusion = RobustFusion(temp_calibration)
        fusion.initialize(acc_samples, mag_samples, gyro_samples, temp_samples)

        while True:
            reading = imu.read()
            quaternion = fusion.update(reading, dt)
    """

    def __init__(
        self,
        temp_cal: Optional[TemperatureCalibration] = None,
        expected_mag_norm: float = 48.0,
        frame: str = "NED"
    ):
        """Initialize fusion manager.

        Args:
            temp_cal: Temperature calibration. If None, uses defaults.
            expected_mag_norm: Expected local magnetic field magnitude in µT.
            frame: Reference frame ('NED' or 'ENU').
        """
        if temp_cal is None:
            temp_cal = create_default_calibration()

        self.temp_comp = TemperatureCompensator(temp_cal)
        self.mag_health = MagnetometerHealthCheck(expected_norm=expected_mag_norm)
        self.ekf: Optional[AdaptiveBiasEKF] = None
        self.frame = frame

        # State tracking
        self.is_initialized = False
        self.update_count = 0
        self.last_mag_valid = True
        self.last_temperature = 25.0

    def initialize(
        self,
        acc_samples: List[NDArray[np.float64]],
        mag_samples: List[NDArray[np.float64]],
        gyro_samples: List[NDArray[np.float64]],
        temp_samples: List[float]
    ) -> bool:
        """Initialize from stationary samples.

        Computes initial orientation and gyro bias from stationary data.

        Args:
            acc_samples: Stationary accelerometer readings (m/s²).
            mag_samples: Stationary magnetometer readings (µT).
            gyro_samples: Stationary gyro readings (rad/s).
            temp_samples: Corresponding temperatures (°C).

        Returns:
            True if initialization successful.

        Raises:
            ValueError: If insufficient samples provided.
        """
        if len(acc_samples) < 10:
            raise ValueError("Need at least 10 samples for initialization")

        # Compute mean temperature
        mean_temp = np.mean(temp_samples)
        self.last_temperature = mean_temp

        # Estimate initial gyro bias from stationary data
        gyro_array = np.array(gyro_samples)
        gyro_mean = np.mean(gyro_array, axis=0)

        # Apply temperature compensation to get corrected bias estimate
        temp_predicted_bias = self.temp_comp.get_expected_bias_at_temp(mean_temp)

        # Combine stationary measurement with temperature model
        # The stationary gyro reading IS the bias (when stationary)
        initial_bias = gyro_mean

        # Compute initial quaternion from acc+mag
        acc_array = np.array(acc_samples)
        mag_array = np.array(mag_samples)
        acc_mean = np.mean(acc_array, axis=0)
        mag_mean = np.mean(mag_array, axis=0)

        q0 = self._quaternion_from_acc_mag(acc_mean, mag_mean)

        # Initialize magnetic field reference
        mag_norm = np.linalg.norm(mag_mean)
        self.mag_health.update_expected_norm(mag_norm)

        # Initialize EKF
        self.ekf = AdaptiveBiasEKF(
            q0=q0,
            initial_bias=initial_bias,
            mag_ref=self._compute_mag_reference(mag_mean, q0)
        )

        self.is_initialized = True
        return True

    def initialize_simple(
        self,
        acc: NDArray[np.float64],
        mag: NDArray[np.float64],
        gyro_bias: Optional[NDArray[np.float64]] = None
    ) -> bool:
        """Simple initialization from single readings.

        For quick startup when stationary data isn't available.

        Args:
            acc: Single accelerometer reading (m/s²).
            mag: Single magnetometer reading (µT).
            gyro_bias: Initial gyro bias estimate (rad/s), or zeros.

        Returns:
            True if initialization successful.
        """
        if gyro_bias is None:
            gyro_bias = np.zeros(3)

        q0 = self._quaternion_from_acc_mag(acc, mag)

        mag_norm = np.linalg.norm(mag)
        self.mag_health.update_expected_norm(mag_norm)

        self.ekf = AdaptiveBiasEKF(
            q0=q0,
            initial_bias=gyro_bias,
            mag_ref=self._compute_mag_reference(mag, q0)
        )

        self.is_initialized = True
        return True

    def update(self, reading: ImuReading, dt: float) -> NDArray[np.float64]:
        """Update fusion with new sensor reading.

        Args:
            reading: IMU reading with all sensor data.
            dt: Time since last update in seconds.

        Returns:
            Quaternion [w,x,y,z] representing current orientation.

        Raises:
            RuntimeError: If not initialized.
        """
        if not self.is_initialized or self.ekf is None:
            raise RuntimeError("Fusion not initialized. Call initialize() first.")

        # Extract sensor data
        acc_raw = reading.acc
        gyro_raw = reading.gyr
        mag_raw = reading.mag
        temp = reading.temperature

        self.last_temperature = temp

        # Temperature compensation
        acc = self.temp_comp.compensate_accel(acc_raw, temp)
        gyro = self.temp_comp.compensate_gyro(gyro_raw, temp)

        # Check magnetometer health
        mag_status = self.mag_health.check(mag_raw)
        mag_valid = mag_status.is_valid
        self.last_mag_valid = mag_valid

        # EKF prediction
        self.ekf.predict(gyro, dt)

        # EKF update
        self.ekf.update(acc, mag_raw, use_mag=mag_valid)

        self.update_count += 1

        return self.ekf.get_quaternion()

    def update_raw(
        self,
        acc: NDArray[np.float64],
        gyro: NDArray[np.float64],
        mag: NDArray[np.float64],
        temp: float,
        dt: float
    ) -> NDArray[np.float64]:
        """Update fusion with raw sensor arrays.

        Alternative to update() when not using ImuReading dataclass.

        Args:
            acc: Accelerometer [ax,ay,az] in m/s².
            gyro: Gyroscope [gx,gy,gz] in rad/s.
            mag: Magnetometer [mx,my,mz] in µT.
            temp: Temperature in °C.
            dt: Time step in seconds.

        Returns:
            Quaternion [w,x,y,z].
        """
        if not self.is_initialized or self.ekf is None:
            raise RuntimeError("Fusion not initialized")

        self.last_temperature = temp

        # Temperature compensation
        acc_comp = self.temp_comp.compensate_accel(acc, temp)
        gyro_comp = self.temp_comp.compensate_gyro(gyro, temp)

        # Check magnetometer health
        mag_valid = self.mag_health.is_valid(mag)
        self.last_mag_valid = mag_valid

        # EKF steps
        self.ekf.predict(gyro_comp, dt)
        self.ekf.update(acc_comp, mag, use_mag=mag_valid)

        self.update_count += 1

        return self.ekf.get_quaternion()

    def get_quaternion(self) -> NDArray[np.float64]:
        """Get current quaternion estimate."""
        if self.ekf is None:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return self.ekf.get_quaternion()

    def get_euler(self) -> Tuple[float, float, float]:
        """Get current Euler angles in radians.

        Returns:
            (roll, pitch, yaw) in radians.
        """
        q = self.get_quaternion()
        quat = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
        euler = QuaternionOps.to_euler(quat)
        return (euler.roll, euler.pitch, euler.yaw)

    def get_euler_deg(self) -> Tuple[float, float, float]:
        """Get current Euler angles in degrees.

        Returns:
            (roll, pitch, yaw) in degrees.
        """
        roll, pitch, yaw = self.get_euler()
        return (
            np.rad2deg(roll),
            np.rad2deg(pitch),
            np.rad2deg(yaw)
        )

    def get_bias_estimate(self) -> NDArray[np.float64]:
        """Get current gyro bias estimate in rad/s."""
        if self.ekf is None:
            return np.zeros(3)
        return self.ekf.get_bias()

    def get_status(self) -> FusionStatus:
        """Get complete fusion status."""
        q = self.get_quaternion()
        euler = self.get_euler_deg()

        return FusionStatus(
            quaternion=q,
            euler_deg=np.array(euler),
            gyro_bias=self.get_bias_estimate(),
            mag_valid=self.last_mag_valid,
            temperature=self.last_temperature,
            update_count=self.update_count
        )

    def _quaternion_from_acc_mag(
        self,
        acc: NDArray[np.float64],
        mag: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute initial quaternion from accelerometer and magnetometer.

        Args:
            acc: Accelerometer reading (m/s²).
            mag: Magnetometer reading (µT).

        Returns:
            Quaternion [w,x,y,z].
        """
        quat = QuaternionOps.from_acc_mag(acc, mag, frame=self.frame)
        return quat.to_array()

    def _compute_mag_reference(
        self,
        mag: NDArray[np.float64],
        q: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute magnetic reference vector in world frame.

        Args:
            mag: Magnetometer reading in body frame.
            q: Current orientation quaternion.

        Returns:
            Magnetic reference in world frame (horizontal projection).
        """
        # Rotate mag to world frame
        quat = Quaternion.from_array(q)
        q_arr = quat.to_array()

        # q ⊗ m ⊗ q*
        m_quat = np.array([0, mag[0], mag[1], mag[2]])

        q_conj = np.array([q_arr[0], -q_arr[1], -q_arr[2], -q_arr[3]])

        def qmul(a, b):
            return np.array([
                a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
                a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
                a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
                a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
            ])

        temp = qmul(q_arr, m_quat)
        m_world = qmul(temp, q_conj)[1:4]

        # Project to horizontal (NED: zero out Z component)
        m_horizontal = np.array([m_world[0], m_world[1], 0.0])
        m_norm = np.linalg.norm(m_horizontal)

        if m_norm > 1e-6:
            return m_horizontal / m_norm * np.linalg.norm(mag)
        else:
            return m_world

    def reset(self) -> None:
        """Reset fusion state."""
        self.ekf = None
        self.is_initialized = False
        self.update_count = 0
        self.mag_health.reset()


def startup_gyro_calibration(
    imu_source,
    duration_s: float = 3.0,
    sample_rate_hz: float = 100.0
) -> NDArray[np.float64]:
    """Quick gyro bias calibration at startup.

    Assumes sensor is stationary for the duration.

    Args:
        imu_source: IMU source with read_measurement() method.
        duration_s: Calibration duration in seconds.
        sample_rate_hz: Expected sample rate.

    Returns:
        Gyro bias [bx, by, bz] in rad/s.
    """
    import time

    samples = []
    start = time.time()
    expected_samples = int(duration_s * sample_rate_hz)

    while time.time() - start < duration_s:
        reading = imu_source.read_measurement()
        if reading is not None:
            samples.append(reading.gyr)

        # Brief sleep to not hammer CPU
        time.sleep(1.0 / sample_rate_hz / 2)

    if len(samples) < 10:
        return np.zeros(3)

    gyro_array = np.array(samples)
    bias = np.mean(gyro_array, axis=0)

    return bias


def collect_initialization_samples(
    imu_source,
    num_samples: int = 100,
    timeout_s: float = 5.0
) -> Tuple[
    List[NDArray[np.float64]],
    List[NDArray[np.float64]],
    List[NDArray[np.float64]],
    List[float]
]:
    """Collect stationary samples for fusion initialization.

    Args:
        imu_source: IMU source with read_measurement() method.
        num_samples: Number of samples to collect.
        timeout_s: Maximum time to wait.

    Returns:
        Tuple of (acc_samples, mag_samples, gyro_samples, temp_samples).
    """
    import time

    acc_samples = []
    mag_samples = []
    gyro_samples = []
    temp_samples = []

    start = time.time()

    while len(acc_samples) < num_samples:
        if time.time() - start > timeout_s:
            break

        reading = imu_source.read_measurement()
        if reading is not None:
            acc_samples.append(reading.acc)
            mag_samples.append(reading.mag)
            gyro_samples.append(reading.gyr)
            temp_samples.append(reading.temperature)

        time.sleep(0.005)  # 200 Hz max

    return acc_samples, mag_samples, gyro_samples, temp_samples
