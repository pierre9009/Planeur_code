"""Gyroscope calibration utilities.

Provides startup calibration and stationarity validation for gyroscope
bias estimation.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple
import time


class ImuSource(Protocol):
    """Protocol for IMU data source."""

    def read_measurement(self, timeout_s: float = 1.0):
        """Read single IMU measurement."""
        ...


@dataclass
class CalibrationResult:
    """Result of gyroscope calibration."""
    bias: NDArray[np.float64]      # [bx, by, bz] in rad/s
    std: NDArray[np.float64]       # Standard deviation
    num_samples: int               # Number of samples used
    duration_s: float              # Actual calibration duration
    is_valid: bool                 # Whether calibration passed checks
    mean_temperature: float        # Mean temperature during calibration
    warnings: List[str]            # Any warning messages

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'bias': self.bias.tolist(),
            'std': self.std.tolist(),
            'num_samples': self.num_samples,
            'duration_s': self.duration_s,
            'is_valid': self.is_valid,
            'mean_temperature': self.mean_temperature,
            'warnings': self.warnings
        }


def startup_gyro_calibration(
    imu_source: ImuSource,
    duration_s: float = 3.0,
    max_std_rad: float = 0.01,
    max_bias_rad: float = 0.1
) -> CalibrationResult:
    """Quick gyro bias calibration at startup.

    Assumes sensor is stationary for the calibration duration.
    Validates that sensor is actually stationary by checking variance.

    Args:
        imu_source: IMU data source.
        duration_s: Calibration duration in seconds.
        max_std_rad: Maximum allowed standard deviation (rad/s).
        max_bias_rad: Maximum allowed bias magnitude (rad/s).

    Returns:
        CalibrationResult with bias estimate and quality metrics.
    """
    samples = []
    temperatures = []
    warnings = []

    start_time = time.time()

    while time.time() - start_time < duration_s:
        reading = imu_source.read_measurement(timeout_s=0.1)
        if reading is not None:
            samples.append(reading.gyr.copy())
            temperatures.append(reading.temperature)

        time.sleep(0.005)  # 200 Hz max sampling

    actual_duration = time.time() - start_time

    if len(samples) < 10:
        return CalibrationResult(
            bias=np.zeros(3),
            std=np.ones(3) * float('inf'),
            num_samples=len(samples),
            duration_s=actual_duration,
            is_valid=False,
            mean_temperature=0.0,
            warnings=["Insufficient samples collected"]
        )

    gyro_array = np.array(samples)
    bias = np.mean(gyro_array, axis=0)
    std = np.std(gyro_array, axis=0)
    mean_temp = float(np.mean(temperatures))

    is_valid = True

    # Check standard deviation (sensor should be stable)
    if np.any(std > max_std_rad):
        warnings.append(
            f"High variance detected (std={std}). Sensor may not be stationary."
        )
        is_valid = False

    # Check bias magnitude (shouldn't be unreasonably large)
    bias_magnitude = np.linalg.norm(bias)
    if bias_magnitude > max_bias_rad:
        warnings.append(
            f"Large bias detected ({bias_magnitude:.4f} rad/s). "
            "Sensor may be faulty or moving."
        )
        is_valid = False

    return CalibrationResult(
        bias=bias,
        std=std,
        num_samples=len(samples),
        duration_s=actual_duration,
        is_valid=is_valid,
        mean_temperature=mean_temp,
        warnings=warnings
    )


def validate_stationary(
    imu_source: ImuSource,
    duration_s: float = 1.0,
    max_gyro_rad: float = 0.05,
    gravity_tolerance: float = 0.5
) -> Tuple[bool, str]:
    """Validate that sensor is stationary.

    Checks both gyroscope (low angular rate) and accelerometer
    (measuring only gravity).

    Args:
        imu_source: IMU data source.
        duration_s: Validation duration.
        max_gyro_rad: Maximum allowed gyro magnitude (rad/s).
        gravity_tolerance: Allowed deviation from 9.81 m/s² (m/s²).

    Returns:
        (is_stationary, reason) tuple.
    """
    gyro_samples = []
    accel_magnitudes = []

    start_time = time.time()

    while time.time() - start_time < duration_s:
        reading = imu_source.read_measurement(timeout_s=0.1)
        if reading is not None:
            gyro_samples.append(reading.gyr.copy())
            accel_magnitudes.append(reading.acc_magnitude)

        time.sleep(0.005)

    if len(gyro_samples) < 5:
        return False, "Insufficient samples"

    # Check gyroscope
    gyro_array = np.array(gyro_samples)
    gyro_magnitudes = np.linalg.norm(gyro_array, axis=1)
    max_gyro_measured = np.max(gyro_magnitudes)

    if max_gyro_measured > max_gyro_rad:
        return False, f"Sensor rotating (max gyro: {max_gyro_measured:.4f} rad/s)"

    # Check accelerometer (should be near 9.81 m/s²)
    mean_accel_mag = np.mean(accel_magnitudes)
    accel_error = abs(mean_accel_mag - 9.81)

    if accel_error > gravity_tolerance:
        return False, f"Accelerometer deviation ({accel_error:.3f} m/s² from gravity)"

    return True, "Stationary"


def continuous_bias_estimation(
    samples: List[NDArray[np.float64]],
    window_size: int = 100
) -> NDArray[np.float64]:
    """Estimate gyro bias from continuous samples.

    Uses windowed averaging for online bias estimation.
    Only valid during quasi-stationary periods.

    Args:
        samples: List of gyro readings.
        window_size: Number of samples to average.

    Returns:
        Current bias estimate.
    """
    if len(samples) < window_size:
        if len(samples) > 0:
            return np.mean(samples, axis=0)
        return np.zeros(3)

    recent = samples[-window_size:]
    return np.mean(recent, axis=0)


def detect_motion(
    gyro_samples: List[NDArray[np.float64]],
    threshold_rad: float = 0.02,
    window_size: int = 10
) -> bool:
    """Detect if sensor is in motion.

    Uses short-term variance to detect motion events.

    Args:
        gyro_samples: Recent gyro readings.
        threshold_rad: Variance threshold for motion detection.
        window_size: Window size for variance calculation.

    Returns:
        True if motion detected.
    """
    if len(gyro_samples) < window_size:
        return False

    recent = np.array(gyro_samples[-window_size:])
    variance = np.var(recent, axis=0)
    max_variance = np.max(variance)

    return max_variance > threshold_rad ** 2


class AdaptiveBiasEstimator:
    """Adaptive gyro bias estimator for continuous operation.

    Updates bias estimate only during stationary periods.
    Maintains uncertainty estimate for bias.
    """

    def __init__(
        self,
        initial_bias: Optional[NDArray[np.float64]] = None,
        learning_rate: float = 0.01,
        motion_threshold: float = 0.02,
        window_size: int = 50
    ):
        """Initialize estimator.

        Args:
            initial_bias: Initial bias estimate (zeros if None).
            learning_rate: How fast to update bias during stationary.
            motion_threshold: Gyro variance threshold for motion.
            window_size: Window size for motion detection.
        """
        self.bias = initial_bias if initial_bias is not None else np.zeros(3)
        self.learning_rate = learning_rate
        self.motion_threshold = motion_threshold
        self.window_size = window_size

        self.gyro_history: List[NDArray[np.float64]] = []
        self.is_stationary = True
        self.stationary_count = 0
        self.motion_count = 0

    def update(self, gyro: NDArray[np.float64]) -> NDArray[np.float64]:
        """Update bias estimate with new gyro reading.

        Args:
            gyro: Raw gyro reading [gx, gy, gz] in rad/s.

        Returns:
            Corrected gyro (gyro - bias).
        """
        # Add to history
        self.gyro_history.append(gyro.copy())
        if len(self.gyro_history) > self.window_size * 2:
            self.gyro_history.pop(0)

        # Detect motion state
        in_motion = detect_motion(
            self.gyro_history,
            threshold_rad=self.motion_threshold,
            window_size=self.window_size
        )

        if in_motion:
            self.is_stationary = False
            self.motion_count += 1
        else:
            self.is_stationary = True
            self.stationary_count += 1

            # Update bias estimate during stationary periods
            if len(self.gyro_history) >= self.window_size:
                recent_mean = np.mean(
                    self.gyro_history[-self.window_size:],
                    axis=0
                )
                # Exponential moving average update
                self.bias = (
                    (1 - self.learning_rate) * self.bias +
                    self.learning_rate * recent_mean
                )

        # Return corrected gyro
        return gyro - self.bias

    def get_bias(self) -> NDArray[np.float64]:
        """Get current bias estimate."""
        return self.bias.copy()

    def reset(self, initial_bias: Optional[NDArray[np.float64]] = None) -> None:
        """Reset estimator state."""
        self.bias = initial_bias if initial_bias is not None else np.zeros(3)
        self.gyro_history.clear()
        self.is_stationary = True
        self.stationary_count = 0
        self.motion_count = 0

    def get_stats(self) -> dict:
        """Get estimator statistics."""
        total = self.stationary_count + self.motion_count
        return {
            'bias': self.bias.tolist(),
            'is_stationary': self.is_stationary,
            'stationary_ratio': (
                self.stationary_count / max(1, total)
            ),
            'history_size': len(self.gyro_history)
        }
