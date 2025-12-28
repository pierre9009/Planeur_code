"""Input validation for sensor data."""

from typing import Optional
import numpy as np

from .types import ImuReading, ValidationResult, Quaternion
from .config import Config


class SensorValidator:
    """Validates IMU sensor readings for plausibility and safety."""

    def __init__(self, config: Config):
        """Initialize validator with configuration.

        Args:
            config: System configuration with validation thresholds.
        """
        self._config = config
        self._last_seq: Optional[int] = None
        self._last_timestamp: Optional[float] = None

    def validate_reading(self, reading: ImuReading) -> ValidationResult:
        """Validate a complete IMU reading.

        Args:
            reading: IMU measurement to validate.

        Returns:
            ValidationResult with validation status and any errors/warnings.
        """
        result = ValidationResult(is_valid=True)

        self._check_finite(reading, result)
        self._check_accelerometer(reading, result)
        self._check_gyroscope(reading, result)
        self._check_magnetometer(reading, result)
        self._check_timestamp(reading, result)
        self._check_sequence(reading, result)

        self._last_seq = reading.seq
        self._last_timestamp = reading.timestamp

        return result

    def _check_finite(self, reading: ImuReading, result: ValidationResult) -> None:
        """Check all values are finite (not NaN or Inf)."""
        values = [
            reading.ax, reading.ay, reading.az,
            reading.gx, reading.gy, reading.gz,
            reading.mx, reading.my, reading.mz,
            reading.temperature,
        ]

        for i, val in enumerate(values):
            if not np.isfinite(val):
                result.add_error(f"Non-finite value at index {i}: {val}")

    def _check_accelerometer(self, reading: ImuReading, result: ValidationResult) -> None:
        """Validate accelerometer readings."""
        cfg = self._config.sensor.accelerometer
        max_acc = cfg.range_g * cfg.gravity_nominal

        acc_mag = reading.acc_magnitude

        if abs(reading.ax) > max_acc:
            result.add_error(f"ax out of range: {reading.ax:.2f} m/s^2")
        if abs(reading.ay) > max_acc:
            result.add_error(f"ay out of range: {reading.ay:.2f} m/s^2")
        if abs(reading.az) > max_acc:
            result.add_error(f"az out of range: {reading.az:.2f} m/s^2")

        expected_g = cfg.gravity_nominal
        tolerance = cfg.gravity_tolerance

        if abs(acc_mag - expected_g) > tolerance:
            result.add_warning(
                f"Acceleration magnitude {acc_mag:.2f} deviates from "
                f"expected {expected_g:.2f} +/- {tolerance:.2f} m/s^2"
            )

    def _check_gyroscope(self, reading: ImuReading, result: ValidationResult) -> None:
        """Validate gyroscope readings."""
        cfg = self._config.sensor.gyroscope
        max_gyr = np.deg2rad(cfg.range_dps)

        if abs(reading.gx) > max_gyr:
            result.add_error(f"gx out of range: {np.rad2deg(reading.gx):.1f} deg/s")
        if abs(reading.gy) > max_gyr:
            result.add_error(f"gy out of range: {np.rad2deg(reading.gy):.1f} deg/s")
        if abs(reading.gz) > max_gyr:
            result.add_error(f"gz out of range: {np.rad2deg(reading.gz):.1f} deg/s")

    def _check_magnetometer(self, reading: ImuReading, result: ValidationResult) -> None:
        """Validate magnetometer readings."""
        cfg = self._config.sensor.magnetometer

        if abs(reading.mx) > cfg.range_ut:
            result.add_error(f"mx out of range: {reading.mx:.1f} uT")
        if abs(reading.my) > cfg.range_ut:
            result.add_error(f"my out of range: {reading.my:.1f} uT")
        if abs(reading.mz) > cfg.range_ut:
            result.add_error(f"mz out of range: {reading.mz:.1f} uT")

        mag_mag = reading.mag_magnitude

        if mag_mag < cfg.min_field_ut:
            result.add_warning(f"Magnetic field too weak: {mag_mag:.1f} uT")
        elif mag_mag > cfg.max_field_ut:
            result.add_warning(f"Magnetic field too strong: {mag_mag:.1f} uT")

    def _check_timestamp(self, reading: ImuReading, result: ValidationResult) -> None:
        """Validate timestamp monotonicity and dt."""
        if self._last_timestamp is None:
            return

        cfg = self._config.validation.timestamp
        dt = reading.timestamp - self._last_timestamp

        if dt <= 0:
            result.add_error(f"Non-monotonic timestamp: dt={dt:.6f}s")
        elif dt < cfg.min_dt_s:
            result.add_warning(f"dt too small: {dt*1000:.2f}ms")
        elif dt > cfg.max_dt_s:
            result.add_warning(f"dt too large: {dt*1000:.2f}ms")

    def _check_sequence(self, reading: ImuReading, result: ValidationResult) -> None:
        """Check for sequence number gaps."""
        if self._last_seq is None:
            return

        expected = (self._last_seq + 1) % (2**32)
        if reading.seq != expected:
            gap = (reading.seq - self._last_seq) % (2**32)
            result.add_warning(f"Sequence gap: expected {expected}, got {reading.seq} (gap: {gap})")

    def reset(self) -> None:
        """Reset validator state."""
        self._last_seq = None
        self._last_timestamp = None


class QuaternionValidator:
    """Validates quaternion state for EKF health."""

    def __init__(self, config: Config):
        """Initialize validator with configuration.

        Args:
            config: System configuration with quaternion thresholds.
        """
        self._config = config

    def validate(self, q: Quaternion) -> ValidationResult:
        """Validate quaternion state.

        Args:
            q: Quaternion to validate.

        Returns:
            ValidationResult with status and any issues.
        """
        result = ValidationResult(is_valid=True)
        cfg = self._config.validation.quaternion

        if not q._is_finite():
            result.add_error("Quaternion contains non-finite values")
            return result

        norm_error = abs(q.norm - 1.0)

        if norm_error > cfg.divergence_threshold:
            result.add_error(f"Quaternion diverged: norm={q.norm:.4f}")
        elif norm_error > cfg.norm_tolerance:
            result.add_warning(f"Quaternion norm drift: {q.norm:.6f}")

        return result

    def needs_reinitialization(self, q: Quaternion) -> bool:
        """Check if EKF needs to be reinitialized.

        Args:
            q: Current quaternion state.

        Returns:
            True if quaternion has diverged beyond recovery.
        """
        threshold = self._config.validation.quaternion.divergence_threshold
        return abs(q.norm - 1.0) > threshold or not q._is_finite()


def validate_dt(dt: float, config: Config) -> ValidationResult:
    """Validate time step for EKF update.

    Args:
        dt: Time step in seconds.
        config: System configuration.

    Returns:
        ValidationResult with status.
    """
    result = ValidationResult(is_valid=True)
    cfg = config.validation.timestamp

    if not np.isfinite(dt):
        result.add_error(f"Non-finite dt: {dt}")
    elif dt <= 0:
        result.add_error(f"Non-positive dt: {dt}")
    elif dt < cfg.min_dt_s:
        result.add_warning(f"dt too small: {dt*1000:.2f}ms")
    elif dt > cfg.max_dt_s:
        result.add_warning(f"dt too large: {dt*1000:.2f}ms")

    return result
