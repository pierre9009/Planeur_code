"""Tests for sensor data validation."""

import pytest
import numpy as np

from core.types import ImuReading, Quaternion
from core.validation import SensorValidator, QuaternionValidator, validate_dt


class TestSensorValidator:
    """Tests for SensorValidator class."""

    def test_valid_reading_passes(self, config, sample_imu_reading):
        """Valid IMU reading should pass validation."""
        validator = SensorValidator(config)
        result = validator.validate_reading(sample_imu_reading)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_nan_values_detected(self, config, invalid_reading_nan):
        """NaN values should be detected as errors."""
        validator = SensorValidator(config)
        result = validator.validate_reading(invalid_reading_nan)

        assert not result.is_valid
        assert any("Non-finite" in e for e in result.errors)

    def test_inf_values_detected(self, config, invalid_reading_inf):
        """Inf values should be detected as errors."""
        validator = SensorValidator(config)
        result = validator.validate_reading(invalid_reading_inf)

        assert not result.is_valid
        assert any("Non-finite" in e for e in result.errors)

    def test_acc_out_of_range(self, config, reading_out_of_range):
        """Out-of-range accelerometer values should be detected."""
        validator = SensorValidator(config)
        result = validator.validate_reading(reading_out_of_range)

        assert not result.is_valid
        assert any("out of range" in e for e in result.errors)

    def test_gyro_out_of_range(self, config):
        """Out-of-range gyroscope values should be detected."""
        reading = ImuReading(
            seq=1,
            timestamp=1000.0,
            ax=0.0, ay=0.0, az=9.81,
            gx=100.0, gy=0.0, gz=0.0,
            mx=20.0, my=5.0, mz=45.0,
            temperature=25.0,
        )
        validator = SensorValidator(config)
        result = validator.validate_reading(reading)

        assert not result.is_valid
        assert any("out of range" in e for e in result.errors)

    def test_mag_weak_field_warning(self, config):
        """Weak magnetic field should trigger warning."""
        reading = ImuReading(
            seq=1,
            timestamp=1000.0,
            ax=0.0, ay=0.0, az=9.81,
            gx=0.0, gy=0.0, gz=0.0,
            mx=1.0, my=1.0, mz=1.0,
            temperature=25.0,
        )
        validator = SensorValidator(config)
        result = validator.validate_reading(reading)

        assert result.is_valid
        assert any("too weak" in w for w in result.warnings)

    def test_mag_strong_field_warning(self, config):
        """Strong magnetic field should trigger warning."""
        reading = ImuReading(
            seq=1,
            timestamp=1000.0,
            ax=0.0, ay=0.0, az=9.81,
            gx=0.0, gy=0.0, gz=0.0,
            mx=100.0, my=100.0, mz=100.0,
            temperature=25.0,
        )
        validator = SensorValidator(config)
        result = validator.validate_reading(reading)

        assert result.is_valid
        assert any("too strong" in w for w in result.warnings)

    def test_sequence_gap_detection(self, config):
        """Sequence gaps should be detected as warnings."""
        validator = SensorValidator(config)

        reading1 = ImuReading(
            seq=10, timestamp=1000.0,
            ax=0.0, ay=0.0, az=9.81,
            gx=0.0, gy=0.0, gz=0.0,
            mx=20.0, my=5.0, mz=45.0,
            temperature=25.0,
        )
        reading2 = ImuReading(
            seq=15, timestamp=1000.01,
            ax=0.0, ay=0.0, az=9.81,
            gx=0.0, gy=0.0, gz=0.0,
            mx=20.0, my=5.0, mz=45.0,
            temperature=25.0,
        )

        validator.validate_reading(reading1)
        result = validator.validate_reading(reading2)

        assert result.is_valid
        assert any("Sequence gap" in w for w in result.warnings)

    def test_non_monotonic_timestamp(self, config):
        """Non-monotonic timestamps should be detected."""
        validator = SensorValidator(config)

        reading1 = ImuReading(
            seq=1, timestamp=1000.0,
            ax=0.0, ay=0.0, az=9.81,
            gx=0.0, gy=0.0, gz=0.0,
            mx=20.0, my=5.0, mz=45.0,
            temperature=25.0,
        )
        reading2 = ImuReading(
            seq=2, timestamp=999.0,
            ax=0.0, ay=0.0, az=9.81,
            gx=0.0, gy=0.0, gz=0.0,
            mx=20.0, my=5.0, mz=45.0,
            temperature=25.0,
        )

        validator.validate_reading(reading1)
        result = validator.validate_reading(reading2)

        assert not result.is_valid
        assert any("Non-monotonic" in e for e in result.errors)

    def test_reset_clears_state(self, config, sample_imu_reading):
        """Reset should clear validator state."""
        validator = SensorValidator(config)
        validator.validate_reading(sample_imu_reading)
        validator.reset()

        reading = ImuReading(
            seq=100, timestamp=2000.0,
            ax=0.0, ay=0.0, az=9.81,
            gx=0.0, gy=0.0, gz=0.0,
            mx=20.0, my=5.0, mz=45.0,
            temperature=25.0,
        )
        result = validator.validate_reading(reading)

        assert result.is_valid
        assert len(result.warnings) == 0


class TestQuaternionValidator:
    """Tests for QuaternionValidator class."""

    def test_valid_quaternion(self, config, identity_quaternion):
        """Valid unit quaternion should pass validation."""
        validator = QuaternionValidator(config)
        result = validator.validate(identity_quaternion)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_non_unit_quaternion_warning(self, config):
        """Non-unit quaternion should trigger warning."""
        q = Quaternion(w=0.9, x=0.0, y=0.0, z=0.0)
        validator = QuaternionValidator(config)
        result = validator.validate(q)

        assert result.is_valid
        assert any("drift" in w for w in result.warnings)

    def test_diverged_quaternion_error(self, config):
        """Diverged quaternion should trigger error."""
        q = Quaternion(w=0.5, x=0.0, y=0.0, z=0.0)
        validator = QuaternionValidator(config)
        result = validator.validate(q)

        assert not result.is_valid
        assert any("diverged" in e for e in result.errors)

    def test_nan_quaternion(self, config):
        """Quaternion with NaN should trigger error."""
        q = Quaternion(w=float("nan"), x=0.0, y=0.0, z=0.0)
        validator = QuaternionValidator(config)
        result = validator.validate(q)

        assert not result.is_valid
        assert any("non-finite" in e for e in result.errors)

    def test_needs_reinitialization(self, config):
        """needs_reinitialization should detect diverged quaternions."""
        validator = QuaternionValidator(config)

        good_q = Quaternion.identity()
        assert not validator.needs_reinitialization(good_q)

        bad_q = Quaternion(w=0.5, x=0.0, y=0.0, z=0.0)
        assert validator.needs_reinitialization(bad_q)


class TestValidateDt:
    """Tests for validate_dt function."""

    def test_valid_dt(self, config):
        """Valid dt should pass."""
        result = validate_dt(0.01, config)
        assert result.is_valid

    def test_negative_dt(self, config):
        """Negative dt should fail."""
        result = validate_dt(-0.01, config)
        assert not result.is_valid
        assert any("Non-positive" in e for e in result.errors)

    def test_zero_dt(self, config):
        """Zero dt should fail."""
        result = validate_dt(0.0, config)
        assert not result.is_valid

    def test_nan_dt(self, config):
        """NaN dt should fail."""
        result = validate_dt(float("nan"), config)
        assert not result.is_valid

    def test_dt_too_large(self, config):
        """Large dt should trigger warning."""
        result = validate_dt(0.5, config)
        assert result.is_valid
        assert any("too large" in w for w in result.warnings)

    def test_dt_too_small(self, config):
        """Small dt should trigger warning."""
        result = validate_dt(0.0001, config)
        assert result.is_valid
        assert any("too small" in w for w in result.warnings)
