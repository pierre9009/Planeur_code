"""Tests for temperature compensation."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_allclose
import tempfile
import os

from fusion.temp_compensation import (
    TempCalPoint,
    TemperatureCalibration,
    TemperatureCompensator,
    create_default_calibration
)


class TestTempCalPoint:
    """Tests for TempCalPoint dataclass."""

    def test_creation(self):
        """Test creating a calibration point."""
        point = TempCalPoint(
            temperature=25.0,
            gyro_bias=np.array([0.01, -0.005, 0.002]),
            accel_scale=np.array([1.0, 1.0, 1.0]),
            duration_s=1800.0,
            std_gyro=np.array([0.001, 0.001, 0.001]),
            std_accel=np.array([0.01, 0.01, 0.01])
        )

        assert point.temperature == 25.0
        assert_array_almost_equal(point.gyro_bias, [0.01, -0.005, 0.002])


class TestTemperatureCalibration:
    """Tests for TemperatureCalibration."""

    def test_identity_calibration(self):
        """Test identity calibration applies no corrections."""
        cal = TemperatureCalibration.identity()

        # Gyro bias should be zero at any temperature
        for temp in [-10, 0, 20, 40]:
            bias = (cal.gyro_bias_coeffs[:, 0] +
                   cal.gyro_bias_coeffs[:, 1] * temp +
                   cal.gyro_bias_coeffs[:, 2] * temp**2)
            assert_array_almost_equal(bias, np.zeros(3))

        # Accel scale should be 1.0 at any temperature
        for temp in [-10, 0, 20, 40]:
            scale = cal.accel_scale_coeffs[:, 0] + cal.accel_scale_coeffs[:, 1] * temp
            assert_array_almost_equal(scale, np.ones(3))

    def test_from_calibration_data_linear(self):
        """Test fitting from two temperature points (linear)."""
        points = [
            TempCalPoint(
                temperature=0.0,
                gyro_bias=np.array([0.01, 0.01, 0.01]),
                accel_scale=np.array([0.99, 0.99, 0.99]),
                duration_s=1800,
                std_gyro=np.zeros(3),
                std_accel=np.zeros(3)
            ),
            TempCalPoint(
                temperature=40.0,
                gyro_bias=np.array([0.05, 0.05, 0.05]),
                accel_scale=np.array([1.01, 1.01, 1.01]),
                duration_s=1800,
                std_gyro=np.zeros(3),
                std_accel=np.zeros(3)
            )
        ]

        cal = TemperatureCalibration.from_calibration_data(points)

        # Check that model interpolates correctly
        # At T=20, bias should be 0.03
        comp = TemperatureCompensator(cal)
        bias_20 = comp._predict_gyro_bias(20.0)
        assert_allclose(bias_20, [0.03, 0.03, 0.03], atol=1e-6)

    def test_from_calibration_data_quadratic(self):
        """Test fitting from three temperature points (quadratic)."""
        points = [
            TempCalPoint(
                temperature=-10.0,
                gyro_bias=np.array([0.02, 0.0, 0.0]),
                accel_scale=np.ones(3),
                duration_s=1800,
                std_gyro=np.zeros(3),
                std_accel=np.zeros(3)
            ),
            TempCalPoint(
                temperature=20.0,
                gyro_bias=np.array([0.01, 0.0, 0.0]),
                accel_scale=np.ones(3),
                duration_s=1800,
                std_gyro=np.zeros(3),
                std_accel=np.zeros(3)
            ),
            TempCalPoint(
                temperature=50.0,
                gyro_bias=np.array([0.03, 0.0, 0.0]),
                accel_scale=np.ones(3),
                duration_s=1800,
                std_gyro=np.zeros(3),
                std_accel=np.zeros(3)
            )
        ]

        cal = TemperatureCalibration.from_calibration_data(points)

        # Should have non-zero quadratic term
        assert cal.gyro_bias_coeffs[0, 2] != 0

    def test_insufficient_points(self):
        """Test that single point raises error."""
        points = [
            TempCalPoint(
                temperature=20.0,
                gyro_bias=np.zeros(3),
                accel_scale=np.ones(3),
                duration_s=1800,
                std_gyro=np.zeros(3),
                std_accel=np.zeros(3)
            )
        ]

        with pytest.raises(ValueError):
            TemperatureCalibration.from_calibration_data(points)

    def test_save_load(self):
        """Test saving and loading calibration."""
        cal = create_default_calibration()

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            filepath = f.name

        try:
            cal.save(filepath)
            loaded = TemperatureCalibration.load(filepath)

            assert_array_almost_equal(
                loaded.gyro_bias_coeffs, cal.gyro_bias_coeffs
            )
            assert_array_almost_equal(
                loaded.accel_scale_coeffs, cal.accel_scale_coeffs
            )
            assert loaded.calibration_temp_range == cal.calibration_temp_range
        finally:
            os.unlink(filepath)


class TestTemperatureCompensator:
    """Tests for TemperatureCompensator."""

    def test_gyro_compensation(self):
        """Test gyroscope temperature compensation."""
        # Create calibration with known bias model
        gyro_coeffs = np.array([
            [0.01, 0.001, 0.0],  # X: bias = 0.01 + 0.001*T
            [0.0, 0.0, 0.0],     # Y: no bias
            [-0.01, 0.001, 0.0]  # Z: bias = -0.01 + 0.001*T
        ])

        cal = TemperatureCalibration(
            gyro_bias_coeffs=gyro_coeffs,
            accel_scale_coeffs=np.column_stack([np.ones(3), np.zeros(3)]),
            calibration_temp_range=(-10.0, 50.0)
        )

        comp = TemperatureCompensator(cal)

        # At T=10°C, expected bias is [0.02, 0.0, 0.0]
        gyro_raw = np.array([0.12, 0.05, 0.03])
        gyro_comp = comp.compensate_gyro(gyro_raw, 10.0)

        expected_bias = np.array([0.02, 0.0, 0.0])
        expected_comp = gyro_raw - expected_bias

        assert_allclose(gyro_comp, expected_comp)

    def test_accel_compensation(self):
        """Test accelerometer temperature compensation."""
        # Create calibration with known scale model
        accel_coeffs = np.array([
            [1.0, 0.001],   # X: scale = 1.0 + 0.001*T
            [1.0, -0.001],  # Y: scale = 1.0 - 0.001*T
            [1.0, 0.0]      # Z: scale = 1.0 (no temp dependency)
        ])

        cal = TemperatureCalibration(
            gyro_bias_coeffs=np.zeros((3, 3)),
            accel_scale_coeffs=accel_coeffs,
            calibration_temp_range=(-10.0, 50.0)
        )

        comp = TemperatureCompensator(cal)

        # At T=10°C, scale factors are [1.01, 0.99, 1.0]
        accel_raw = np.array([1.0, 1.0, 9.81])
        accel_comp = comp.compensate_accel(accel_raw, 10.0)

        expected_scale = np.array([1.01, 0.99, 1.0])
        expected_comp = accel_raw * expected_scale

        assert_allclose(accel_comp, expected_comp, atol=1e-6)

    def test_temperature_clamping(self):
        """Test that temperatures are clamped to valid range."""
        cal = TemperatureCalibration(
            gyro_bias_coeffs=np.array([
                [0.0, 0.01, 0.0],
                [0.0, 0.01, 0.0],
                [0.0, 0.01, 0.0]
            ]),
            accel_scale_coeffs=np.column_stack([np.ones(3), np.zeros(3)]),
            calibration_temp_range=(0.0, 40.0)
        )

        comp = TemperatureCompensator(cal, extrapolation_limit=5.0)

        # Temperature way below range
        bias_cold = comp._predict_gyro_bias(-20.0)
        bias_clamped = comp._predict_gyro_bias(-5.0)  # Clamped to -5

        # Should be clamped
        assert_allclose(bias_cold, bias_clamped)

    def test_in_calibration_range(self):
        """Test calibration range checking."""
        cal = TemperatureCalibration(
            gyro_bias_coeffs=np.zeros((3, 3)),
            accel_scale_coeffs=np.column_stack([np.ones(3), np.zeros(3)]),
            calibration_temp_range=(0.0, 40.0)
        )

        comp = TemperatureCompensator(cal)

        assert comp.is_in_calibration_range(20.0)
        assert comp.is_in_calibration_range(0.0)
        assert comp.is_in_calibration_range(40.0)
        assert not comp.is_in_calibration_range(-5.0)
        assert not comp.is_in_calibration_range(45.0)


class TestDriftReduction:
    """Tests that temperature compensation reduces drift."""

    def test_drift_reduction_simulation(self):
        """Simulate gyro integration with and without temp compensation."""
        np.random.seed(42)

        # Temperature model: bias increases with temperature
        gyro_coeffs = np.array([
            [0.0, 0.005, 0.0],   # 0.005 rad/s per degree
            [0.0, 0.005, 0.0],
            [0.0, 0.005, 0.0]
        ])

        cal = TemperatureCalibration(
            gyro_bias_coeffs=gyro_coeffs,
            accel_scale_coeffs=np.column_stack([np.ones(3), np.zeros(3)]),
            calibration_temp_range=(-10.0, 50.0)
        )

        comp = TemperatureCompensator(cal)

        # Simulate temperature varying from 20 to 30°C over 60 seconds
        dt = 0.01
        duration = 60.0
        num_samples = int(duration / dt)

        angle_uncompensated = 0.0
        angle_compensated = 0.0

        for i in range(num_samples):
            t = i * dt
            temp = 20.0 + 10.0 * (t / duration)  # Linear ramp

            # True gyro rate is zero (stationary)
            # But measured gyro has temperature-dependent bias
            true_bias = 0.005 * temp  # This is what sensor reads
            gyro_raw = np.array([true_bias, 0.0, 0.0])

            # Without compensation
            angle_uncompensated += gyro_raw[0] * dt

            # With compensation
            gyro_comp = comp.compensate_gyro(gyro_raw, temp)
            angle_compensated += gyro_comp[0] * dt

        # Compensated angle should be much smaller
        assert abs(angle_compensated) < abs(angle_uncompensated) * 0.2


class TestCreateDefaultCalibration:
    """Tests for default calibration factory."""

    def test_default_calibration_reasonable(self):
        """Test that default calibration has reasonable values."""
        cal = create_default_calibration()

        # Temperature range should be reasonable
        t_min, t_max = cal.calibration_temp_range
        assert t_min < 0
        assert t_max > 30

        # Gyro temperature coefficient should be small
        gyro_temp_coeff = np.abs(cal.gyro_bias_coeffs[:, 1])
        assert np.all(gyro_temp_coeff < 0.01)  # < 0.01 rad/s per degree

        # Accel scale should be near 1.0 at reference temp
        comp = TemperatureCompensator(cal)
        scale = comp._predict_accel_scale(cal.reference_temp)
        assert_allclose(scale, np.ones(3), atol=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
