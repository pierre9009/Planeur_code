"""Tests for EKF wrapper and magnetometer processing."""

import pytest
import numpy as np

from core.config import Config
from core.types import ImuReading, Quaternion
from fusion.ekf_wrapper import EkfWrapper
from fusion.magnetometer import MagnetometerProcessor
from fusion.initializer import FusionInitializer


class TestMagnetometerProcessor:
    """Tests for MagnetometerProcessor class."""

    def test_initialization(self, config, mag_samples):
        """Processor should initialize with reference magnitude."""
        processor = MagnetometerProcessor(config)
        processor.initialize(mag_samples)

        state = processor.state
        assert state.reference_magnitude > 0
        assert state.trust == config.magnetometer.trust.nominal
        assert not state.is_disturbed

    def test_nominal_trust(self, config, mag_samples):
        """Normal readings should maintain nominal trust."""
        processor = MagnetometerProcessor(config)
        processor.initialize(mag_samples)

        nominal_mag = np.array([20.0, 5.0, 45.0])
        trust = processor.process(nominal_mag)

        assert trust >= 0.9

    def test_disturbance_detection(self, config, mag_samples):
        """Large field change should trigger disturbance detection."""
        processor = MagnetometerProcessor(config)
        processor.initialize(mag_samples)

        disturbed_mag = np.array([100.0, 100.0, 100.0])
        trust = processor.process(disturbed_mag)

        assert processor.is_disturbed
        assert trust < 0.5

    def test_trust_recovery(self, config, mag_samples):
        """Trust should recover after disturbance ends."""
        processor = MagnetometerProcessor(config)
        processor.initialize(mag_samples)

        processor.process(np.array([100.0, 100.0, 100.0]))
        assert processor.is_disturbed

        nominal_mag = np.array([20.0, 5.0, 45.0])
        for _ in range(60):
            processor.process(nominal_mag)

        assert not processor.is_disturbed
        assert processor.trust > 0.5

    def test_reset(self, config, mag_samples):
        """Reset should clear processor state."""
        processor = MagnetometerProcessor(config)
        processor.initialize(mag_samples)
        processor.process(np.array([100.0, 100.0, 100.0]))

        processor.reset()

        assert processor.state.reference_magnitude == 0.0
        assert not processor.is_disturbed


class TestEkfWrapper:
    """Tests for EkfWrapper class."""

    def test_initialization(self, config, acc_samples, mag_samples):
        """EKF should initialize with samples."""
        ekf = EkfWrapper(config)
        q = ekf.initialize(acc_samples, mag_samples)

        assert ekf.is_initialized
        assert q.is_valid()
        assert abs(ekf.euler.roll_deg) < 10
        assert abs(ekf.euler.pitch_deg) < 10

    def test_initialization_insufficient_samples(self, config):
        """Initialization should fail with insufficient samples."""
        ekf = EkfWrapper(config)
        acc = np.zeros((10, 3))
        mag = np.zeros((10, 3))

        with pytest.raises(ValueError):
            ekf.initialize(acc, mag)

    def test_update(self, config, acc_samples, mag_samples, sample_imu_reading):
        """EKF should update with new reading."""
        ekf = EkfWrapper(config)
        ekf.initialize(acc_samples, mag_samples)

        state, valid = ekf.update(sample_imu_reading, dt=0.01)

        assert valid
        assert state.is_initialized
        assert state.quaternion.is_valid()

    def test_update_not_initialized(self, config, sample_imu_reading):
        """Update should fail if not initialized."""
        ekf = EkfWrapper(config)

        with pytest.raises(RuntimeError):
            ekf.update(sample_imu_reading, dt=0.01)

    def test_health_metrics(self, config, acc_samples, mag_samples, sample_imu_reading):
        """Health metrics should be updated."""
        ekf = EkfWrapper(config)
        ekf.initialize(acc_samples, mag_samples)

        for _ in range(10):
            ekf.update(sample_imu_reading, dt=0.01)

        health = ekf.health
        assert abs(health.quaternion_norm - 1.0) < 0.01
        assert not health.is_diverged
        assert health.reinit_count == 0

    def test_quaternion_normalization(self, config, acc_samples, mag_samples):
        """Quaternion should remain normalized after updates."""
        ekf = EkfWrapper(config)
        ekf.initialize(acc_samples, mag_samples)

        reading = ImuReading(
            seq=1, timestamp=1000.0,
            ax=0.1, ay=0.1, az=9.80,
            gx=0.01, gy=-0.01, gz=0.02,
            mx=20.0, my=5.0, mz=45.0,
            temperature=25.0,
        )

        for i in range(100):
            ekf.update(reading, dt=0.01)

        assert abs(ekf.quaternion.norm - 1.0) < 0.01


class TestFusionInitializer:
    """Tests for FusionInitializer class."""

    def test_check_stationary_stationary(self, config, sample_imu_reading):
        """Stationary readings should be detected."""
        initializer = FusionInitializer(config)
        readings = [sample_imu_reading] * 10

        is_stationary, max_rate = initializer.check_stationary(readings)

        assert is_stationary
        assert max_rate < 1.0

    def test_check_stationary_moving(self, config):
        """Moving readings should be detected."""
        initializer = FusionInitializer(config)
        readings = [
            ImuReading(
                seq=i, timestamp=1000.0 + i * 0.01,
                ax=0.0, ay=0.0, az=9.81,
                gx=1.0, gy=0.0, gz=0.0,
                mx=20.0, my=5.0, mz=45.0,
                temperature=25.0,
            )
            for i in range(10)
        ]

        is_stationary, max_rate = initializer.check_stationary(readings)

        assert not is_stationary
        assert max_rate > 10.0

    def test_check_stationary_empty(self, config):
        """Empty readings should return not stationary."""
        initializer = FusionInitializer(config)

        is_stationary, _ = initializer.check_stationary([])

        assert not is_stationary
