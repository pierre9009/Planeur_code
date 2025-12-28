"""Integration tests for end-to-end sensor fusion."""

import pytest
import time
import numpy as np

from core.config import Config, load_config
from core.types import ImuReading
from core.validation import SensorValidator
from communication.uart import MockImuUart
from fusion import EkfWrapper, FusionInitializer
from monitoring import PerformanceMonitor


class TestEndToEndFusion:
    """End-to-end integration tests for sensor fusion pipeline."""

    def test_complete_fusion_loop(self, config):
        """Complete fusion loop should run without errors."""
        with MockImuUart(config) as imu:
            validator = SensorValidator(config)
            ekf = EkfWrapper(config)
            initializer = FusionInitializer(config)
            monitor = PerformanceMonitor(config)

            init_result = initializer.collect_samples(imu)
            assert init_result.success

            ekf.initialize(init_result.acc_samples, init_result.mag_samples)
            assert ekf.is_initialized

            last_timestamp = None
            valid_updates = 0

            for _ in range(100):
                monitor.start_iteration()

                reading = imu.read_measurement()
                assert reading is not None

                validation = validator.validate_reading(reading)
                assert validation.is_valid

                if last_timestamp is None:
                    last_timestamp = reading.timestamp
                    continue

                dt = reading.timestamp - last_timestamp
                last_timestamp = reading.timestamp

                state, valid = ekf.update(reading, dt)
                if valid:
                    valid_updates += 1

                monitor.end_iteration(reading.timestamp)

            assert valid_updates > 90
            assert abs(ekf.quaternion.norm - 1.0) < 0.01

            stats = monitor.get_stats()
            assert stats.effective_rate_hz > 0
            assert stats.total_iterations > 0

    def test_stationary_orientation_stability(self, config):
        """Orientation should remain stable for stationary input."""
        with MockImuUart(config) as imu:
            ekf = EkfWrapper(config)
            initializer = FusionInitializer(config)

            init_result = initializer.collect_samples(imu)
            ekf.initialize(init_result.acc_samples, init_result.mag_samples)

            initial_euler = ekf.euler
            last_timestamp = None

            for i in range(500):
                reading = imu.read_measurement()

                if last_timestamp is None:
                    last_timestamp = reading.timestamp
                    continue

                dt = reading.timestamp - last_timestamp
                last_timestamp = reading.timestamp

                ekf.update(reading, dt)

            final_euler = ekf.euler

            roll_drift = abs(final_euler.roll_deg - initial_euler.roll_deg)
            pitch_drift = abs(final_euler.pitch_deg - initial_euler.pitch_deg)

            assert roll_drift < 2.0, f"Roll drift: {roll_drift:.2f} deg"
            assert pitch_drift < 2.0, f"Pitch drift: {pitch_drift:.2f} deg"

    def test_quaternion_remains_valid(self, config):
        """Quaternion should remain valid throughout operation."""
        with MockImuUart(config) as imu:
            ekf = EkfWrapper(config)
            initializer = FusionInitializer(config)

            init_result = initializer.collect_samples(imu)
            ekf.initialize(init_result.acc_samples, init_result.mag_samples)

            last_timestamp = None
            max_norm_deviation = 0.0

            for _ in range(200):
                reading = imu.read_measurement()

                if last_timestamp is None:
                    last_timestamp = reading.timestamp
                    continue

                dt = reading.timestamp - last_timestamp
                last_timestamp = reading.timestamp

                ekf.update(reading, dt)

                norm_deviation = abs(ekf.quaternion.norm - 1.0)
                max_norm_deviation = max(max_norm_deviation, norm_deviation)

                assert ekf.quaternion.is_valid(), \
                    f"Quaternion invalid: norm={ekf.quaternion.norm}"

            assert max_norm_deviation < 0.01, \
                f"Max norm deviation: {max_norm_deviation}"

    def test_magnetometer_disturbance_handling(self, config):
        """Fusion should handle magnetometer disturbances gracefully."""
        with MockImuUart(config) as imu:
            ekf = EkfWrapper(config)
            initializer = FusionInitializer(config)

            init_result = initializer.collect_samples(imu)
            ekf.initialize(init_result.acc_samples, init_result.mag_samples)

            last_timestamp = None

            for i in range(100):
                reading = imu.read_measurement()

                if i >= 40 and i < 60:
                    reading = ImuReading(
                        seq=reading.seq,
                        timestamp=reading.timestamp,
                        ax=reading.ax, ay=reading.ay, az=reading.az,
                        gx=reading.gx, gy=reading.gy, gz=reading.gz,
                        mx=500.0, my=500.0, mz=500.0,
                        temperature=reading.temperature,
                    )

                if last_timestamp is None:
                    last_timestamp = reading.timestamp
                    continue

                dt = reading.timestamp - last_timestamp
                last_timestamp = reading.timestamp

                state, valid = ekf.update(reading, dt)

                if i >= 40 and i < 60:
                    assert ekf.mag_processor.is_disturbed or i < 42
                    assert state.mag_trust < 1.0 or i < 42

            assert ekf.quaternion.is_valid()

    def test_validation_pipeline(self, config):
        """Validation pipeline should catch invalid data."""
        validator = SensorValidator(config)

        valid_reading = ImuReading(
            seq=1, timestamp=1000.0,
            ax=0.0, ay=0.0, az=9.81,
            gx=0.0, gy=0.0, gz=0.0,
            mx=20.0, my=5.0, mz=45.0,
            temperature=25.0,
        )

        result = validator.validate_reading(valid_reading)
        assert result.is_valid

        invalid_reading = ImuReading(
            seq=2, timestamp=1000.01,
            ax=float("nan"), ay=0.0, az=9.81,
            gx=0.0, gy=0.0, gz=0.0,
            mx=20.0, my=5.0, mz=45.0,
            temperature=25.0,
        )

        result = validator.validate_reading(invalid_reading)
        assert not result.is_valid

    def test_performance_monitoring(self, config):
        """Performance monitoring should track metrics correctly."""
        monitor = PerformanceMonitor(config)
        base_time = 1000.0

        for i in range(100):
            monitor.start_iteration()
            time.sleep(0.001)
            monitor.end_iteration(base_time + i * 0.01)

        stats = monitor.get_stats()

        assert stats.total_iterations == 100
        assert abs(stats.mean_dt_ms - 10.0) < 1.0
        assert stats.effective_rate_hz > 90

    def test_sensor_stats_accumulation(self, config):
        """Sensor statistics should accumulate correctly."""
        with MockImuUart(config) as imu:
            for _ in range(100):
                imu.read_measurement()

            stats = imu.stats

            assert stats.total_packets == 100
            assert stats.valid_packets == 100
            assert stats.crc_errors == 0
            assert stats.packet_loss_rate == 0.0


class TestConfigurationLoading:
    """Tests for configuration loading."""

    def test_default_config(self):
        """Default config should have valid values."""
        config = Config()

        assert config.uart.baudrate == 115200
        assert config.sensor.sample_rate_hz == 100
        assert config.ekf.frame == "NED"
        assert config.web.port == 5000

    def test_config_dataclass_defaults(self):
        """Config dataclass should have sensible defaults."""
        config = Config()

        assert config.sensor.accelerometer.gravity_nominal == 9.81
        assert config.magnetometer.disturbance.field_magnitude_tolerance == 0.15
        assert config.validation.quaternion.divergence_threshold == 0.1


class TestImuReadingProperties:
    """Tests for ImuReading computed properties."""

    def test_vector_properties(self, sample_imu_reading):
        """Vector properties should return correct arrays."""
        acc = sample_imu_reading.acc
        gyr = sample_imu_reading.gyr
        mag = sample_imu_reading.mag

        assert len(acc) == 3
        assert len(gyr) == 3
        assert len(mag) == 3

        assert acc[2] == 9.81
        assert gyr[0] == 0.0
        assert mag[0] == 20.0

    def test_magnitude_properties(self, sample_imu_reading):
        """Magnitude properties should compute correctly."""
        assert abs(sample_imu_reading.acc_magnitude - 9.81) < 0.01
        assert sample_imu_reading.gyr_magnitude == 0.0
        assert sample_imu_reading.mag_magnitude > 0
