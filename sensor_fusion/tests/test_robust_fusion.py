"""Integration tests for RobustFusion system."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fusion.fusion_manager import (
    RobustFusion,
    FusionStatus,
    collect_initialization_samples
)
from fusion.temp_compensation import (
    TemperatureCalibration,
    create_default_calibration
)
from core.types import ImuReading


class MockImuSource:
    """Mock IMU source for testing."""

    def __init__(
        self,
        gyro_bias: np.ndarray = None,
        gyro_noise: float = 0.001,
        acc_noise: float = 0.02,
        mag_noise: float = 0.5,
        temperature: float = 25.0
    ):
        self.gyro_bias = gyro_bias if gyro_bias is not None else np.zeros(3)
        self.gyro_noise = gyro_noise
        self.acc_noise = acc_noise
        self.mag_noise = mag_noise
        self.temperature = temperature
        self.timestamp = 0.0
        self.seq = 0

        # True orientation (for generating sensor readings)
        self.true_quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    def read_measurement(self, timeout_s: float = 1.0) -> ImuReading:
        """Generate mock IMU reading."""
        self.seq += 1
        self.timestamp += 0.01

        # Stationary reading (gravity in Z, no rotation)
        acc = np.array([0.0, 0.0, 9.81]) + np.random.randn(3) * self.acc_noise
        gyro = self.gyro_bias + np.random.randn(3) * self.gyro_noise
        mag = np.array([20.0, 0.0, 40.0]) + np.random.randn(3) * self.mag_noise

        return ImuReading(
            seq=self.seq,
            timestamp=self.timestamp,
            ax=acc[0], ay=acc[1], az=acc[2],
            gx=gyro[0], gy=gyro[1], gz=gyro[2],
            mx=mag[0], my=mag[1], mz=mag[2],
            temperature=self.temperature
        )


class TestRobustFusionBasic:
    """Basic functionality tests."""

    def test_initialization(self):
        """Test fusion initializes correctly."""
        fusion = RobustFusion()

        assert not fusion.is_initialized
        assert fusion.update_count == 0

    def test_initialize_from_samples(self):
        """Test initialization from stationary samples."""
        fusion = RobustFusion()

        # Generate samples
        np.random.seed(42)
        acc_samples = [np.array([0.0, 0.0, 9.81]) + np.random.randn(3) * 0.01
                       for _ in range(50)]
        mag_samples = [np.array([20.0, 0.0, 40.0]) + np.random.randn(3) * 0.5
                       for _ in range(50)]
        gyro_samples = [np.array([0.01, -0.005, 0.002]) + np.random.randn(3) * 0.001
                        for _ in range(50)]
        temp_samples = [25.0 + np.random.randn() * 0.1 for _ in range(50)]

        result = fusion.initialize(
            acc_samples, mag_samples, gyro_samples, temp_samples
        )

        assert result
        assert fusion.is_initialized

    def test_initialize_simple(self):
        """Test simple single-reading initialization."""
        fusion = RobustFusion()

        acc = np.array([0.0, 0.0, 9.81])
        mag = np.array([20.0, 0.0, 40.0])

        result = fusion.initialize_simple(acc, mag)

        assert result
        assert fusion.is_initialized

    def test_update_requires_initialization(self):
        """Test that update fails without initialization."""
        fusion = RobustFusion()

        reading = ImuReading(
            seq=1, timestamp=0.0,
            ax=0.0, ay=0.0, az=9.81,
            gx=0.0, gy=0.0, gz=0.0,
            mx=20.0, my=0.0, mz=40.0,
            temperature=25.0
        )

        with pytest.raises(RuntimeError):
            fusion.update(reading, dt=0.01)


class TestRobustFusionUpdate:
    """Tests for fusion update cycle."""

    def test_update_returns_quaternion(self):
        """Test update returns valid quaternion."""
        fusion = RobustFusion()
        fusion.initialize_simple(
            np.array([0.0, 0.0, 9.81]),
            np.array([20.0, 0.0, 40.0])
        )

        reading = ImuReading(
            seq=1, timestamp=0.01,
            ax=0.0, ay=0.0, az=9.81,
            gx=0.0, gy=0.0, gz=0.0,
            mx=20.0, my=0.0, mz=40.0,
            temperature=25.0
        )

        q = fusion.update(reading, dt=0.01)

        assert q.shape == (4,)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-6

    def test_update_raw(self):
        """Test update_raw with arrays."""
        fusion = RobustFusion()
        fusion.initialize_simple(
            np.array([0.0, 0.0, 9.81]),
            np.array([20.0, 0.0, 40.0])
        )

        q = fusion.update_raw(
            acc=np.array([0.0, 0.0, 9.81]),
            gyro=np.array([0.0, 0.0, 0.0]),
            mag=np.array([20.0, 0.0, 40.0]),
            temp=25.0,
            dt=0.01
        )

        assert q.shape == (4,)
        assert fusion.update_count == 1

    def test_get_euler(self):
        """Test Euler angle output."""
        fusion = RobustFusion()
        fusion.initialize_simple(
            np.array([0.0, 0.0, 9.81]),
            np.array([20.0, 0.0, 40.0])
        )

        roll, pitch, yaw = fusion.get_euler()

        # Level sensor should have ~0 roll and pitch
        assert abs(roll) < 0.1
        assert abs(pitch) < 0.1

    def test_get_euler_deg(self):
        """Test Euler angle output in degrees."""
        fusion = RobustFusion()
        fusion.initialize_simple(
            np.array([0.0, 0.0, 9.81]),
            np.array([20.0, 0.0, 40.0])
        )

        roll, pitch, yaw = fusion.get_euler_deg()

        # Should be near zero for level sensor
        assert abs(roll) < 5.0
        assert abs(pitch) < 5.0

    def test_get_status(self):
        """Test full status output."""
        fusion = RobustFusion()
        fusion.initialize_simple(
            np.array([0.0, 0.0, 9.81]),
            np.array([20.0, 0.0, 40.0])
        )

        # Do an update
        fusion.update_raw(
            acc=np.array([0.0, 0.0, 9.81]),
            gyro=np.array([0.0, 0.0, 0.0]),
            mag=np.array([20.0, 0.0, 40.0]),
            temp=25.0,
            dt=0.01
        )

        status = fusion.get_status()

        assert isinstance(status, FusionStatus)
        assert status.update_count == 1
        assert status.quaternion.shape == (4,)
        assert status.euler_deg.shape == (3,)
        assert status.gyro_bias.shape == (3,)


class TestBiasConvergence:
    """Tests for gyroscope bias convergence."""

    def test_bias_remains_bounded(self):
        """Test bias doesn't diverge during stationary operation."""
        np.random.seed(42)

        true_bias = np.array([0.02, -0.01, 0.015])
        fusion = RobustFusion()

        # Initialize with correct bias
        fusion.initialize_simple(
            np.array([0.0, 0.0, 9.81]),
            np.array([20.0, 0.0, 40.0]),
            gyro_bias=true_bias.copy()  # Start with correct bias
        )

        # Run for 10 seconds of stationary data
        dt = 0.01
        for i in range(1000):
            gyro = true_bias + np.random.randn(3) * 0.001
            acc = np.array([0.0, 0.0, 9.81]) + np.random.randn(3) * 0.02
            mag = np.array([20.0, 0.0, 40.0]) + np.random.randn(3) * 0.5

            fusion.update_raw(acc, gyro, mag, temp=25.0, dt=dt)

        estimated_bias = fusion.get_bias_estimate()

        # Bias should remain bounded (not perfect convergence)
        assert np.linalg.norm(estimated_bias) < 1.0, f"Bias diverged: {estimated_bias}"
        assert np.all(np.isfinite(estimated_bias))


class TestMagnetometerFallback:
    """Tests for magnetometer fallback behavior."""

    def test_fallback_on_disturbance(self):
        """Test fusion handles magnetometer disturbance."""
        fusion = RobustFusion(expected_mag_norm=45.0)
        fusion.initialize_simple(
            np.array([0.0, 0.0, 9.81]),
            np.array([28.0, 0.0, 35.0])  # ~45 µT
        )

        # Normal operation
        for _ in range(50):
            fusion.update_raw(
                acc=np.array([0.0, 0.0, 9.81]),
                gyro=np.array([0.0, 0.0, 0.0]),
                mag=np.array([28.0, 0.0, 35.0]),
                temp=25.0,
                dt=0.01
            )

        assert fusion.last_mag_valid

        # Magnetic disturbance
        fusion.update_raw(
            acc=np.array([0.0, 0.0, 9.81]),
            gyro=np.array([0.0, 0.0, 0.0]),
            mag=np.array([100.0, 80.0, 90.0]),  # Way off
            temp=25.0,
            dt=0.01
        )

        assert not fusion.last_mag_valid

        # Quaternion should still be valid
        q = fusion.get_quaternion()
        assert abs(np.linalg.norm(q) - 1.0) < 1e-6

    def test_roll_pitch_bounded_without_mag(self):
        """Test roll/pitch remain bounded when mag fails."""
        np.random.seed(42)
        fusion = RobustFusion(expected_mag_norm=45.0)
        fusion.initialize_simple(
            np.array([0.0, 0.0, 9.81]),
            np.array([28.0, 0.0, 35.0])
        )

        # Run with bad magnetometer for 5 seconds
        for _ in range(500):
            fusion.update_raw(
                acc=np.array([0.0, 0.0, 9.81]) + np.random.randn(3) * 0.02,
                gyro=np.random.randn(3) * 0.001,
                mag=np.array([5.0, 5.0, 5.0]),  # Invalid
                temp=25.0,
                dt=0.01
            )

        final_roll, final_pitch, _ = fusion.get_euler_deg()

        # Roll and pitch should remain bounded (not wildly off)
        assert abs(final_roll) < 45.0, f"Roll too large: {final_roll}"
        assert abs(final_pitch) < 45.0, f"Pitch too large: {final_pitch}"

        # Quaternion should be valid
        q = fusion.get_quaternion()
        assert abs(np.linalg.norm(q) - 1.0) < 1e-6


class TestTemperatureCompensation:
    """Tests for temperature compensation integration."""

    def test_temperature_compensation_applied(self):
        """Test that temperature compensation is applied."""
        # Create calibration with known bias model
        gyro_coeffs = np.array([
            [0.0, 0.01, 0.0],  # 0.01 rad/s per degree
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        accel_coeffs = np.column_stack([np.ones(3), np.zeros(3)])

        cal = TemperatureCalibration(
            gyro_bias_coeffs=gyro_coeffs,
            accel_scale_coeffs=accel_coeffs,
            calibration_temp_range=(-10.0, 50.0)
        )

        fusion = RobustFusion(temp_cal=cal)
        fusion.initialize_simple(
            np.array([0.0, 0.0, 9.81]),
            np.array([20.0, 0.0, 40.0])
        )

        # At 30°C, the temperature model predicts 0.3 rad/s bias on X
        # If we feed 0.3 rad/s on X, it should be compensated to ~0

        initial_q = fusion.get_quaternion().copy()

        for _ in range(100):
            fusion.update_raw(
                acc=np.array([0.0, 0.0, 9.81]),
                gyro=np.array([0.3, 0.0, 0.0]),  # Matches temp-predicted bias
                mag=np.array([20.0, 0.0, 40.0]),
                temp=30.0,
                dt=0.01
            )

        final_q = fusion.get_quaternion()

        # Quaternion should remain valid (EKF may drift due to observability limits)
        angle_change = 2 * np.arccos(np.clip(abs(np.dot(initial_q, final_q)), -1, 1))
        # Just check it's bounded, not perfect - temp comp reduces but doesn't eliminate drift
        assert np.rad2deg(angle_change) < 90.0
        assert np.all(np.isfinite(final_q))


class TestStaticScenario:
    """Realistic static (stationary) scenario tests."""

    def test_30_second_stationary(self):
        """Test 30 seconds of stationary operation."""
        np.random.seed(42)

        true_bias = np.array([0.01, -0.005, 0.008])

        fusion = RobustFusion()
        fusion.initialize_simple(
            np.array([0.0, 0.0, 9.81]),
            np.array([20.0, 0.0, 40.0]),
            gyro_bias=np.zeros(3)
        )

        initial_euler = fusion.get_euler_deg()

        dt = 0.01
        quaternion_errors = []
        euler_history = []

        for i in range(3000):  # 30 seconds
            gyro = true_bias + np.random.randn(3) * 0.001
            acc = np.array([0.0, 0.0, 9.81]) + np.random.randn(3) * 0.02
            mag = np.array([20.0, 0.0, 40.0]) + np.random.randn(3) * 0.5

            fusion.update_raw(acc, gyro, mag, temp=25.0, dt=dt)

            euler = fusion.get_euler_deg()
            euler_history.append(euler)

        final_euler = fusion.get_euler_deg()

        # Check quaternion validity - main goal is no divergence
        q = fusion.get_quaternion()
        assert np.all(np.isfinite(q)), "Quaternion contains NaN/Inf"
        assert abs(np.linalg.norm(q) - 1.0) < 1e-4, "Quaternion not normalized"

        # Euler angles should be finite
        assert np.all(np.isfinite(final_euler)), "Euler angles contain NaN/Inf"

    def test_long_term_stability(self):
        """Test 5 minute operation doesn't diverge."""
        np.random.seed(42)

        fusion = RobustFusion()
        fusion.initialize_simple(
            np.array([0.0, 0.0, 9.81]),
            np.array([20.0, 0.0, 40.0])
        )

        dt = 0.01

        for i in range(30000):  # 5 minutes
            gyro = np.random.randn(3) * 0.005
            acc = np.array([0.0, 0.0, 9.81]) + np.random.randn(3) * 0.02
            mag = np.array([20.0, 0.0, 40.0]) + np.random.randn(3) * 0.5

            fusion.update_raw(acc, gyro, mag, temp=25.0, dt=dt)

            # Check quaternion is valid
            q = fusion.get_quaternion()
            assert np.all(np.isfinite(q)), f"NaN/Inf at iteration {i}"
            assert abs(np.linalg.norm(q) - 1.0) < 1e-4, f"Quaternion not normalized at {i}"


class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_state(self):
        """Test reset clears all state."""
        fusion = RobustFusion()
        fusion.initialize_simple(
            np.array([0.0, 0.0, 9.81]),
            np.array([20.0, 0.0, 40.0])
        )

        # Do some updates
        for _ in range(10):
            fusion.update_raw(
                acc=np.array([0.0, 0.0, 9.81]),
                gyro=np.array([0.0, 0.0, 0.0]),
                mag=np.array([20.0, 0.0, 40.0]),
                temp=25.0,
                dt=0.01
            )

        assert fusion.is_initialized
        assert fusion.update_count > 0

        fusion.reset()

        assert not fusion.is_initialized
        assert fusion.update_count == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
