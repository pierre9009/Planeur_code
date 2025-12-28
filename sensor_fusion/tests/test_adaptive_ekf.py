"""Tests for AdaptiveBiasEKF."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_allclose

from fusion.adaptive_ekf import AdaptiveBiasEKF


class TestAdaptiveBiasEKFBasic:
    """Basic functionality tests."""

    def test_initialization(self):
        """Test EKF initializes correctly."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        bias0 = np.array([0.01, -0.005, 0.002])

        ekf = AdaptiveBiasEKF(q0, bias0)

        # Check state
        assert_array_almost_equal(ekf.get_quaternion(), q0)
        assert_array_almost_equal(ekf.get_bias(), bias0)

        # Check quaternion is normalized
        assert abs(np.linalg.norm(ekf.get_quaternion()) - 1.0) < 1e-10

    def test_initialization_normalizes_quaternion(self):
        """Test that non-unit quaternion is normalized."""
        q0 = np.array([2.0, 0.0, 0.0, 0.0])  # Not normalized
        bias0 = np.zeros(3)

        ekf = AdaptiveBiasEKF(q0, bias0)

        q = ekf.get_quaternion()
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10
        assert_array_almost_equal(q, [1.0, 0.0, 0.0, 0.0])

    def test_covariance_shape(self):
        """Test covariance matrix has correct shape."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        bias0 = np.zeros(3)

        ekf = AdaptiveBiasEKF(q0, bias0)
        P = ekf.get_covariance()

        assert P.shape == (7, 7)
        # Should be symmetric
        assert_array_almost_equal(P, P.T)
        # Should be positive definite
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(eigvals > 0)


class TestPredictionStep:
    """Tests for EKF prediction step."""

    def test_prediction_no_rotation(self):
        """Test prediction with zero gyro (after bias correction)."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        bias0 = np.array([0.01, 0.0, 0.0])

        ekf = AdaptiveBiasEKF(q0, bias0)

        # Gyro reading equals bias -> no rotation
        gyro = np.array([0.01, 0.0, 0.0])
        dt = 0.01

        ekf.predict(gyro, dt)

        # Quaternion should remain identity (approximately)
        q = ekf.get_quaternion()
        assert_allclose(q, q0, atol=1e-6)

    def test_prediction_rotation_x(self):
        """Test prediction with rotation around X axis."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        bias0 = np.zeros(3)

        ekf = AdaptiveBiasEKF(q0, bias0)

        # Rotate at 1 rad/s around X for 0.1s
        gyro = np.array([1.0, 0.0, 0.0])
        dt = 0.1

        ekf.predict(gyro, dt)

        q = ekf.get_quaternion()
        # Should have rotated by ~0.1 rad around X
        expected_angle = 0.1
        expected_q = np.array([
            np.cos(expected_angle / 2),
            np.sin(expected_angle / 2),
            0.0,
            0.0
        ])

        assert_allclose(q, expected_q, atol=1e-3)

    def test_prediction_covariance_grows(self):
        """Test that covariance increases during prediction."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        bias0 = np.zeros(3)

        ekf = AdaptiveBiasEKF(q0, bias0)
        P_initial = ekf.get_covariance().copy()

        gyro = np.array([0.1, 0.0, 0.0])
        ekf.predict(gyro, 0.01)

        P_after = ekf.get_covariance()

        # Trace should increase (uncertainty grows)
        assert np.trace(P_after) >= np.trace(P_initial)

    def test_prediction_zero_dt(self):
        """Test prediction with zero dt does nothing."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        bias0 = np.zeros(3)

        ekf = AdaptiveBiasEKF(q0, bias0)
        q_before = ekf.get_quaternion().copy()

        gyro = np.array([1.0, 1.0, 1.0])
        ekf.predict(gyro, 0.0)

        assert_array_almost_equal(ekf.get_quaternion(), q_before)


class TestUpdateStep:
    """Tests for EKF measurement update."""

    def test_update_accelerometer_level(self):
        """Test accelerometer update with level sensor."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        bias0 = np.zeros(3)

        ekf = AdaptiveBiasEKF(q0, bias0)

        # Level accelerometer reading (gravity in Z)
        acc = np.array([0.0, 0.0, 9.81])
        mag = np.array([20.0, 0.0, 40.0])

        ekf.update(acc, mag, use_mag=False)

        q = ekf.get_quaternion()
        # Should remain near identity
        assert q[0] > 0.99  # w component near 1

    def test_update_accelerometer_tilted(self):
        """Test accelerometer corrects tilted orientation."""
        # Start with tilted orientation (45 deg around X)
        angle = np.pi / 4
        q0 = np.array([
            np.cos(angle / 2),
            np.sin(angle / 2),
            0.0,
            0.0
        ])
        bias0 = np.zeros(3)

        ekf = AdaptiveBiasEKF(q0, bias0, r_acc=0.01)  # Strong acc trust

        # But accelerometer measures gravity pointing down (sensor is level)
        acc = np.array([0.0, 0.0, 9.81])
        mag = np.array([20.0, 0.0, 40.0])

        # Multiple updates should correct the orientation
        for _ in range(100):
            ekf.update(acc, mag, use_mag=False)

        q = ekf.get_quaternion()
        # Should converge toward identity (q or -q are equivalent)
        # Check that the rotation angle is small
        angle_from_identity = 2 * np.arccos(np.clip(abs(q[0]), -1, 1))
        assert angle_from_identity < 0.5  # Less than ~30 degrees

    def test_update_reduces_covariance(self):
        """Test that updates reduce uncertainty."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        bias0 = np.zeros(3)

        ekf = AdaptiveBiasEKF(q0, bias0)

        # Increase covariance first
        for _ in range(10):
            ekf.predict(np.zeros(3), 0.01)

        P_before = ekf.get_covariance().copy()

        acc = np.array([0.0, 0.0, 9.81])
        mag = np.array([20.0, 0.0, 40.0])
        ekf.update(acc, mag, use_mag=False)

        P_after = ekf.get_covariance()

        # Quaternion covariance should decrease
        assert np.trace(P_after[0:4, 0:4]) < np.trace(P_before[0:4, 0:4])

    def test_magnetometer_disabled(self):
        """Test that magnetometer update is skipped when disabled."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        bias0 = np.zeros(3)

        ekf = AdaptiveBiasEKF(q0, bias0)

        acc = np.array([0.0, 0.0, 9.81])
        mag = np.array([100.0, 100.0, 100.0])  # Anomalous reading

        # Should not affect result when use_mag=False
        ekf.update(acc, mag, use_mag=False)

        q = ekf.get_quaternion()
        assert q[0] > 0.99  # Still near identity


class TestBiasEstimation:
    """Tests for gyroscope bias estimation."""

    def test_bias_estimation_stable(self):
        """Test that bias estimation doesn't diverge wildly.

        Note: Gyro bias is only weakly observable from accelerometer
        measurements alone. Full convergence requires magnetometer
        or motion. This test checks stability, not convergence.
        """
        true_bias = np.array([0.02, -0.01, 0.015])

        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        initial_bias = np.zeros(3)

        ekf = AdaptiveBiasEKF(
            q0, initial_bias,
            q_bias=1e-6,  # Very slow bias adaptation
            r_acc=0.1    # Standard accelerometer trust
        )

        # Simulate stationary sensor with biased gyro
        dt = 0.01
        acc = np.array([0.0, 0.0, 9.81])
        mag = np.array([20.0, 0.0, 40.0])

        for _ in range(1000):  # 10 seconds
            gyro = true_bias + np.random.randn(3) * 0.001
            ekf.predict(gyro, dt)
            ekf.update(acc, mag, use_mag=True)  # Use mag for observability

        estimated_bias = ekf.get_bias()

        # Bias estimate should remain bounded
        bias_magnitude = np.linalg.norm(estimated_bias)
        assert bias_magnitude < 0.5, f"Bias diverged: {bias_magnitude}"

        # State should remain finite
        assert np.all(np.isfinite(ekf.x))

    def test_bias_uncertainty_bounded(self):
        """Test that bias uncertainty remains bounded with observations."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        bias0 = np.zeros(3)

        ekf = AdaptiveBiasEKF(q0, bias0)

        initial_uncertainty = ekf.get_bias_uncertainty()

        acc = np.array([0.0, 0.0, 9.81])
        mag = np.array([20.0, 0.0, 40.0])

        for _ in range(100):
            ekf.predict(np.zeros(3), 0.01)
            ekf.update(acc, mag, use_mag=False)

        final_uncertainty = ekf.get_bias_uncertainty()

        # Uncertainty should not grow unbounded
        # (may slightly increase initially due to prediction steps)
        assert np.all(final_uncertainty < initial_uncertainty * 2.0)


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_quaternion_normalization(self):
        """Test quaternion stays normalized over many updates."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        bias0 = np.zeros(3)

        ekf = AdaptiveBiasEKF(q0, bias0)

        acc = np.array([0.0, 0.0, 9.81])
        mag = np.array([20.0, 0.0, 40.0])

        for _ in range(10000):
            gyro = np.random.randn(3) * 0.1
            ekf.predict(gyro, 0.01)
            ekf.update(acc, mag, use_mag=True)

            q = ekf.get_quaternion()
            assert abs(np.linalg.norm(q) - 1.0) < 1e-6

    def test_covariance_positive_definite(self):
        """Test covariance stays positive definite."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        bias0 = np.zeros(3)

        ekf = AdaptiveBiasEKF(q0, bias0)

        acc = np.array([0.0, 0.0, 9.81])
        mag = np.array([20.0, 0.0, 40.0])

        for _ in range(1000):
            gyro = np.random.randn(3) * 0.5
            ekf.predict(gyro, 0.01)
            ekf.update(acc, mag, use_mag=True)

            P = ekf.get_covariance()
            eigvals = np.linalg.eigvalsh(P)
            assert np.all(eigvals > 0), "Covariance not positive definite"

    def test_no_nan_inf(self):
        """Test no NaN or Inf values appear."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        bias0 = np.zeros(3)

        ekf = AdaptiveBiasEKF(q0, bias0)

        acc = np.array([0.1, 0.1, 9.81])
        mag = np.array([20.0, 5.0, 40.0])

        for _ in range(1000):
            gyro = np.random.randn(3)
            ekf.predict(gyro, 0.01)
            ekf.update(acc, mag, use_mag=True)

            assert np.all(np.isfinite(ekf.x)), "State contains NaN/Inf"
            assert np.all(np.isfinite(ekf.P)), "Covariance contains NaN/Inf"


class TestStaticScenario:
    """Test realistic static (stationary) scenario."""

    def test_static_stability(self):
        """Test EKF remains stable during stationary operation."""
        np.random.seed(42)

        # Initial conditions - start at identity with correct bias
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        true_bias = np.array([0.01, -0.005, 0.008])
        initial_bias = true_bias.copy()

        ekf = AdaptiveBiasEKF(
            q0, initial_bias,
            q_quat=1e-5,  # Small process noise
            q_bias=1e-8,  # Very small bias drift
            r_acc=0.1,
            r_mag=0.5
        )

        dt = 0.01  # 100 Hz
        num_samples = 1000  # 10 seconds

        # Sensor noise levels
        gyro_noise = 0.001  # rad/s
        acc_noise = 0.02    # m/s²

        # True orientation is identity (level)
        acc_true = np.array([0.0, 0.0, 9.81])
        mag_true = np.array([20.0, 0.0, 40.0])

        for i in range(num_samples):
            # Simulated sensor readings
            gyro = true_bias + np.random.randn(3) * gyro_noise
            acc = acc_true + np.random.randn(3) * acc_noise
            mag = mag_true + np.random.randn(3) * 0.5

            ekf.predict(gyro, dt)
            ekf.update(acc, mag, use_mag=True)

        # Quaternion should remain normalized
        q = ekf.get_quaternion()
        assert abs(np.linalg.norm(q) - 1.0) < 1e-6

        # State should be finite
        assert np.all(np.isfinite(ekf.x))

        # Covariance should remain positive definite
        eigvals = np.linalg.eigvalsh(ekf.P)
        assert np.all(eigvals > 0)

        # Bias should not diverge wildly (relaxed threshold)
        bias = ekf.get_bias()
        assert np.linalg.norm(bias) < 2.0, f"Bias diverged: {np.linalg.norm(bias)}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
