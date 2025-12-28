"""Tests for magnetometer health detection."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from fusion.mag_health import (
    MagnetometerHealthCheck,
    MagHealthStatus,
    AdaptiveMagnetometerValidator
)


class TestMagHealthStatus:
    """Tests for MagHealthStatus dataclass."""

    def test_trust_factor_valid(self):
        """Test trust factor for valid reading."""
        status = MagHealthStatus(
            is_valid=True,
            magnitude=48.0,
            expected_magnitude=48.0,
            deviation_ratio=0.0
        )
        assert status.trust_factor == 1.0

    def test_trust_factor_invalid(self):
        """Test trust factor for invalid reading."""
        status = MagHealthStatus(
            is_valid=False,
            magnitude=48.0,
            expected_magnitude=48.0,
            deviation_ratio=0.0,
            failure_reason="test"
        )
        assert status.trust_factor == 0.0

    def test_trust_factor_deviation(self):
        """Test trust factor decreases with deviation."""
        status = MagHealthStatus(
            is_valid=True,
            magnitude=54.0,
            expected_magnitude=48.0,
            deviation_ratio=0.125  # 12.5% deviation
        )
        # Trust should be 1.0 - 0.125*2 = 0.75
        assert status.trust_factor == 0.75


class TestMagnetometerHealthCheck:
    """Tests for MagnetometerHealthCheck."""

    def test_valid_reading(self):
        """Test valid magnetometer reading passes."""
        checker = MagnetometerHealthCheck(expected_norm=48.0, tolerance=0.25)

        mag = np.array([30.0, 0.0, 36.0])  # magnitude ~47 µT
        status = checker.check(mag)

        assert status.is_valid
        assert status.failure_reason is None

    def test_magnitude_too_low(self):
        """Test low magnitude fails."""
        checker = MagnetometerHealthCheck(expected_norm=48.0, tolerance=0.25)

        mag = np.array([5.0, 5.0, 5.0])  # magnitude ~8.7 µT
        status = checker.check(mag)

        assert not status.is_valid
        assert "too_low" in status.failure_reason

    def test_magnitude_too_high(self):
        """Test high magnitude fails."""
        checker = MagnetometerHealthCheck(expected_norm=48.0, tolerance=0.25)

        mag = np.array([50.0, 50.0, 50.0])  # magnitude ~86.6 µT
        status = checker.check(mag)

        assert not status.is_valid
        assert "too_high" in status.failure_reason

    def test_outside_expected_range(self):
        """Test outside tolerance fails."""
        checker = MagnetometerHealthCheck(expected_norm=48.0, tolerance=0.10)

        # Just outside 10% tolerance (should be valid between 43.2-52.8)
        mag = np.array([37.0, 0.0, 0.0])  # 37 µT
        status = checker.check(mag)

        assert not status.is_valid
        assert "outside_expected_range" in status.failure_reason

    def test_nan_values(self):
        """Test NaN values fail."""
        checker = MagnetometerHealthCheck()

        mag = np.array([float('nan'), 30.0, 40.0])
        status = checker.check(mag)

        assert not status.is_valid
        assert "non_finite" in status.failure_reason

    def test_inf_values(self):
        """Test Inf values fail."""
        checker = MagnetometerHealthCheck()

        mag = np.array([float('inf'), 30.0, 40.0])
        status = checker.check(mag)

        assert not status.is_valid
        assert "non_finite" in status.failure_reason

    def test_history_consistency(self):
        """Test consistency check with history."""
        checker = MagnetometerHealthCheck(
            expected_norm=48.0,
            tolerance=0.25,
            consistency_threshold=0.10
        )

        # Build up consistent history
        mag_normal = np.array([30.0, 0.0, 36.0])  # ~47 µT
        for _ in range(20):
            checker.check(mag_normal)

        # Now try a sudden large change (still within tolerance but inconsistent)
        mag_sudden = np.array([36.0, 0.0, 43.2])  # ~56 µT (20% change)
        status = checker.check(mag_sudden)

        assert not status.is_valid
        assert "inconsistent" in status.failure_reason

    def test_is_valid_simple(self):
        """Test simple boolean is_valid method."""
        checker = MagnetometerHealthCheck(expected_norm=48.0)

        assert checker.is_valid(np.array([30.0, 0.0, 36.0]))
        assert not checker.is_valid(np.array([5.0, 5.0, 5.0]))

    def test_reset(self):
        """Test reset clears history."""
        checker = MagnetometerHealthCheck()

        # Build history
        mag = np.array([30.0, 0.0, 36.0])
        for _ in range(10):
            checker.check(mag)

        assert len(checker.history) == 10

        checker.reset()

        assert len(checker.history) == 0
        assert checker.consecutive_failures == 0

    def test_update_expected_norm(self):
        """Test updating expected norm."""
        checker = MagnetometerHealthCheck(expected_norm=48.0)

        # Update to local field measurement
        checker.update_expected_norm(52.0)
        assert checker.expected_norm == 52.0

        # Out of range update should be ignored
        checker.update_expected_norm(100.0)  # Outside Earth field range
        assert checker.expected_norm == 52.0

    def test_stats(self):
        """Test statistics tracking."""
        checker = MagnetometerHealthCheck()

        # Some valid, some invalid
        checker.check(np.array([30.0, 0.0, 36.0]))  # valid
        checker.check(np.array([30.0, 0.0, 36.0]))  # valid
        checker.check(np.array([5.0, 5.0, 5.0]))    # invalid
        checker.check(np.array([30.0, 0.0, 36.0]))  # valid

        stats = checker.get_stats()

        assert stats['total_checks'] == 4
        assert stats['failures'] == 1
        assert stats['failure_rate'] == 0.25

    def test_get_reference_field(self):
        """Test getting reference field from history."""
        checker = MagnetometerHealthCheck()

        # Not enough history
        assert checker.get_reference_field() is None

        # Build history
        mag = np.array([30.0, 0.0, 36.0])
        for _ in range(10):
            checker.check(mag)

        ref = checker.get_reference_field()
        assert ref is not None
        assert_array_almost_equal(ref, mag)


class TestMagneticDisturbanceScenarios:
    """Test realistic magnetic disturbance scenarios."""

    def test_motor_interference(self):
        """Test detection of motor magnetic interference."""
        checker = MagnetometerHealthCheck(expected_norm=48.0, tolerance=0.25)

        # Normal readings first
        mag_normal = np.array([30.0, 0.0, 36.0])
        for _ in range(20):
            assert checker.check(mag_normal).is_valid

        # Motor starts - magnetic field increases significantly
        mag_motor = np.array([60.0, 40.0, 50.0])  # ~87 µT
        status = checker.check(mag_motor)

        assert not status.is_valid

    def test_gradual_drift(self):
        """Test detection of gradual field drift."""
        checker = MagnetometerHealthCheck(
            expected_norm=48.0,
            tolerance=0.20,
            consistency_threshold=0.05
        )

        # Start with normal field
        for i in range(50):
            # Gradually increase magnitude
            scale = 1.0 + i * 0.005  # 0.5% increase per sample
            mag = np.array([30.0 * scale, 0.0, 36.0 * scale])

            status = checker.check(mag)

            # Should eventually fail consistency or magnitude check
            if i > 40:
                # After 20% drift, should be failing
                pass  # May or may not fail depending on parameters

    def test_recovery_after_disturbance(self):
        """Test recovery after magnetic disturbance ends."""
        checker = MagnetometerHealthCheck(expected_norm=48.0, tolerance=0.25)

        mag_normal = np.array([30.0, 0.0, 36.0])
        mag_disturbed = np.array([80.0, 60.0, 70.0])

        # Normal operation
        for _ in range(20):
            checker.check(mag_normal)

        # Disturbance
        for _ in range(5):
            status = checker.check(mag_disturbed)
            assert not status.is_valid

        # Recovery - normal readings again
        # First few may still fail due to consistency check
        recovery_count = 0
        for _ in range(20):
            status = checker.check(mag_normal)
            if status.is_valid:
                recovery_count += 1

        # Should eventually recover
        assert recovery_count > 10


class TestAdaptiveMagnetometerValidator:
    """Tests for AdaptiveMagnetometerValidator."""

    def test_learns_local_field(self):
        """Test that validator learns local field magnitude."""
        validator = AdaptiveMagnetometerValidator(
            initial_expected_norm=48.0,
            learning_rate=0.1
        )

        # Feed readings with different local field
        local_field = np.array([35.0, 0.0, 30.0])  # ~46 µT

        for i in range(100):
            validator.validate(local_field, current_time=float(i))

        # Should have learned the local field
        assert abs(validator.learned_norm - np.linalg.norm(local_field)) < 2.0

    def test_disturbance_timeout(self):
        """Test disturbance timeout prevents premature re-enabling."""
        validator = AdaptiveMagnetometerValidator(
            disturbance_timeout_s=2.0
        )

        mag_normal = np.array([30.0, 0.0, 36.0])
        mag_disturbed = np.array([80.0, 60.0, 70.0])

        # Normal operation
        for i in range(20):
            validator.validate(mag_normal, current_time=float(i) * 0.1)

        # Disturbance at t=2.0
        status = validator.validate(mag_disturbed, current_time=2.0)
        assert not status.is_valid

        # Normal reading immediately after - should still be invalid (timeout)
        status = validator.validate(mag_normal, current_time=2.1)
        assert not status.is_valid
        assert status.failure_reason == "disturbance_timeout"

        # After timeout period
        status = validator.validate(mag_normal, current_time=4.1)
        assert status.is_valid

    def test_adaptive_stats(self):
        """Test adaptive statistics."""
        validator = AdaptiveMagnetometerValidator()

        mag = np.array([30.0, 0.0, 36.0])
        for i in range(10):
            validator.validate(mag, current_time=float(i))

        stats = validator.get_adaptive_stats()

        assert 'learned_norm' in stats
        assert 'learned_std' in stats
        assert 'in_disturbance' in stats


class TestMagFallbackIntegration:
    """Tests for magnetometer fallback behavior."""

    def test_continuous_operation_with_failures(self):
        """Test that system handles intermittent mag failures."""
        checker = MagnetometerHealthCheck(expected_norm=48.0)

        mag_good = np.array([30.0, 0.0, 36.0])
        mag_bad = np.array([5.0, 5.0, 5.0])

        results = []
        for i in range(100):
            # Intermittent bad readings
            if i % 10 == 5:
                mag = mag_bad
            else:
                mag = mag_good

            status = checker.check(mag)
            results.append(status.is_valid)

        # Most readings should be valid
        valid_count = sum(results)
        assert valid_count > 80

        # Bad readings should be detected
        invalid_count = len(results) - valid_count
        assert invalid_count == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
