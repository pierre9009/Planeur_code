"""Magnetometer health detection and validation.

Detects when magnetometer readings are unreliable due to:
- Magnetic interference (motors, ferrous materials)
- Sensor saturation
- Hardware faults

Provides automatic fallback signaling for fusion algorithms.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Optional
from collections import deque


@dataclass
class MagHealthStatus:
    """Current magnetometer health status."""
    is_valid: bool
    magnitude: float
    expected_magnitude: float
    deviation_ratio: float
    failure_reason: Optional[str] = None

    @property
    def trust_factor(self) -> float:
        """Get trust factor (0.0 = don't trust, 1.0 = full trust)."""
        if not self.is_valid:
            return 0.0
        # Linear falloff as deviation increases
        return max(0.0, 1.0 - self.deviation_ratio * 2.0)


class MagnetometerHealthCheck:
    """Detect when magnetometer is unreliable.

    Uses multiple criteria:
    1. Magnitude within Earth field range (25-65 µT typically)
    2. Consistency with recent history
    3. Rate of change limits
    """

    # Earth's magnetic field range (µT)
    EARTH_FIELD_MIN = 25.0
    EARTH_FIELD_MAX = 65.0
    EARTH_FIELD_TYPICAL = 48.0

    def __init__(
        self,
        expected_norm: float = 48.0,
        tolerance: float = 0.25,
        history_size: int = 50,
        consistency_threshold: float = 0.15
    ):
        """Initialize health checker.

        Args:
            expected_norm: Expected Earth field magnitude in µT.
            tolerance: Allowed deviation from expected (0.25 = ±25%).
            history_size: Number of samples to keep in history.
            consistency_threshold: Max sudden change allowed (fraction).
        """
        self.expected_norm = expected_norm
        self.tolerance = tolerance
        self.history_size = history_size
        self.consistency_threshold = consistency_threshold

        # Circular buffer for history
        self.history: deque = deque(maxlen=history_size)
        self.magnitude_history: deque = deque(maxlen=history_size)

        # Running statistics
        self.total_checks = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 0

        # Reference magnitude (updated adaptively)
        self.reference_magnitude = expected_norm

        # Last valid reading
        self.last_valid: Optional[NDArray[np.float64]] = None

    def check(self, mag: NDArray[np.float64]) -> MagHealthStatus:
        """Check if magnetometer reading is valid.

        Args:
            mag: Magnetometer reading [mx,my,mz] in µT.

        Returns:
            MagHealthStatus with validation results.
        """
        self.total_checks += 1

        mag_norm = float(np.linalg.norm(mag))

        # Check for NaN/Inf
        if not np.isfinite(mag).all():
            return self._fail(mag_norm, "non_finite_values")

        # Check absolute range (Earth field bounds)
        if mag_norm < self.EARTH_FIELD_MIN:
            return self._fail(mag_norm, "magnitude_too_low")

        if mag_norm > self.EARTH_FIELD_MAX:
            return self._fail(mag_norm, "magnitude_too_high")

        # Check against expected magnitude
        min_norm = self.expected_norm * (1 - self.tolerance)
        max_norm = self.expected_norm * (1 + self.tolerance)

        if not (min_norm < mag_norm < max_norm):
            return self._fail(mag_norm, "outside_expected_range")

        # Check consistency with recent history
        if len(self.magnitude_history) >= 5:
            recent_mean = np.mean(list(self.magnitude_history)[-10:])

            if abs(mag_norm - recent_mean) / recent_mean > self.consistency_threshold:
                return self._fail(mag_norm, "inconsistent_with_history")

        # Check rate of change if we have previous reading
        if self.last_valid is not None:
            delta = np.linalg.norm(mag - self.last_valid)
            # Allow up to 20% change per sample
            if delta / mag_norm > 0.20:
                return self._fail(mag_norm, "excessive_rate_of_change")

        # All checks passed
        return self._pass(mag, mag_norm)

    def is_valid(self, mag: NDArray[np.float64]) -> bool:
        """Simple boolean check if magnetometer is valid.

        Args:
            mag: Magnetometer reading [mx,my,mz] in µT.

        Returns:
            True if magnetometer can be trusted.
        """
        return self.check(mag).is_valid

    def _pass(
        self,
        mag: NDArray[np.float64],
        mag_norm: float
    ) -> MagHealthStatus:
        """Record successful validation."""
        # Update history
        self.history.append(mag.copy())
        self.magnitude_history.append(mag_norm)

        # Update reference (slow adaptation)
        alpha = 0.01
        self.reference_magnitude = (
            (1 - alpha) * self.reference_magnitude + alpha * mag_norm
        )

        # Update last valid
        self.last_valid = mag.copy()

        # Reset consecutive failures
        self.consecutive_failures = 0

        deviation = abs(mag_norm - self.expected_norm) / self.expected_norm

        return MagHealthStatus(
            is_valid=True,
            magnitude=mag_norm,
            expected_magnitude=self.expected_norm,
            deviation_ratio=deviation
        )

    def _fail(self, mag_norm: float, reason: str) -> MagHealthStatus:
        """Record failed validation."""
        self.failures += 1
        self.consecutive_failures += 1
        self.max_consecutive_failures = max(
            self.max_consecutive_failures,
            self.consecutive_failures
        )

        deviation = abs(mag_norm - self.expected_norm) / self.expected_norm

        return MagHealthStatus(
            is_valid=False,
            magnitude=mag_norm,
            expected_magnitude=self.expected_norm,
            deviation_ratio=deviation,
            failure_reason=reason
        )

    def reset(self) -> None:
        """Reset health checker state."""
        self.history.clear()
        self.magnitude_history.clear()
        self.consecutive_failures = 0
        self.last_valid = None

    def update_expected_norm(self, new_norm: float) -> None:
        """Update expected field magnitude.

        Useful after initialization with measured local field.

        Args:
            new_norm: New expected magnitude in µT.
        """
        if self.EARTH_FIELD_MIN <= new_norm <= self.EARTH_FIELD_MAX:
            self.expected_norm = new_norm

    def get_stats(self) -> dict:
        """Get health check statistics."""
        return {
            'total_checks': self.total_checks,
            'failures': self.failures,
            'failure_rate': self.failures / max(1, self.total_checks),
            'consecutive_failures': self.consecutive_failures,
            'max_consecutive_failures': self.max_consecutive_failures,
            'reference_magnitude': self.reference_magnitude,
            'history_size': len(self.history)
        }

    def get_reference_field(self) -> Optional[NDArray[np.float64]]:
        """Get reference field vector from history.

        Returns:
            Mean of recent valid readings, or None if no history.
        """
        if len(self.history) < 5:
            return None

        return np.mean(list(self.history), axis=0)


class AdaptiveMagnetometerValidator:
    """Extended magnetometer validation with adaptive thresholds.

    Learns the local magnetic environment and adjusts expectations
    based on operating conditions.
    """

    def __init__(
        self,
        initial_expected_norm: float = 48.0,
        learning_rate: float = 0.001,
        disturbance_timeout_s: float = 5.0
    ):
        """Initialize adaptive validator.

        Args:
            initial_expected_norm: Initial expected field magnitude.
            learning_rate: Rate of adaptation (0.001 = slow).
            disturbance_timeout_s: Time to wait after disturbance before
                                   re-enabling magnetometer.
        """
        self.core_checker = MagnetometerHealthCheck(
            expected_norm=initial_expected_norm
        )
        self.learning_rate = learning_rate
        self.disturbance_timeout_s = disturbance_timeout_s

        # Adaptive state
        self.learned_norm = initial_expected_norm
        self.learned_variance = 5.0  # Initial variance estimate

        # Disturbance tracking
        self.disturbance_start_time: Optional[float] = None
        self.in_disturbance = False

    def validate(
        self,
        mag: NDArray[np.float64],
        current_time: float
    ) -> MagHealthStatus:
        """Validate magnetometer with adaptive thresholds.

        Args:
            mag: Magnetometer reading [mx,my,mz] in µT.
            current_time: Current timestamp in seconds.

        Returns:
            Health status with adaptive evaluation.
        """
        status = self.core_checker.check(mag)

        # First, check if we're still in disturbance timeout
        if self.in_disturbance and self.disturbance_start_time is not None:
            time_since_disturbance = current_time - self.disturbance_start_time
            if time_since_disturbance < self.disturbance_timeout_s:
                # Still in timeout, reject even valid readings
                return MagHealthStatus(
                    is_valid=False,
                    magnitude=status.magnitude,
                    expected_magnitude=self.learned_norm,
                    deviation_ratio=status.deviation_ratio,
                    failure_reason="disturbance_timeout"
                )
            else:
                # Timeout expired, can clear disturbance state
                self.in_disturbance = False
                self.disturbance_start_time = None

        if status.is_valid:
            # Update learned parameters
            mag_norm = status.magnitude
            self.learned_norm += self.learning_rate * (
                mag_norm - self.learned_norm
            )

            # Update variance estimate
            error = mag_norm - self.learned_norm
            self.learned_variance += self.learning_rate * (
                error**2 - self.learned_variance
            )

            # Update core checker's expected norm slowly
            if self.core_checker.total_checks % 100 == 0:
                self.core_checker.update_expected_norm(self.learned_norm)

        else:
            # Track disturbance start
            if not self.in_disturbance:
                self.in_disturbance = True
                self.disturbance_start_time = current_time

        return status

    def get_adaptive_stats(self) -> dict:
        """Get adaptive validation statistics."""
        base_stats = self.core_checker.get_stats()
        base_stats.update({
            'learned_norm': self.learned_norm,
            'learned_std': np.sqrt(self.learned_variance),
            'in_disturbance': self.in_disturbance
        })
        return base_stats
