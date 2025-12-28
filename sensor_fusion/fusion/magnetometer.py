"""Magnetometer processing with disturbance detection.

This module implements adaptive magnetometer trust to handle
magnetic disturbances that cause yaw drift in EKF-based
orientation estimation.
"""

import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from core.config import Config

logger = logging.getLogger(__name__)


@dataclass
class MagnetometerState:
    """Current state of magnetometer processing."""
    trust: float
    is_disturbed: bool
    reference_magnitude: float
    current_magnitude: float
    samples_since_disturbance: int


class MagnetometerProcessor:
    """Adaptive magnetometer processing with disturbance detection.

    Monitors magnetic field magnitude to detect disturbances
    (e.g., nearby motors, metal objects) and adjusts trust
    accordingly to prevent yaw drift.
    """

    def __init__(self, config: Config):
        """Initialize magnetometer processor.

        Args:
            config: System configuration with magnetometer settings.
        """
        self._config = config
        self._dist_cfg = config.magnetometer.disturbance
        self._trust_cfg = config.magnetometer.trust

        self._reference_magnitude: Optional[float] = None
        self._current_trust = self._trust_cfg.nominal
        self._is_disturbed = False
        self._samples_since_disturbance = 0
        self._initialized = False

    def initialize(self, mag_samples: NDArray[np.float64]) -> None:
        """Initialize reference magnetic field from samples.

        Should be called during EKF initialization with
        averaged magnetometer readings.

        Args:
            mag_samples: Array of magnetometer readings (N x 3).
        """
        magnitudes = np.linalg.norm(mag_samples, axis=1)
        self._reference_magnitude = float(np.median(magnitudes))
        self._initialized = True
        logger.info(
            "Magnetometer reference initialized: %.2f uT",
            self._reference_magnitude
        )

    def process(self, mag: NDArray[np.float64]) -> float:
        """Process magnetometer reading and return trust factor.

        Compares current field magnitude to reference and
        adjusts trust based on deviation.

        Args:
            mag: Magnetometer reading [mx, my, mz] in uT.

        Returns:
            Trust factor [0, 1] for magnetometer in EKF update.
        """
        if not self._initialized:
            return self._trust_cfg.nominal

        current_magnitude = float(np.linalg.norm(mag))
        self._update_reference(current_magnitude)
        self._detect_disturbance(current_magnitude)
        self._update_trust()

        return self._current_trust

    def _update_reference(self, magnitude: float) -> None:
        """Update reference magnitude with low-pass filter.

        Only updates when not disturbed to prevent contamination
        of reference by disturbances.

        Args:
            magnitude: Current field magnitude.
        """
        if self._is_disturbed or self._reference_magnitude is None:
            return

        alpha = self._dist_cfg.update_rate
        self._reference_magnitude = (
            (1 - alpha) * self._reference_magnitude + alpha * magnitude
        )

    def _detect_disturbance(self, magnitude: float) -> None:
        """Detect magnetic disturbance from magnitude deviation.

        Args:
            magnitude: Current field magnitude.
        """
        if self._reference_magnitude is None:
            return

        deviation = abs(magnitude - self._reference_magnitude) / self._reference_magnitude
        was_disturbed = self._is_disturbed

        if deviation > self._dist_cfg.field_magnitude_tolerance:
            self._is_disturbed = True
            self._samples_since_disturbance = 0
            if not was_disturbed:
                logger.warning(
                    "Magnetic disturbance detected: %.1f uT (ref: %.1f uT, dev: %.1f%%)",
                    magnitude,
                    self._reference_magnitude,
                    deviation * 100
                )
        else:
            self._samples_since_disturbance += 1
            if self._samples_since_disturbance >= self._dist_cfg.recovery_samples:
                if self._is_disturbed:
                    logger.info("Magnetic field recovered")
                self._is_disturbed = False

    def _update_trust(self) -> None:
        """Update trust factor based on disturbance state."""
        if self._is_disturbed:
            self._current_trust = self._trust_cfg.disturbed
        else:
            target = self._trust_cfg.nominal
            alpha = self._trust_cfg.recovery_rate
            self._current_trust = (
                (1 - alpha) * self._current_trust + alpha * target
            )

    @property
    def state(self) -> MagnetometerState:
        """Get current magnetometer processing state."""
        return MagnetometerState(
            trust=self._current_trust,
            is_disturbed=self._is_disturbed,
            reference_magnitude=self._reference_magnitude or 0.0,
            current_magnitude=0.0,
            samples_since_disturbance=self._samples_since_disturbance,
        )

    @property
    def trust(self) -> float:
        """Current magnetometer trust factor."""
        return self._current_trust

    @property
    def is_disturbed(self) -> bool:
        """Whether magnetic disturbance is currently detected."""
        return self._is_disturbed

    def reset(self) -> None:
        """Reset processor state."""
        self._reference_magnitude = None
        self._current_trust = self._trust_cfg.nominal
        self._is_disturbed = False
        self._samples_since_disturbance = 0
        self._initialized = False
