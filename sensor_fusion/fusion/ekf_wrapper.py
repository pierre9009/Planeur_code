"""EKF wrapper with health monitoring and automatic recovery.

This module wraps the AHRS EKF filter with additional
functionality for divergence detection, automatic
re-initialization, and health metrics.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from ahrs.filters import EKF
from ahrs.common.orientation import q2euler

from ..core.types import Quaternion, EulerAngles, FusionState, ImuReading
from ..core.config import Config
from ..core.quaternion import QuaternionOps
from ..core.validation import QuaternionValidator
from .magnetometer import MagnetometerProcessor

logger = logging.getLogger(__name__)


@dataclass
class EkfHealth:
    """EKF health metrics."""
    quaternion_norm: float
    is_diverged: bool
    reinit_count: int
    consecutive_good_updates: int
    mag_trust: float


class EkfWrapper:
    """Extended Kalman Filter wrapper with health monitoring.

    Provides a robust interface to the AHRS EKF with:
    - Automatic divergence detection
    - Quaternion normalization
    - Integration with magnetometer disturbance handling
    - Automatic re-initialization on failure
    """

    def __init__(self, config: Config):
        """Initialize EKF wrapper.

        Args:
            config: System configuration.
        """
        self._config = config
        self._ekf: Optional[EKF] = None
        self._quaternion = Quaternion.identity()
        self._euler = EulerAngles(roll=0.0, pitch=0.0, yaw=0.0)

        self._mag_processor = MagnetometerProcessor(config)
        self._quat_validator = QuaternionValidator(config)

        self._iteration = 0
        self._reinit_count = 0
        self._consecutive_good = 0
        self._is_initialized = False
        self._last_timestamp = 0.0

    def initialize(
        self,
        acc_samples: NDArray[np.float64],
        mag_samples: NDArray[np.float64],
    ) -> Quaternion:
        """Initialize EKF from sensor samples.

        Should be called with averaged sensor data collected
        while stationary.

        Args:
            acc_samples: Accelerometer samples (N x 3) in m/s^2.
            mag_samples: Magnetometer samples (N x 3) in uT.

        Returns:
            Initial orientation quaternion.

        Raises:
            ValueError: If insufficient samples provided.
        """
        min_samples = self._config.ekf.initialization.min_samples

        if len(acc_samples) < min_samples or len(mag_samples) < min_samples:
            raise ValueError(
                f"Insufficient samples: need {min_samples}, "
                f"got acc={len(acc_samples)}, mag={len(mag_samples)}"
            )

        acc_mean = np.mean(acc_samples, axis=0)
        mag_mean = np.mean(mag_samples, axis=0)

        acc_magnitude = np.linalg.norm(acc_mean)
        expected_g = self._config.sensor.accelerometer.gravity_nominal
        if abs(acc_magnitude - expected_g) > 1.0:
            logger.warning(
                "Acceleration magnitude (%.2f m/s^2) differs from expected (%.2f)",
                acc_magnitude, expected_g
            )

        initial_tilt = np.rad2deg(np.arccos(
            np.clip(acc_mean[2] / acc_magnitude, -1.0, 1.0)
        ))
        max_tilt = self._config.ekf.initialization.max_tilt_deg
        if initial_tilt > max_tilt:
            logger.warning(
                "Initial tilt (%.1f deg) exceeds threshold (%.1f deg)",
                initial_tilt, max_tilt
            )

        q0 = QuaternionOps.from_acc_mag(
            acc_mean, mag_mean, frame=self._config.ekf.frame
        )

        self._mag_processor.initialize(mag_samples)

        self._ekf = EKF(
            gyr=np.zeros((1, 3)),
            acc=acc_mean.reshape((1, 3)),
            mag=mag_mean.reshape((1, 3)),
            frequency=self._config.sensor.sample_rate_hz,
            q0=q0.to_array(),
            frame=self._config.ekf.frame,
        )

        self._quaternion = q0
        self._euler = QuaternionOps.to_euler(q0)
        self._is_initialized = True
        self._iteration = 0

        logger.info(
            "EKF initialized: roll=%.1f, pitch=%.1f, yaw=%.1f deg",
            self._euler.roll_deg, self._euler.pitch_deg, self._euler.yaw_deg
        )

        return q0

    def update(
        self,
        reading: ImuReading,
        dt: float,
    ) -> Tuple[FusionState, bool]:
        """Process IMU reading and update orientation estimate.

        Args:
            reading: Current IMU measurement.
            dt: Time since last update in seconds.

        Returns:
            Tuple of (FusionState, bool indicating if update was valid).

        Raises:
            RuntimeError: If EKF not initialized.
        """
        if not self._is_initialized or self._ekf is None:
            raise RuntimeError("EKF not initialized")

        mag_trust = self._mag_processor.process(reading.mag)

        q_array = self._ekf.update(
            q=self._quaternion.to_array(),
            gyr=reading.gyr,
            acc=reading.acc,
            mag=reading.mag,
            dt=dt,
        )

        new_q = Quaternion.from_array(q_array)
        validation = self._quat_validator.validate(new_q)

        if not validation.is_valid:
            logger.error("Quaternion validation failed: %s", validation.errors)
            self._consecutive_good = 0
            return self._create_state(dt, mag_trust), False

        new_q = new_q.normalized()
        self._quaternion = new_q
        self._euler = QuaternionOps.to_euler(new_q)

        self._iteration += 1
        self._consecutive_good += 1
        self._last_timestamp = reading.timestamp

        return self._create_state(dt, mag_trust), True

    def _create_state(self, dt: float, mag_trust: float) -> FusionState:
        """Create current fusion state snapshot.

        Args:
            dt: Time step.
            mag_trust: Current magnetometer trust.

        Returns:
            FusionState with current values.
        """
        return FusionState(
            quaternion=self._quaternion,
            euler=self._euler,
            timestamp=self._last_timestamp,
            dt=dt,
            iteration=self._iteration,
            is_initialized=self._is_initialized,
            mag_trust=mag_trust,
            quaternion_norm=self._quaternion.norm,
        )

    def needs_reinitialization(self) -> bool:
        """Check if EKF needs to be reinitialized.

        Returns:
            True if quaternion has diverged.
        """
        return self._quat_validator.needs_reinitialization(self._quaternion)

    def reinitialize(
        self,
        acc_samples: NDArray[np.float64],
        mag_samples: NDArray[np.float64],
    ) -> Quaternion:
        """Reinitialize EKF after divergence.

        Args:
            acc_samples: Fresh accelerometer samples.
            mag_samples: Fresh magnetometer samples.

        Returns:
            New initial quaternion.
        """
        self._reinit_count += 1
        logger.warning("EKF reinitialization #%d", self._reinit_count)
        return self.initialize(acc_samples, mag_samples)

    @property
    def health(self) -> EkfHealth:
        """Get EKF health metrics."""
        return EkfHealth(
            quaternion_norm=self._quaternion.norm,
            is_diverged=self.needs_reinitialization(),
            reinit_count=self._reinit_count,
            consecutive_good_updates=self._consecutive_good,
            mag_trust=self._mag_processor.trust,
        )

    @property
    def quaternion(self) -> Quaternion:
        """Current orientation quaternion."""
        return self._quaternion

    @property
    def euler(self) -> EulerAngles:
        """Current Euler angles."""
        return self._euler

    @property
    def is_initialized(self) -> bool:
        """Whether EKF has been initialized."""
        return self._is_initialized

    @property
    def mag_processor(self) -> MagnetometerProcessor:
        """Access to magnetometer processor."""
        return self._mag_processor
