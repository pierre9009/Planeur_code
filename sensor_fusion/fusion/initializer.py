"""Robust EKF initialization from sensor data.

This module handles the collection and validation of initial
sensor samples for EKF initialization.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Protocol
import numpy as np
from numpy.typing import NDArray

from core.types import ImuReading
from core.config import Config

logger = logging.getLogger(__name__)


class ImuSource(Protocol):
    """Protocol for IMU data sources."""

    def read_measurement(self, timeout_s: float) -> ImuReading | None:
        """Read a single IMU measurement."""
        ...


@dataclass
class InitializationResult:
    """Result of initialization sample collection."""
    success: bool
    acc_samples: NDArray[np.float64]
    mag_samples: NDArray[np.float64]
    acc_mean: NDArray[np.float64]
    mag_mean: NDArray[np.float64]
    acc_std: NDArray[np.float64]
    mag_std: NDArray[np.float64]
    num_samples: int
    message: str


class FusionInitializer:
    """Handles sensor data collection for EKF initialization.

    Collects and validates sensor samples while the device
    is stationary to compute initial orientation.
    """

    def __init__(self, config: Config):
        """Initialize the initializer.

        Args:
            config: System configuration.
        """
        self._config = config
        self._init_cfg = config.ekf.initialization

    def collect_samples(
        self,
        imu_source: ImuSource,
        timeout_s: float = 5.0,
    ) -> InitializationResult:
        """Collect sensor samples for initialization.

        Reads samples from IMU source and validates that
        the device is sufficiently stationary.

        Args:
            imu_source: Source of IMU readings.
            timeout_s: Timeout per sample read.

        Returns:
            InitializationResult with collected data.
        """
        num_samples = self._init_cfg.num_samples
        min_samples = self._init_cfg.min_samples

        acc_samples: List[NDArray[np.float64]] = []
        mag_samples: List[NDArray[np.float64]] = []

        logger.info("Collecting %d initialization samples...", num_samples)

        for i in range(num_samples):
            reading = imu_source.read_measurement(timeout_s=timeout_s)
            if reading is None:
                logger.warning("Timeout reading sample %d", i)
                continue

            acc_samples.append(reading.acc)
            mag_samples.append(reading.mag)

        if len(acc_samples) < min_samples:
            return InitializationResult(
                success=False,
                acc_samples=np.array([]),
                mag_samples=np.array([]),
                acc_mean=np.zeros(3),
                mag_mean=np.zeros(3),
                acc_std=np.zeros(3),
                mag_std=np.zeros(3),
                num_samples=len(acc_samples),
                message=f"Insufficient samples: {len(acc_samples)} < {min_samples}",
            )

        acc_array = np.array(acc_samples)
        mag_array = np.array(mag_samples)

        acc_mean = np.mean(acc_array, axis=0)
        mag_mean = np.mean(mag_array, axis=0)
        acc_std = np.std(acc_array, axis=0)
        mag_std = np.std(mag_array, axis=0)

        validation = self._validate_samples(acc_array, mag_array, acc_std)

        return InitializationResult(
            success=validation[0],
            acc_samples=acc_array,
            mag_samples=mag_array,
            acc_mean=acc_mean,
            mag_mean=mag_mean,
            acc_std=acc_std,
            mag_std=mag_std,
            num_samples=len(acc_samples),
            message=validation[1],
        )

    def _validate_samples(
        self,
        acc_samples: NDArray[np.float64],
        mag_samples: NDArray[np.float64],
        acc_std: NDArray[np.float64],
    ) -> Tuple[bool, str]:
        """Validate collected samples for initialization quality.

        Args:
            acc_samples: Collected accelerometer samples.
            mag_samples: Collected magnetometer samples.
            acc_std: Standard deviation of accelerometer samples.

        Returns:
            Tuple of (is_valid, message).
        """
        acc_mean = np.mean(acc_samples, axis=0)
        mag_mean = np.mean(mag_samples, axis=0)

        acc_mag = np.linalg.norm(acc_mean)
        expected_g = self._config.sensor.accelerometer.gravity_nominal
        tolerance = self._config.sensor.accelerometer.gravity_tolerance

        if abs(acc_mag - expected_g) > tolerance:
            return (
                True,
                f"Warning: Acceleration magnitude {acc_mag:.2f} differs from "
                f"expected {expected_g:.2f} m/s^2. Proceeding anyway."
            )

        gyr_threshold = np.deg2rad(
            self._config.sensor.gyroscope.stationary_threshold_dps
        )
        if np.any(acc_std > 0.5):
            return (
                True,
                "Warning: High accelerometer noise detected. "
                "Device may not be stationary."
            )

        mag_mag = np.linalg.norm(mag_mean)
        min_field = self._config.sensor.magnetometer.min_field_ut
        max_field = self._config.sensor.magnetometer.max_field_ut

        if mag_mag < min_field or mag_mag > max_field:
            return (
                True,
                f"Warning: Magnetic field magnitude {mag_mag:.1f} uT "
                f"outside expected range [{min_field}, {max_field}] uT"
            )

        return (True, "Initialization samples validated successfully")

    def check_stationary(
        self,
        readings: List[ImuReading],
    ) -> Tuple[bool, float]:
        """Check if device is stationary based on gyroscope readings.

        Args:
            readings: Recent IMU readings to analyze.

        Returns:
            Tuple of (is_stationary, max_angular_rate_dps).
        """
        if not readings:
            return False, 0.0

        gyr_magnitudes = [r.gyr_magnitude for r in readings]
        max_rate = max(gyr_magnitudes)
        max_rate_dps = np.rad2deg(max_rate)

        threshold = self._config.sensor.gyroscope.stationary_threshold_dps
        is_stationary = max_rate_dps < threshold

        return is_stationary, max_rate_dps
