"""Sensor fusion module for IMU orientation estimation."""

from .magnetometer import MagnetometerProcessor
from .ekf_wrapper import EkfWrapper
from .initializer import FusionInitializer

# New robust fusion components
from .adaptive_ekf import AdaptiveBiasEKF
from .temp_compensation import (
    TemperatureCalibration,
    TemperatureCompensator,
    TempCalPoint,
    create_default_calibration
)
from .mag_health import (
    MagnetometerHealthCheck,
    MagHealthStatus,
    AdaptiveMagnetometerValidator
)
from .fusion_manager import (
    RobustFusion,
    FusionStatus,
    startup_gyro_calibration,
    collect_initialization_samples
)

__all__ = [
    # Legacy components
    "MagnetometerProcessor",
    "EkfWrapper",
    "FusionInitializer",
    # New robust fusion
    "AdaptiveBiasEKF",
    "TemperatureCalibration",
    "TemperatureCompensator",
    "TempCalPoint",
    "create_default_calibration",
    "MagnetometerHealthCheck",
    "MagHealthStatus",
    "AdaptiveMagnetometerValidator",
    "RobustFusion",
    "FusionStatus",
    "startup_gyro_calibration",
    "collect_initialization_samples",
]
