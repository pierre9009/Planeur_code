"""Configuration management for IMU sensor fusion."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class UartConfig:
    """UART communication configuration."""
    port: str = "/dev/ttyS0"
    baudrate: int = 115200
    timeout_s: float = 0.1
    write_timeout_s: float = 1.0


@dataclass
class AccelerometerConfig:
    """Accelerometer sensor configuration."""
    range_g: float = 16.0
    gravity_nominal: float = 9.81
    gravity_tolerance: float = 0.5


@dataclass
class GyroscopeConfig:
    """Gyroscope sensor configuration."""
    range_dps: float = 2000.0
    stationary_threshold_dps: float = 5.0


@dataclass
class MagnetometerSensorConfig:
    """Magnetometer sensor configuration."""
    range_ut: float = 4900.0
    min_field_ut: float = 20.0
    max_field_ut: float = 100.0


@dataclass
class SensorConfig:
    """Sensor configuration."""
    sample_rate_hz: int = 100
    accelerometer: AccelerometerConfig = field(default_factory=AccelerometerConfig)
    gyroscope: GyroscopeConfig = field(default_factory=GyroscopeConfig)
    magnetometer: MagnetometerSensorConfig = field(default_factory=MagnetometerSensorConfig)


@dataclass
class NoiseConfig:
    """EKF noise configuration."""
    gyroscope: float = 0.3
    accelerometer: float = 0.5
    magnetometer: float = 0.8


@dataclass
class InitializationConfig:
    """EKF initialization configuration."""
    num_samples: int = 100
    min_samples: int = 50
    max_tilt_deg: float = 30.0


@dataclass
class EkfConfig:
    """EKF algorithm configuration."""
    frame: str = "NED"
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    initialization: InitializationConfig = field(default_factory=InitializationConfig)


@dataclass
class QuaternionValidationConfig:
    """Quaternion validation configuration."""
    norm_tolerance: float = 0.01
    divergence_threshold: float = 0.1


@dataclass
class TimestampValidationConfig:
    """Timestamp validation configuration."""
    max_dt_s: float = 0.1
    min_dt_s: float = 0.001


@dataclass
class ValidationConfig:
    """Validation configuration."""
    quaternion: QuaternionValidationConfig = field(default_factory=QuaternionValidationConfig)
    timestamp: TimestampValidationConfig = field(default_factory=TimestampValidationConfig)


@dataclass
class DisturbanceConfig:
    """Magnetometer disturbance detection configuration."""
    field_magnitude_tolerance: float = 0.15
    update_rate: float = 0.01
    recovery_samples: int = 50


@dataclass
class TrustConfig:
    """Magnetometer trust configuration."""
    nominal: float = 1.0
    disturbed: float = 0.1
    recovery_rate: float = 0.02


@dataclass
class MagnetometerConfig:
    """Magnetometer processing configuration."""
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    trust: TrustConfig = field(default_factory=TrustConfig)


@dataclass
class LoopTimingConfig:
    """Loop timing monitoring configuration."""
    target_hz: int = 100
    jitter_warning_ms: float = 5.0


@dataclass
class MonitoringConfig:
    """Performance monitoring configuration."""
    loop_timing: LoopTimingConfig = field(default_factory=LoopTimingConfig)
    window_size: int = 1000
    log_interval_s: float = 10.0


@dataclass
class WebConfig:
    """Web server configuration."""
    host: str = "0.0.0.0"
    port: int = 5000
    emit_rate_hz: int = 10


@dataclass
class Config:
    """Complete configuration for IMU sensor fusion."""
    uart: UartConfig = field(default_factory=UartConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    ekf: EkfConfig = field(default_factory=EkfConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    magnetometer: MagnetometerConfig = field(default_factory=MagnetometerConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    web: WebConfig = field(default_factory=WebConfig)


def _dict_to_dataclass(data: dict, cls: type) -> object:
    """Recursively convert dictionary to dataclass."""
    if not hasattr(cls, "__dataclass_fields__"):
        return data

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key in field_types:
            field_type = field_types[key]
            if hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
                kwargs[key] = _dict_to_dataclass(value, field_type)
            else:
                kwargs[key] = value

    return cls(**kwargs)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, uses default.

    Returns:
        Configuration object with all settings.

    Raises:
        FileNotFoundError: If specified config file doesn't exist.
        ImportError: If PyYAML is not installed.
    """
    if config_path is None:
        env_path = os.environ.get("IMU_CONFIG_PATH")
        if env_path:
            config_path = env_path
        else:
            default_path = Path(__file__).parent.parent / "config" / "default.yaml"
            if default_path.exists():
                config_path = str(default_path)
            else:
                return Config()

    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is required to load configuration files")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return Config()

    return _build_config(data)


def _build_config(data: dict) -> Config:
    """Build Config object from dictionary."""
    uart = UartConfig(**data.get("uart", {}))

    sensor_data = data.get("sensor", {})
    sensor = SensorConfig(
        sample_rate_hz=sensor_data.get("sample_rate_hz", 100),
        accelerometer=AccelerometerConfig(**sensor_data.get("accelerometer", {})),
        gyroscope=GyroscopeConfig(**sensor_data.get("gyroscope", {})),
        magnetometer=MagnetometerSensorConfig(**sensor_data.get("magnetometer", {})),
    )

    ekf_data = data.get("ekf", {})
    ekf = EkfConfig(
        frame=ekf_data.get("frame", "NED"),
        noise=NoiseConfig(**ekf_data.get("noise", {})),
        initialization=InitializationConfig(**ekf_data.get("initialization", {})),
    )

    val_data = data.get("validation", {})
    validation = ValidationConfig(
        quaternion=QuaternionValidationConfig(**val_data.get("quaternion", {})),
        timestamp=TimestampValidationConfig(**val_data.get("timestamp", {})),
    )

    mag_data = data.get("magnetometer", {})
    magnetometer = MagnetometerConfig(
        disturbance=DisturbanceConfig(**mag_data.get("disturbance", {})),
        trust=TrustConfig(**mag_data.get("trust", {})),
    )

    mon_data = data.get("monitoring", {})
    monitoring = MonitoringConfig(
        loop_timing=LoopTimingConfig(**mon_data.get("loop_timing", {})),
        window_size=mon_data.get("window_size", 1000),
        log_interval_s=mon_data.get("log_interval_s", 10.0),
    )

    web = WebConfig(**data.get("web", {}))

    return Config(
        uart=uart,
        sensor=sensor,
        ekf=ekf,
        validation=validation,
        magnetometer=magnetometer,
        monitoring=monitoring,
        web=web,
    )
