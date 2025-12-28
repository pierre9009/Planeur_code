"""Data types for IMU sensor fusion."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ImuReading:
    """Single IMU measurement from all sensors.

    All values use SI units:
    - Accelerometer: m/s^2
    - Gyroscope: rad/s
    - Magnetometer: uT (microtesla)
    """
    seq: int
    timestamp: float  # Unix timestamp in seconds
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    mx: float
    my: float
    mz: float
    temperature: float  # Celsius

    @property
    def acc(self) -> NDArray[np.float64]:
        """Accelerometer vector [ax, ay, az]."""
        return np.array([self.ax, self.ay, self.az], dtype=np.float64)

    @property
    def gyr(self) -> NDArray[np.float64]:
        """Gyroscope vector [gx, gy, gz]."""
        return np.array([self.gx, self.gy, self.gz], dtype=np.float64)

    @property
    def mag(self) -> NDArray[np.float64]:
        """Magnetometer vector [mx, my, mz]."""
        return np.array([self.mx, self.my, self.mz], dtype=np.float64)

    @property
    def acc_magnitude(self) -> float:
        """Magnitude of acceleration vector."""
        return float(np.linalg.norm(self.acc))

    @property
    def gyr_magnitude(self) -> float:
        """Magnitude of angular rate vector."""
        return float(np.linalg.norm(self.gyr))

    @property
    def mag_magnitude(self) -> float:
        """Magnitude of magnetic field vector."""
        return float(np.linalg.norm(self.mag))


@dataclass
class Quaternion:
    """Unit quaternion representing orientation.

    Convention: [w, x, y, z] where w is the scalar component.
    """
    w: float
    x: float
    y: float
    z: float

    @classmethod
    def identity(cls) -> "Quaternion":
        """Return identity quaternion (no rotation)."""
        return cls(w=1.0, x=0.0, y=0.0, z=0.0)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "Quaternion":
        """Create from numpy array [w, x, y, z]."""
        return cls(w=float(arr[0]), x=float(arr[1]),
                   y=float(arr[2]), z=float(arr[3]))

    def to_array(self) -> NDArray[np.float64]:
        """Convert to numpy array [w, x, y, z]."""
        return np.array([self.w, self.x, self.y, self.z], dtype=np.float64)

    @property
    def norm(self) -> float:
        """Euclidean norm of quaternion."""
        return float(np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2))

    def is_valid(self, tolerance: float = 0.01) -> bool:
        """Check if quaternion is unit quaternion within tolerance."""
        return abs(self.norm - 1.0) <= tolerance and self._is_finite()

    def _is_finite(self) -> bool:
        """Check all components are finite."""
        return all(np.isfinite([self.w, self.x, self.y, self.z]))

    def normalized(self) -> "Quaternion":
        """Return normalized copy."""
        n = self.norm
        if n < 1e-10:
            return Quaternion.identity()
        return Quaternion(w=self.w/n, x=self.x/n, y=self.y/n, z=self.z/n)


@dataclass(frozen=True)
class EulerAngles:
    """Euler angles in radians.

    Convention: ZYX (yaw-pitch-roll) intrinsic rotations.
    """
    roll: float   # Rotation about X axis
    pitch: float  # Rotation about Y axis
    yaw: float    # Rotation about Z axis

    @property
    def roll_deg(self) -> float:
        """Roll angle in degrees."""
        return np.rad2deg(self.roll)

    @property
    def pitch_deg(self) -> float:
        """Pitch angle in degrees."""
        return np.rad2deg(self.pitch)

    @property
    def yaw_deg(self) -> float:
        """Yaw angle in degrees."""
        return np.rad2deg(self.yaw)


@dataclass
class FusionState:
    """Current state of the sensor fusion algorithm."""
    quaternion: Quaternion
    euler: EulerAngles
    timestamp: float
    dt: float
    iteration: int
    is_initialized: bool = False
    mag_trust: float = 1.0
    quaternion_norm: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "qw": self.quaternion.w,
            "qx": self.quaternion.x,
            "qy": self.quaternion.y,
            "qz": self.quaternion.z,
            "roll": self.euler.roll_deg,
            "pitch": self.euler.pitch_deg,
            "yaw": self.euler.yaw_deg,
            "dt_ms": self.dt * 1000,
            "q_norm": self.quaternion_norm,
            "mag_trust": self.mag_trust,
            "iteration": self.iteration,
        }


@dataclass
class ValidationResult:
    """Result of sensor data validation."""
    is_valid: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add validation error."""
        self.is_valid = False
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        """Add validation warning."""
        self.warnings.append(message)


@dataclass
class SensorStats:
    """Statistics for sensor data quality."""
    total_packets: int = 0
    valid_packets: int = 0
    crc_errors: int = 0
    timeouts: int = 0
    validation_failures: int = 0
    sequence_gaps: int = 0

    @property
    def packet_loss_rate(self) -> float:
        """Fraction of packets lost."""
        if self.total_packets == 0:
            return 0.0
        return 1.0 - (self.valid_packets / self.total_packets)

    @property
    def crc_error_rate(self) -> float:
        """Fraction of packets with CRC errors."""
        if self.total_packets == 0:
            return 0.0
        return self.crc_errors / self.total_packets
