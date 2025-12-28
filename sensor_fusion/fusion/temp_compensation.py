"""Temperature-dependent sensor compensation.

Provides polynomial models for gyroscope bias and accelerometer scale
as functions of temperature, fitted from multi-temperature calibration data.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import json


@dataclass
class TempCalPoint:
    """Single temperature calibration point.

    Captured from stationary sensor at a specific temperature.
    """
    temperature: float  # Celsius
    gyro_bias: NDArray[np.float64]  # [bx, by, bz] in rad/s
    accel_scale: NDArray[np.float64]  # [sx, sy, sz] scale factors
    duration_s: float  # Recording duration
    std_gyro: NDArray[np.float64]  # Standard deviation of gyro
    std_accel: NDArray[np.float64]  # Standard deviation of accel


@dataclass
class TemperatureCalibration:
    """Temperature compensation parameters.

    Derived from multi-temperature stationary calibration.
    Models bias/scale as polynomial function of temperature.
    """
    # Gyro bias model: bias(T) = a + b*T + c*T^2
    # Shape (3, 3) for [x,y,z] x [a,b,c]
    gyro_bias_coeffs: NDArray[np.float64]

    # Accel scale model: scale(T) = a + b*T
    # Shape (3, 2) for [x,y,z] x [a,b]
    accel_scale_coeffs: NDArray[np.float64]

    # Valid temperature range
    calibration_temp_range: Tuple[float, float]

    # Reference temperature (center of calibration range)
    reference_temp: float = 20.0

    # Calibration quality metrics
    gyro_residual_std: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )
    accel_residual_std: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )

    @staticmethod
    def from_calibration_data(
        data: List[TempCalPoint]
    ) -> 'TemperatureCalibration':
        """Fit temperature models from calibration measurements.

        Args:
            data: List of calibration points at different temperatures.

        Returns:
            Fitted TemperatureCalibration object.

        Raises:
            ValueError: If insufficient calibration points.
        """
        if len(data) < 2:
            raise ValueError("Need at least 2 temperature points for calibration")

        # Extract temperatures and measurements
        temps = np.array([p.temperature for p in data])
        gyro_biases = np.array([p.gyro_bias for p in data])  # (N, 3)
        accel_scales = np.array([p.accel_scale for p in data])  # (N, 3)

        # Fit gyro bias polynomial (quadratic)
        gyro_coeffs = np.zeros((3, 3))
        gyro_residual = np.zeros(3)

        for axis in range(3):
            if len(data) >= 3:
                # Fit quadratic
                coeffs = np.polyfit(temps, gyro_biases[:, axis], deg=2)
                gyro_coeffs[axis, :] = coeffs[::-1]  # [a, b, c] format
            else:
                # Fit linear
                coeffs = np.polyfit(temps, gyro_biases[:, axis], deg=1)
                gyro_coeffs[axis, 0] = coeffs[1]  # a (intercept)
                gyro_coeffs[axis, 1] = coeffs[0]  # b (slope)
                gyro_coeffs[axis, 2] = 0.0  # c = 0

            # Compute residual
            predicted = (gyro_coeffs[axis, 0] +
                        gyro_coeffs[axis, 1] * temps +
                        gyro_coeffs[axis, 2] * temps**2)
            gyro_residual[axis] = np.std(gyro_biases[:, axis] - predicted)

        # Fit accel scale polynomial (linear)
        accel_coeffs = np.zeros((3, 2))
        accel_residual = np.zeros(3)

        for axis in range(3):
            coeffs = np.polyfit(temps, accel_scales[:, axis], deg=1)
            accel_coeffs[axis, 0] = coeffs[1]  # a (intercept)
            accel_coeffs[axis, 1] = coeffs[0]  # b (slope)

            # Compute residual
            predicted = accel_coeffs[axis, 0] + accel_coeffs[axis, 1] * temps
            accel_residual[axis] = np.std(accel_scales[:, axis] - predicted)

        return TemperatureCalibration(
            gyro_bias_coeffs=gyro_coeffs,
            accel_scale_coeffs=accel_coeffs,
            calibration_temp_range=(float(temps.min()), float(temps.max())),
            reference_temp=float(np.mean(temps)),
            gyro_residual_std=gyro_residual,
            accel_residual_std=accel_residual
        )

    @staticmethod
    def identity() -> 'TemperatureCalibration':
        """Create identity calibration (no temperature compensation).

        Returns:
            Calibration that applies no corrections.
        """
        return TemperatureCalibration(
            gyro_bias_coeffs=np.zeros((3, 3)),
            accel_scale_coeffs=np.column_stack([
                np.ones(3),   # a = 1 (unity scale)
                np.zeros(3)   # b = 0 (no temp dependency)
            ]),
            calibration_temp_range=(-40.0, 85.0),
            reference_temp=20.0
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'gyro_bias_coeffs': self.gyro_bias_coeffs.tolist(),
            'accel_scale_coeffs': self.accel_scale_coeffs.tolist(),
            'calibration_temp_range': list(self.calibration_temp_range),
            'reference_temp': self.reference_temp,
            'gyro_residual_std': self.gyro_residual_std.tolist(),
            'accel_residual_std': self.accel_residual_std.tolist()
        }

    @staticmethod
    def from_dict(d: dict) -> 'TemperatureCalibration':
        """Create from dictionary."""
        return TemperatureCalibration(
            gyro_bias_coeffs=np.array(d['gyro_bias_coeffs']),
            accel_scale_coeffs=np.array(d['accel_scale_coeffs']),
            calibration_temp_range=tuple(d['calibration_temp_range']),
            reference_temp=d.get('reference_temp', 20.0),
            gyro_residual_std=np.array(d.get('gyro_residual_std', [0, 0, 0])),
            accel_residual_std=np.array(d.get('accel_residual_std', [0, 0, 0]))
        )

    def save(self, filepath: str) -> None:
        """Save calibration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(filepath: str) -> 'TemperatureCalibration':
        """Load calibration from JSON file."""
        with open(filepath, 'r') as f:
            return TemperatureCalibration.from_dict(json.load(f))


class TemperatureCompensator:
    """Apply temperature-dependent corrections to sensor readings.

    Uses polynomial models to predict sensor errors at the current
    temperature and compensates accordingly.
    """

    def __init__(
        self,
        calibration: TemperatureCalibration,
        extrapolation_limit: float = 10.0
    ):
        """Initialize compensator.

        Args:
            calibration: Temperature calibration parameters.
            extrapolation_limit: Max degrees to extrapolate beyond
                                calibration range before clamping.
        """
        self.cal = calibration
        self.extrapolation_limit = extrapolation_limit

        # Track compensation statistics
        self.compensation_count = 0
        self.extrapolation_count = 0

    def compensate_gyro(
        self,
        gyro_raw: NDArray[np.float64],
        temp: float
    ) -> NDArray[np.float64]:
        """Apply temperature correction to gyroscope.

        Args:
            gyro_raw: Raw gyro reading [gx,gy,gz] in rad/s.
            temp: Current temperature in Celsius.

        Returns:
            Temperature-compensated gyro reading.
        """
        predicted_bias = self._predict_gyro_bias(temp)
        self.compensation_count += 1
        return gyro_raw - predicted_bias

    def compensate_accel(
        self,
        accel_raw: NDArray[np.float64],
        temp: float
    ) -> NDArray[np.float64]:
        """Apply temperature correction to accelerometer.

        Args:
            accel_raw: Raw accel reading [ax,ay,az] in m/s^2.
            temp: Current temperature in Celsius.

        Returns:
            Temperature-compensated accel reading.
        """
        scale_factor = self._predict_accel_scale(temp)
        return accel_raw * scale_factor

    def _predict_gyro_bias(self, temp: float) -> NDArray[np.float64]:
        """Predict gyro bias at given temperature.

        Uses quadratic model: bias(T) = a + b*T + c*T^2

        Args:
            temp: Temperature in Celsius.

        Returns:
            Predicted bias [bx, by, bz] in rad/s.
        """
        T = self._clamp_temperature(temp)
        coeffs = self.cal.gyro_bias_coeffs

        bias = (coeffs[:, 0] +
                coeffs[:, 1] * T +
                coeffs[:, 2] * T**2)

        return bias

    def _predict_accel_scale(self, temp: float) -> NDArray[np.float64]:
        """Predict accelerometer scale factor at given temperature.

        Uses linear model: scale(T) = a + b*T

        Args:
            temp: Temperature in Celsius.

        Returns:
            Scale factors [sx, sy, sz].
        """
        T = self._clamp_temperature(temp)
        coeffs = self.cal.accel_scale_coeffs

        scale = coeffs[:, 0] + coeffs[:, 1] * T

        # Ensure scale is reasonable (0.9 to 1.1)
        scale = np.clip(scale, 0.9, 1.1)

        return scale

    def _clamp_temperature(self, temp: float) -> float:
        """Clamp temperature to valid calibration range.

        Allows limited extrapolation beyond calibration range.

        Args:
            temp: Input temperature.

        Returns:
            Clamped temperature.
        """
        t_min, t_max = self.cal.calibration_temp_range
        t_min_ext = t_min - self.extrapolation_limit
        t_max_ext = t_max + self.extrapolation_limit

        if temp < t_min_ext or temp > t_max_ext:
            self.extrapolation_count += 1

        return np.clip(temp, t_min_ext, t_max_ext)

    def is_in_calibration_range(self, temp: float) -> bool:
        """Check if temperature is within calibration range."""
        t_min, t_max = self.cal.calibration_temp_range
        return t_min <= temp <= t_max

    def get_expected_bias_at_temp(self, temp: float) -> NDArray[np.float64]:
        """Get expected gyro bias at a specific temperature.

        Useful for debugging and validation.
        """
        return self._predict_gyro_bias(temp)

    def get_stats(self) -> dict:
        """Get compensation statistics."""
        return {
            'compensation_count': self.compensation_count,
            'extrapolation_count': self.extrapolation_count,
            'extrapolation_rate': (
                self.extrapolation_count / max(1, self.compensation_count)
            )
        }


def create_default_calibration() -> TemperatureCalibration:
    """Create a reasonable default calibration for typical MEMS IMU.

    These values are approximations for a typical low-cost IMU.
    For production use, perform actual temperature calibration.

    Returns:
        Default temperature calibration.
    """
    # Typical gyro bias temperature coefficient: ~0.01 deg/s/C
    # Convert to rad/s: 0.01 * pi/180 ≈ 1.7e-4 rad/s/C
    gyro_temp_coeff = 1.7e-4

    gyro_bias_coeffs = np.array([
        [0.0, gyro_temp_coeff, 0.0],  # X axis
        [0.0, gyro_temp_coeff, 0.0],  # Y axis
        [0.0, gyro_temp_coeff, 0.0],  # Z axis
    ])

    # Typical accel scale varies ~0.01% per degree C
    accel_scale_coeffs = np.array([
        [1.0, -1e-4],  # X axis
        [1.0, -1e-4],  # Y axis
        [1.0, -1e-4],  # Z axis
    ])

    return TemperatureCalibration(
        gyro_bias_coeffs=gyro_bias_coeffs,
        accel_scale_coeffs=accel_scale_coeffs,
        calibration_temp_range=(-10.0, 50.0),
        reference_temp=25.0
    )
