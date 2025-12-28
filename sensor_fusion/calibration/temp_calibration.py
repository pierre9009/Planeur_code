"""Temperature calibration procedure for IMU sensors.

Collects stationary sensor data at multiple temperatures to build
polynomial compensation models for gyroscope bias and accelerometer scale.

Typical procedure:
1. Room temperature (20°C): 30 min stationary recording
2. Freezer (-10°C): 30 min stationary (sensor in sealed bag)
3. Optional warm environment (+30°C): 30 min stationary

For each temperature:
- Wait 15 min for thermal stabilization
- Record stationary data
- Extract gyro bias (mean) and accel scale
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Optional, Protocol
import json
import time

from ..fusion.temp_compensation import (
    TempCalPoint,
    TemperatureCalibration
)


class ImuSource(Protocol):
    """Protocol for IMU data source."""

    def read_measurement(self, timeout_s: float = 1.0):
        """Read single IMU measurement."""
        ...


@dataclass
class CalibrationRecording:
    """Raw data from a temperature calibration recording."""
    temperatures: List[float]
    gyro_readings: List[NDArray[np.float64]]
    accel_readings: List[NDArray[np.float64]]
    timestamps: List[float]
    target_temp: float
    duration_s: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'temperatures': self.temperatures,
            'gyro_readings': [g.tolist() for g in self.gyro_readings],
            'accel_readings': [a.tolist() for a in self.accel_readings],
            'timestamps': self.timestamps,
            'target_temp': self.target_temp,
            'duration_s': self.duration_s
        }

    @staticmethod
    def from_dict(d: dict) -> 'CalibrationRecording':
        """Create from dictionary."""
        return CalibrationRecording(
            temperatures=d['temperatures'],
            gyro_readings=[np.array(g) for g in d['gyro_readings']],
            accel_readings=[np.array(a) for a in d['accel_readings']],
            timestamps=d['timestamps'],
            target_temp=d['target_temp'],
            duration_s=d['duration_s']
        )


def analyze_temperature_point(
    recording: CalibrationRecording
) -> TempCalPoint:
    """Analyze a temperature calibration recording.

    Extracts gyro bias and accel scale from stationary data.

    Args:
        recording: Raw calibration recording at one temperature.

    Returns:
        TempCalPoint with extracted parameters.
    """
    # Convert to arrays
    gyro_array = np.array(recording.gyro_readings)  # (N, 3)
    accel_array = np.array(recording.accel_readings)  # (N, 3)
    temp_array = np.array(recording.temperatures)

    # Mean temperature during recording
    mean_temp = float(np.mean(temp_array))

    # Gyro bias = mean of stationary gyro readings
    gyro_mean = np.mean(gyro_array, axis=0)
    gyro_std = np.std(gyro_array, axis=0)

    # Accel scale = actual / expected
    # When stationary, accel should measure [0, 0, 9.81] in NED
    # We measure magnitude and compute scale factor
    accel_magnitudes = np.linalg.norm(accel_array, axis=1)
    mean_magnitude = np.mean(accel_magnitudes)

    # Scale factor per axis (simplified: uniform scaling)
    # For proper calibration, need multi-axis rotations
    accel_scale = np.ones(3) * (9.81 / mean_magnitude)
    accel_std = np.std(accel_array, axis=0)

    return TempCalPoint(
        temperature=mean_temp,
        gyro_bias=gyro_mean,
        accel_scale=accel_scale,
        duration_s=recording.duration_s,
        std_gyro=gyro_std,
        std_accel=accel_std
    )


class TemperatureCalibrator:
    """Interactive temperature calibration manager.

    Guides user through multi-temperature calibration procedure.
    """

    # Recommended temperature points
    DEFAULT_TEMP_POINTS = [-10, 0, 20, 30]

    def __init__(
        self,
        imu_source: ImuSource,
        stabilization_time_s: float = 900.0,  # 15 minutes
        recording_time_s: float = 1800.0,      # 30 minutes
        sample_rate_hz: float = 100.0
    ):
        """Initialize calibrator.

        Args:
            imu_source: IMU data source.
            stabilization_time_s: Time to wait for thermal stabilization.
            recording_time_s: Recording duration per temperature.
            sample_rate_hz: Expected sample rate.
        """
        self.imu = imu_source
        self.stabilization_time_s = stabilization_time_s
        self.recording_time_s = recording_time_s
        self.sample_rate_hz = sample_rate_hz

        self.recordings: List[CalibrationRecording] = []
        self.calibration_points: List[TempCalPoint] = []

    def record_temperature_point(
        self,
        target_temp: float,
        duration_s: Optional[float] = None,
        progress_callback=None
    ) -> CalibrationRecording:
        """Record calibration data at a specific temperature.

        Args:
            target_temp: Target temperature in Celsius.
            duration_s: Recording duration (uses default if None).
            progress_callback: Optional callback(elapsed, total) for progress.

        Returns:
            CalibrationRecording with raw data.
        """
        if duration_s is None:
            duration_s = self.recording_time_s

        temperatures = []
        gyro_readings = []
        accel_readings = []
        timestamps = []

        start_time = time.time()
        sample_interval = 1.0 / self.sample_rate_hz

        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration_s:
                break

            reading = self.imu.read_measurement(timeout_s=0.1)
            if reading is not None:
                temperatures.append(reading.temperature)
                gyro_readings.append(reading.gyr.copy())
                accel_readings.append(reading.acc.copy())
                timestamps.append(reading.timestamp)

            if progress_callback is not None:
                progress_callback(elapsed, duration_s)

            # Rate limiting
            time.sleep(sample_interval * 0.5)

        recording = CalibrationRecording(
            temperatures=temperatures,
            gyro_readings=gyro_readings,
            accel_readings=accel_readings,
            timestamps=timestamps,
            target_temp=target_temp,
            duration_s=duration_s
        )

        self.recordings.append(recording)
        return recording

    def analyze_recording(
        self,
        recording: CalibrationRecording
    ) -> TempCalPoint:
        """Analyze a recording and extract calibration point.

        Args:
            recording: Raw calibration recording.

        Returns:
            TempCalPoint with extracted parameters.
        """
        point = analyze_temperature_point(recording)
        self.calibration_points.append(point)
        return point

    def build_calibration(self) -> TemperatureCalibration:
        """Build temperature calibration from collected points.

        Returns:
            TemperatureCalibration ready for use.

        Raises:
            ValueError: If insufficient calibration points.
        """
        if len(self.calibration_points) < 2:
            raise ValueError(
                f"Need at least 2 temperature points, have {len(self.calibration_points)}"
            )

        return TemperatureCalibration.from_calibration_data(
            self.calibration_points
        )

    def save_recordings(self, filepath: str) -> None:
        """Save raw recordings to JSON file.

        Args:
            filepath: Path to save file.
        """
        data = {
            'recordings': [r.to_dict() for r in self.recordings],
            'calibration_points': [
                {
                    'temperature': p.temperature,
                    'gyro_bias': p.gyro_bias.tolist(),
                    'accel_scale': p.accel_scale.tolist(),
                    'duration_s': p.duration_s,
                    'std_gyro': p.std_gyro.tolist(),
                    'std_accel': p.std_accel.tolist()
                }
                for p in self.calibration_points
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_recordings(self, filepath: str) -> None:
        """Load recordings from JSON file.

        Args:
            filepath: Path to load file.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.recordings = [
            CalibrationRecording.from_dict(r)
            for r in data['recordings']
        ]

        self.calibration_points = [
            TempCalPoint(
                temperature=p['temperature'],
                gyro_bias=np.array(p['gyro_bias']),
                accel_scale=np.array(p['accel_scale']),
                duration_s=p['duration_s'],
                std_gyro=np.array(p['std_gyro']),
                std_accel=np.array(p['std_accel'])
            )
            for p in data['calibration_points']
        ]


def quick_temperature_recording(
    imu_source: ImuSource,
    duration_s: float = 60.0,
    sample_rate_hz: float = 100.0
) -> TempCalPoint:
    """Quick single-temperature calibration recording.

    For rapid testing and validation. Not as accurate as full procedure.

    Args:
        imu_source: IMU data source.
        duration_s: Recording duration.
        sample_rate_hz: Sample rate.

    Returns:
        TempCalPoint from recording.
    """
    calibrator = TemperatureCalibrator(
        imu_source,
        recording_time_s=duration_s,
        sample_rate_hz=sample_rate_hz
    )

    # Record at current temperature
    recording = calibrator.record_temperature_point(
        target_temp=25.0,  # Will be overwritten with actual
        duration_s=duration_s
    )

    return calibrator.analyze_recording(recording)
