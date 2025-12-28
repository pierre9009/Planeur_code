"""Pytest fixtures for IMU sensor fusion tests."""

import sys
from pathlib import Path
import pytest
import numpy as np
from numpy.typing import NDArray

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from core.config import Config
from core.types import ImuReading, Quaternion


@pytest.fixture
def config() -> Config:
    """Create default configuration for tests."""
    return Config()


@pytest.fixture
def sample_imu_reading() -> ImuReading:
    """Create a sample IMU reading for tests.

    Returns a reading representing a stationary IMU in NED frame
    with gravity pointing down (+Z) and a nominal magnetic field.
    """
    return ImuReading(
        seq=1,
        timestamp=1000.0,
        ax=0.0,
        ay=0.0,
        az=9.81,
        gx=0.0,
        gy=0.0,
        gz=0.0,
        mx=20.0,
        my=5.0,
        mz=45.0,
        temperature=25.0,
    )


@pytest.fixture
def noisy_imu_reading() -> ImuReading:
    """Create a noisy IMU reading for tests."""
    np.random.seed(42)
    return ImuReading(
        seq=1,
        timestamp=1000.0,
        ax=0.05,
        ay=-0.03,
        az=9.78,
        gx=0.001,
        gy=-0.002,
        gz=0.001,
        mx=20.5,
        my=4.8,
        mz=44.5,
        temperature=25.2,
    )


@pytest.fixture
def identity_quaternion() -> Quaternion:
    """Create identity quaternion (no rotation)."""
    return Quaternion.identity()


@pytest.fixture
def sample_quaternion() -> Quaternion:
    """Create a sample non-identity quaternion.

    Represents approximately 30 degree rotation about Z axis.
    """
    angle = np.deg2rad(30)
    return Quaternion(
        w=np.cos(angle / 2),
        x=0.0,
        y=0.0,
        z=np.sin(angle / 2),
    )


@pytest.fixture
def acc_samples() -> NDArray[np.float64]:
    """Create accelerometer samples for initialization.

    Returns 100 samples of stationary accelerometer readings
    with small Gaussian noise.
    """
    np.random.seed(42)
    n_samples = 100
    noise = np.random.normal(0, 0.01, (n_samples, 3))
    samples = np.zeros((n_samples, 3))
    samples[:, 2] = 9.81
    return samples + noise


@pytest.fixture
def mag_samples() -> NDArray[np.float64]:
    """Create magnetometer samples for initialization.

    Returns 100 samples of stationary magnetometer readings
    with small Gaussian noise.
    """
    np.random.seed(42)
    n_samples = 100
    noise = np.random.normal(0, 0.1, (n_samples, 3))
    samples = np.zeros((n_samples, 3))
    samples[:, 0] = 20.0
    samples[:, 1] = 5.0
    samples[:, 2] = 45.0
    return samples + noise


@pytest.fixture
def invalid_reading_nan() -> ImuReading:
    """Create an IMU reading with NaN values."""
    return ImuReading(
        seq=1,
        timestamp=1000.0,
        ax=float("nan"),
        ay=0.0,
        az=9.81,
        gx=0.0,
        gy=0.0,
        gz=0.0,
        mx=20.0,
        my=5.0,
        mz=45.0,
        temperature=25.0,
    )


@pytest.fixture
def invalid_reading_inf() -> ImuReading:
    """Create an IMU reading with Inf values."""
    return ImuReading(
        seq=1,
        timestamp=1000.0,
        ax=0.0,
        ay=float("inf"),
        az=9.81,
        gx=0.0,
        gy=0.0,
        gz=0.0,
        mx=20.0,
        my=5.0,
        mz=45.0,
        temperature=25.0,
    )


@pytest.fixture
def reading_out_of_range() -> ImuReading:
    """Create an IMU reading with out-of-range values."""
    return ImuReading(
        seq=1,
        timestamp=1000.0,
        ax=200.0,
        ay=0.0,
        az=9.81,
        gx=0.0,
        gy=0.0,
        gz=0.0,
        mx=20.0,
        my=5.0,
        mz=45.0,
        temperature=25.0,
    )
