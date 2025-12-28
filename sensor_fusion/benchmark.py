#!/usr/bin/env python3
"""Performance benchmarking script for sensor fusion.

Measures timing characteristics and validates real-time performance
requirements for the IMU sensor fusion pipeline.
"""

import argparse
import statistics
import sys
import time
from typing import List

import numpy as np

from core import Config, load_config
from core.types import ImuReading
from core.validation import SensorValidator
from core.quaternion import QuaternionOps
from communication.uart import MockImuUart
from fusion import EkfWrapper, FusionInitializer
from monitoring import PerformanceMonitor


def benchmark_quaternion_ops(iterations: int = 10000) -> dict:
    """Benchmark quaternion operations.

    Args:
        iterations: Number of iterations to run.

    Returns:
        Dictionary with timing results.
    """
    acc = np.array([0.1, -0.05, 9.78])
    mag = np.array([20.0, 5.0, 45.0])

    start = time.perf_counter()
    for _ in range(iterations):
        q = QuaternionOps.from_acc_mag(acc, mag)
    from_acc_mag_time = (time.perf_counter() - start) / iterations * 1e6

    from core.types import Quaternion
    q = Quaternion(w=0.7071, x=0.0, y=0.7071, z=0.0)

    start = time.perf_counter()
    for _ in range(iterations):
        euler = QuaternionOps.to_euler(q)
    to_euler_time = (time.perf_counter() - start) / iterations * 1e6

    q1 = Quaternion(w=0.7071, x=0.0, y=0.7071, z=0.0)
    q2 = Quaternion(w=0.9239, x=0.0, y=0.0, z=0.3827)

    start = time.perf_counter()
    for _ in range(iterations):
        q3 = QuaternionOps.multiply(q1, q2)
    multiply_time = (time.perf_counter() - start) / iterations * 1e6

    return {
        "from_acc_mag_us": from_acc_mag_time,
        "to_euler_us": to_euler_time,
        "multiply_us": multiply_time,
    }


def benchmark_validation(config: Config, iterations: int = 10000) -> dict:
    """Benchmark sensor validation.

    Args:
        config: System configuration.
        iterations: Number of iterations to run.

    Returns:
        Dictionary with timing results.
    """
    validator = SensorValidator(config)

    reading = ImuReading(
        seq=1, timestamp=1000.0,
        ax=0.05, ay=-0.03, az=9.78,
        gx=0.001, gy=-0.002, gz=0.001,
        mx=20.5, my=4.8, mz=44.5,
        temperature=25.2,
    )

    start = time.perf_counter()
    for i in range(iterations):
        reading = ImuReading(
            seq=i, timestamp=1000.0 + i * 0.01,
            ax=0.05, ay=-0.03, az=9.78,
            gx=0.001, gy=-0.002, gz=0.001,
            mx=20.5, my=4.8, mz=44.5,
            temperature=25.2,
        )
        validator.validate_reading(reading)
    validate_time = (time.perf_counter() - start) / iterations * 1e6

    return {
        "validate_reading_us": validate_time,
    }


def benchmark_ekf_update(config: Config, iterations: int = 1000) -> dict:
    """Benchmark EKF update step.

    Args:
        config: System configuration.
        iterations: Number of iterations to run.

    Returns:
        Dictionary with timing results.
    """
    with MockImuUart(config) as imu:
        initializer = FusionInitializer(config)
        ekf = EkfWrapper(config)

        init_result = initializer.collect_samples(imu)
        ekf.initialize(init_result.acc_samples, init_result.mag_samples)

        reading = imu.read_measurement()

        update_times: List[float] = []

        for _ in range(iterations):
            start = time.perf_counter()
            ekf.update(reading, dt=0.01)
            elapsed = (time.perf_counter() - start) * 1e6
            update_times.append(elapsed)

    return {
        "ekf_update_mean_us": statistics.mean(update_times),
        "ekf_update_std_us": statistics.stdev(update_times),
        "ekf_update_max_us": max(update_times),
        "ekf_update_min_us": min(update_times),
    }


def benchmark_full_loop(config: Config, duration_s: float = 5.0) -> dict:
    """Benchmark complete fusion loop.

    Args:
        config: System configuration.
        duration_s: Duration to run benchmark.

    Returns:
        Dictionary with timing results.
    """
    with MockImuUart(config) as imu:
        validator = SensorValidator(config)
        initializer = FusionInitializer(config)
        ekf = EkfWrapper(config)
        monitor = PerformanceMonitor(config)

        init_result = initializer.collect_samples(imu)
        ekf.initialize(init_result.acc_samples, init_result.mag_samples)

        last_timestamp = None
        loop_times: List[float] = []
        start_time = time.time()

        while time.time() - start_time < duration_s:
            loop_start = time.perf_counter()
            monitor.start_iteration()

            reading = imu.read_measurement()
            if reading is None:
                continue

            validator.validate_reading(reading)

            if last_timestamp is None:
                last_timestamp = reading.timestamp
                continue

            dt = reading.timestamp - last_timestamp
            last_timestamp = reading.timestamp

            ekf.update(reading, dt)
            monitor.end_iteration(reading.timestamp)

            loop_time = (time.perf_counter() - loop_start) * 1e6
            loop_times.append(loop_time)

    stats = monitor.get_stats()

    return {
        "loop_mean_us": statistics.mean(loop_times),
        "loop_std_us": statistics.stdev(loop_times),
        "loop_max_us": max(loop_times),
        "loop_min_us": min(loop_times),
        "effective_rate_hz": stats.effective_rate_hz,
        "total_iterations": len(loop_times),
    }


def print_results(title: str, results: dict) -> None:
    """Print benchmark results.

    Args:
        title: Section title.
        results: Dictionary of results.
    """
    print(f"\n{title}")
    print("-" * 50)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


def main() -> int:
    """Run benchmarks.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(description="Benchmark sensor fusion")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with fewer iterations",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception:
        config = Config()

    iterations = 1000 if args.quick else 10000
    duration = 2.0 if args.quick else 5.0

    print("=" * 60)
    print("IMU SENSOR FUSION BENCHMARK")
    print("=" * 60)

    print("\nRunning quaternion benchmarks...")
    quat_results = benchmark_quaternion_ops(iterations)
    print_results("Quaternion Operations (microseconds)", quat_results)

    print("\nRunning validation benchmarks...")
    val_results = benchmark_validation(config, iterations)
    print_results("Validation (microseconds)", val_results)

    print("\nRunning EKF update benchmarks...")
    ekf_results = benchmark_ekf_update(config, iterations // 10)
    print_results("EKF Update (microseconds)", ekf_results)

    print("\nRunning full loop benchmark...")
    loop_results = benchmark_full_loop(config, duration)
    print_results("Full Loop (microseconds)", loop_results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_per_iteration = (
        quat_results["to_euler_us"] +
        val_results["validate_reading_us"] +
        ekf_results["ekf_update_mean_us"]
    )

    print(f"\nEstimated per-iteration time: {total_per_iteration:.2f} us")
    print(f"Actual loop time (mean): {loop_results['loop_mean_us']:.2f} us")
    print(f"Maximum sustainable rate: {1e6 / loop_results['loop_max_us']:.0f} Hz")

    target_rate = config.sensor.sample_rate_hz
    target_time = 1e6 / target_rate

    if loop_results["loop_max_us"] < target_time:
        print(f"\nREAL-TIME CAPABLE at {target_rate} Hz")
        margin = (target_time - loop_results["loop_mean_us"]) / target_time * 100
        print(f"Time margin: {margin:.1f}%")
    else:
        print(f"\nWARNING: Cannot sustain {target_rate} Hz")
        actual_max = 1e6 / loop_results["loop_max_us"]
        print(f"Maximum sustainable: {actual_max:.0f} Hz")

    return 0


if __name__ == "__main__":
    sys.exit(main())
