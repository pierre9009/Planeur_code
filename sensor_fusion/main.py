#!/usr/bin/env python3
"""Main entry point for IMU sensor fusion.

Runs the EKF-based orientation estimation loop and outputs
JSON-formatted orientation data to stdout for web server
integration.
"""

import argparse
import json
import logging
import signal
import sys
import time
from typing import Optional

from core import Config, load_config, ImuReading
from core.validation import SensorValidator, validate_dt
from communication import ImuUart, UartError
from communication.uart import MockImuUart
from fusion import EkfWrapper, FusionInitializer
from monitoring import PerformanceMonitor

logger = logging.getLogger(__name__)

SHUTDOWN_REQUESTED = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    logger.info("Shutdown requested")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def run_fusion_loop(
    config: Config,
    use_mock: bool = False,
) -> int:
    """Run the main sensor fusion loop.

    Args:
        config: System configuration.
        use_mock: If True, use mock IMU for testing.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    imu_class = MockImuUart if use_mock else ImuUart

    try:
        imu = imu_class(config)
    except ImportError as e:
        logger.error("Failed to initialize IMU: %s", e)
        return 1

    validator = SensorValidator(config)
    ekf = EkfWrapper(config)
    initializer = FusionInitializer(config)
    monitor = PerformanceMonitor(config)

    validation_failures = 0
    emit_interval = 1.0 / config.web.emit_rate_hz
    last_emit_time = 0.0
    last_timestamp: Optional[float] = None

    try:
        imu.open()
        logger.info("Starting sensor fusion")

        init_result = initializer.collect_samples(imu)
        if not init_result.success:
            logger.error("Initialization failed: %s", init_result.message)
            return 1

        logger.info(init_result.message)
        ekf.initialize(init_result.acc_samples, init_result.mag_samples)

        while not SHUTDOWN_REQUESTED:
            monitor.start_iteration()

            reading = imu.read_measurement(timeout_s=0.5)
            if reading is None:
                continue

            validation = validator.validate_reading(reading)
            if not validation.is_valid:
                validation_failures += 1
                for error in validation.errors:
                    logger.warning("Validation: %s", error)
                continue

            for warning in validation.warnings:
                logger.debug("Validation warning: %s", warning)

            if last_timestamp is None:
                last_timestamp = reading.timestamp
                continue

            dt = reading.timestamp - last_timestamp
            last_timestamp = reading.timestamp

            dt_validation = validate_dt(dt, config)
            if not dt_validation.is_valid:
                logger.warning("Invalid dt: %s", dt_validation.errors)
                continue

            state, valid = ekf.update(reading, dt)

            if not valid:
                if ekf.needs_reinitialization():
                    logger.warning("EKF diverged, reinitializing...")
                    reinit_result = initializer.collect_samples(imu)
                    if reinit_result.success:
                        ekf.reinitialize(
                            reinit_result.acc_samples,
                            reinit_result.mag_samples,
                        )
                        last_timestamp = None
                    else:
                        logger.error("Reinitialization failed")
                continue

            monitor.end_iteration(reading.timestamp)

            now = time.time()
            if now - last_emit_time >= emit_interval:
                output = state.to_dict()
                output.update({
                    "ax": reading.ax,
                    "ay": reading.ay,
                    "az": reading.az,
                    "gx": reading.gx,
                    "gy": reading.gy,
                    "gz": reading.gz,
                    "mx": reading.mx,
                    "my": reading.my,
                    "mz": reading.mz,
                })
                print(json.dumps(output), flush=True)
                last_emit_time = now

    except UartError as e:
        logger.error("UART error: %s", e)
        return 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        imu.close()
        stats = monitor.get_stats()
        sensor_stats = imu.stats

        logger.info("Final statistics:")
        logger.info("  Iterations: %d", stats.total_iterations)
        logger.info("  Effective rate: %.1f Hz", stats.effective_rate_hz)
        logger.info("  Packets: %d total, %d valid",
                    sensor_stats.total_packets, sensor_stats.valid_packets)
        logger.info("  CRC errors: %d", sensor_stats.crc_errors)
        logger.info("  Validation failures: %d", validation_failures)
        logger.info("  EKF reinitializations: %d", ekf.health.reinit_count)

    return 0


def main() -> int:
    """Application entry point.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="IMU sensor fusion with EKF orientation estimation"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock IMU for testing",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.error("Configuration error: %s", e)
        return 1
    except Exception as e:
        logger.error("Failed to load configuration: %s", e)
        return 1

    return run_fusion_loop(config, use_mock=args.mock)


if __name__ == "__main__":
    sys.exit(main())
