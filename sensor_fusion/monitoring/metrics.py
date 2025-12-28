"""Performance metrics and monitoring for sensor fusion loop."""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque
import numpy as np

from ..core.config import Config
from ..core.types import SensorStats

logger = logging.getLogger(__name__)


@dataclass
class LoopMetrics:
    """Metrics for a single loop iteration."""
    timestamp: float
    dt_ms: float
    loop_time_ms: float
    iteration: int


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    mean_dt_ms: float
    std_dt_ms: float
    max_dt_ms: float
    min_dt_ms: float
    mean_loop_time_ms: float
    max_loop_time_ms: float
    effective_rate_hz: float
    jitter_ms: float
    dropped_samples: int
    total_iterations: int


class PerformanceMonitor:
    """Monitors and reports performance metrics for the fusion loop.

    Tracks timing statistics, detects jitter and dropped samples,
    and provides periodic performance reports.
    """

    def __init__(self, config: Config):
        """Initialize performance monitor.

        Args:
            config: System configuration with monitoring settings.
        """
        self._config = config
        self._mon_cfg = config.monitoring
        self._timing_cfg = config.monitoring.loop_timing

        window = self._mon_cfg.window_size
        self._dt_history: Deque[float] = deque(maxlen=window)
        self._loop_time_history: Deque[float] = deque(maxlen=window)
        self._timestamps: Deque[float] = deque(maxlen=window)

        self._iteration = 0
        self._dropped_samples = 0
        self._last_log_time = 0.0
        self._last_timestamp: Optional[float] = None
        self._loop_start_time: Optional[float] = None

        self._target_dt_ms = 1000.0 / self._timing_cfg.target_hz
        self._jitter_threshold = self._timing_cfg.jitter_warning_ms

    def start_iteration(self) -> None:
        """Mark the start of a loop iteration."""
        self._loop_start_time = time.perf_counter()

    def end_iteration(self, timestamp: float) -> LoopMetrics:
        """Mark the end of a loop iteration and compute metrics.

        Args:
            timestamp: Current sample timestamp.

        Returns:
            Metrics for this iteration.
        """
        now = time.perf_counter()
        loop_time_ms = 0.0

        if self._loop_start_time is not None:
            loop_time_ms = (now - self._loop_start_time) * 1000

        dt_ms = 0.0
        if self._last_timestamp is not None:
            dt_ms = (timestamp - self._last_timestamp) * 1000
            self._dt_history.append(dt_ms)

            expected_samples = int(dt_ms / self._target_dt_ms + 0.5)
            if expected_samples > 1:
                self._dropped_samples += expected_samples - 1

        self._loop_time_history.append(loop_time_ms)
        self._timestamps.append(timestamp)
        self._last_timestamp = timestamp
        self._iteration += 1

        if dt_ms > self._target_dt_ms + self._jitter_threshold:
            logger.debug(
                "High jitter: dt=%.2f ms (target=%.2f ms)",
                dt_ms, self._target_dt_ms
            )

        self._maybe_log_stats()

        return LoopMetrics(
            timestamp=timestamp,
            dt_ms=dt_ms,
            loop_time_ms=loop_time_ms,
            iteration=self._iteration,
        )

    def _maybe_log_stats(self) -> None:
        """Log statistics periodically."""
        now = time.time()
        interval = self._mon_cfg.log_interval_s

        if now - self._last_log_time >= interval:
            stats = self.get_stats()
            logger.info(
                "Performance: rate=%.1f Hz, dt=%.2f+/-%.2f ms, "
                "loop=%.2f ms, dropped=%d",
                stats.effective_rate_hz,
                stats.mean_dt_ms,
                stats.std_dt_ms,
                stats.mean_loop_time_ms,
                stats.dropped_samples,
            )
            self._last_log_time = now

    def get_stats(self) -> PerformanceStats:
        """Get aggregated performance statistics.

        Returns:
            PerformanceStats with current metrics.
        """
        if not self._dt_history:
            return PerformanceStats(
                mean_dt_ms=0.0,
                std_dt_ms=0.0,
                max_dt_ms=0.0,
                min_dt_ms=0.0,
                mean_loop_time_ms=0.0,
                max_loop_time_ms=0.0,
                effective_rate_hz=0.0,
                jitter_ms=0.0,
                dropped_samples=0,
                total_iterations=0,
            )

        dt_array = np.array(self._dt_history)
        loop_array = np.array(self._loop_time_history)

        mean_dt = float(np.mean(dt_array))
        effective_rate = 1000.0 / mean_dt if mean_dt > 0 else 0.0

        return PerformanceStats(
            mean_dt_ms=mean_dt,
            std_dt_ms=float(np.std(dt_array)),
            max_dt_ms=float(np.max(dt_array)),
            min_dt_ms=float(np.min(dt_array)),
            mean_loop_time_ms=float(np.mean(loop_array)),
            max_loop_time_ms=float(np.max(loop_array)),
            effective_rate_hz=effective_rate,
            jitter_ms=float(np.std(dt_array)),
            dropped_samples=self._dropped_samples,
            total_iterations=self._iteration,
        )

    def reset(self) -> None:
        """Reset all metrics."""
        self._dt_history.clear()
        self._loop_time_history.clear()
        self._timestamps.clear()
        self._iteration = 0
        self._dropped_samples = 0
        self._last_timestamp = None
        self._loop_start_time = None


class CombinedStats:
    """Combines sensor and performance statistics for reporting."""

    def __init__(self):
        """Initialize combined stats tracker."""
        self._sensor_stats: Optional[SensorStats] = None
        self._perf_stats: Optional[PerformanceStats] = None
        self._ekf_reinit_count = 0
        self._validation_failures = 0

    def update(
        self,
        sensor_stats: SensorStats,
        perf_stats: PerformanceStats,
        ekf_reinit_count: int,
        validation_failures: int,
    ) -> None:
        """Update combined statistics.

        Args:
            sensor_stats: Current sensor statistics.
            perf_stats: Current performance statistics.
            ekf_reinit_count: Number of EKF reinitializations.
            validation_failures: Number of validation failures.
        """
        self._sensor_stats = sensor_stats
        self._perf_stats = perf_stats
        self._ekf_reinit_count = ekf_reinit_count
        self._validation_failures = validation_failures

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "ekf_reinit_count": self._ekf_reinit_count,
            "validation_failures": self._validation_failures,
        }

        if self._sensor_stats:
            result.update({
                "packets_total": self._sensor_stats.total_packets,
                "packets_valid": self._sensor_stats.valid_packets,
                "crc_errors": self._sensor_stats.crc_errors,
                "timeouts": self._sensor_stats.timeouts,
                "packet_loss_rate": self._sensor_stats.packet_loss_rate,
            })

        if self._perf_stats:
            result.update({
                "rate_hz": self._perf_stats.effective_rate_hz,
                "dt_mean_ms": self._perf_stats.mean_dt_ms,
                "dt_std_ms": self._perf_stats.std_dt_ms,
                "loop_time_ms": self._perf_stats.mean_loop_time_ms,
                "dropped_samples": self._perf_stats.dropped_samples,
            })

        return result
