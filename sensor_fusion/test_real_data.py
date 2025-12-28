#!/usr/bin/env python3
"""Exhaustive validation tests for real IMU sensor data.

Run against recorded data to diagnose yaw drift and other issues.
Each test outputs detailed diagnostics for root cause analysis.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from data_recorder import DataRecording, RawDataSnapshot


# ============================================================================
# Test Results and Reporting
# ============================================================================

@dataclass
class TestResult:
    """Result of a single validation test."""
    name: str
    passed: bool
    message: str
    details: dict


class DiagnosticReport:
    """Collects and reports all test results."""

    def __init__(self, recording_file: str):
        self.recording_file = recording_file
        self.results: List[TestResult] = []

    def add(self, result: TestResult) -> None:
        self.results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.message}")
        if result.details and not result.passed:
            for key, value in result.details.items():
                print(f"       {key}: {value}")

    def summary(self) -> None:
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print(f"\n{'='*60}")
        print(f"SUMMARY: {passed}/{total} tests passed")
        print(f"{'='*60}")

        if passed < total:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}")


# ============================================================================
# Test Suite 1: NED Frame Consistency
# ============================================================================

def test_ned_frame_compliance(recording: DataRecording) -> TestResult:
    """Verify sensor data complies with NED frame conventions.

    In NED frame when horizontal and stationary:
    - az should be positive (~+9.81 m/s^2, gravity points down)
    - ax, ay should be near zero
    - roll, pitch should be near zero
    """
    snapshots = recording.snapshots
    if not snapshots:
        return TestResult("NED Frame", False, "No data", {})

    az_values = [s.az for s in snapshots]
    ax_values = [s.ax for s in snapshots]
    ay_values = [s.ay for s in snapshots]

    az_mean = np.mean(az_values)
    ax_mean = np.mean(ax_values)
    ay_mean = np.mean(ay_values)

    # Check az is positive (NED: gravity = +Z)
    az_positive = az_mean > 0
    az_correct_magnitude = 9.0 < az_mean < 10.5

    # Check ax, ay are small
    ax_small = abs(ax_mean) < 1.0
    ay_small = abs(ay_mean) < 1.0

    passed = az_positive and az_correct_magnitude and ax_small and ay_small

    details = {
        "az_mean": f"{az_mean:.3f} m/s^2 (expect +9.81)",
        "ax_mean": f"{ax_mean:.3f} m/s^2 (expect ~0)",
        "ay_mean": f"{ay_mean:.3f} m/s^2 (expect ~0)",
        "az_positive": az_positive,
        "issue": "" if passed else (
            "az negative - Z axis inverted!" if not az_positive else
            "az magnitude wrong - check calibration" if not az_correct_magnitude else
            "ax/ay too large - sensor tilted or axis error"
        ),
    }

    msg = f"az={az_mean:.2f}, ax={ax_mean:.2f}, ay={ay_mean:.2f}"
    return TestResult("NED Frame Compliance", passed, msg, details)


def test_magnetometer_frame(recording: DataRecording) -> TestResult:
    """Verify magnetometer readings are consistent with expected Earth field.

    Earth's field in France: ~47 uT total, ~20 uT horizontal, ~42 uT vertical
    In NED: mz should be positive (field points into ground in northern hemisphere)
    """
    snapshots = recording.snapshots
    if not snapshots:
        return TestResult("Mag Frame", False, "No data", {})

    mx_values = [s.mx for s in snapshots]
    my_values = [s.my for s in snapshots]
    mz_values = [s.mz for s in snapshots]
    mag_values = [s.mag_mag for s in snapshots]

    mx_mean = np.mean(mx_values)
    my_mean = np.mean(my_values)
    mz_mean = np.mean(mz_values)
    mag_mean = np.mean(mag_values)

    # Earth field range: 25-65 uT
    mag_valid = 20 < mag_mean < 70

    # In northern hemisphere, vertical component should be positive (into ground)
    mz_positive = mz_mean > 0

    # Horizontal component should be significant
    horiz_mag = np.sqrt(mx_mean**2 + my_mean**2)
    horiz_significant = horiz_mag > 10

    passed = mag_valid and mz_positive and horiz_significant

    details = {
        "mag_total": f"{mag_mean:.1f} uT (expect 25-65)",
        "mx_mean": f"{mx_mean:.1f} uT",
        "my_mean": f"{my_mean:.1f} uT",
        "mz_mean": f"{mz_mean:.1f} uT (expect >0 in N. hemisphere)",
        "horiz_mag": f"{horiz_mag:.1f} uT",
        "issue": "" if passed else (
            "Magnitude out of Earth field range" if not mag_valid else
            "mz negative - Z axis or hemisphere issue" if not mz_positive else
            "Horizontal component too small"
        ),
    }

    msg = f"||m||={mag_mean:.1f} uT, mz={mz_mean:.1f}"
    return TestResult("Magnetometer Frame", passed, msg, details)


# ============================================================================
# Test Suite 2: Temporal Consistency
# ============================================================================

def test_sampling_timing(recording: DataRecording) -> TestResult:
    """Analyze timing characteristics.

    Check:
    - dt consistency (expect ~10ms for 100Hz)
    - Jitter (should be < 5ms)
    - No gaps or negative dt
    """
    snapshots = recording.snapshots
    if len(snapshots) < 10:
        return TestResult("Timing", False, "Insufficient data", {})

    dt_values = [s.dt for s in snapshots[1:]]  # Skip first (init)

    dt_mean = np.mean(dt_values) * 1000  # ms
    dt_std = np.std(dt_values) * 1000
    dt_max = np.max(dt_values) * 1000
    dt_min = np.min(dt_values) * 1000

    # Expected ~10ms for 100Hz
    dt_reasonable = 5 < dt_mean < 20
    jitter_ok = dt_std < 5
    no_gaps = dt_max < 50
    positive_dt = dt_min > 0

    actual_rate = 1000 / dt_mean if dt_mean > 0 else 0

    passed = dt_reasonable and jitter_ok and no_gaps and positive_dt

    details = {
        "dt_mean": f"{dt_mean:.2f} ms",
        "dt_std": f"{dt_std:.2f} ms (jitter)",
        "dt_max": f"{dt_max:.2f} ms",
        "dt_min": f"{dt_min:.2f} ms",
        "actual_rate": f"{actual_rate:.1f} Hz",
        "issue": "" if passed else (
            "Negative dt detected!" if not positive_dt else
            "Large gaps (>50ms)" if not no_gaps else
            "High jitter" if not jitter_ok else
            "Unexpected sample rate"
        ),
    }

    msg = f"dt={dt_mean:.1f}+/-{dt_std:.1f}ms, rate={actual_rate:.0f}Hz"
    return TestResult("Sampling Timing", passed, msg, details)


def test_sequence_continuity(recording: DataRecording) -> TestResult:
    """Check for dropped packets via sequence numbers."""
    snapshots = recording.snapshots
    if len(snapshots) < 10:
        return TestResult("Sequence", False, "Insufficient data", {})

    gaps = []
    for i in range(1, len(snapshots)):
        expected = (snapshots[i-1].seq + 1) % (2**32)
        actual = snapshots[i].seq
        if actual != expected:
            gap = (actual - snapshots[i-1].seq) % (2**32)
            gaps.append(gap)

    total_gaps = sum(gaps)
    gap_count = len(gaps)
    loss_rate = total_gaps / len(snapshots) * 100 if snapshots else 0

    passed = loss_rate < 1.0  # Less than 1% packet loss

    details = {
        "total_samples": len(snapshots),
        "gap_events": gap_count,
        "total_missed": total_gaps,
        "loss_rate": f"{loss_rate:.2f}%",
    }

    msg = f"{gap_count} gaps, {total_gaps} missed ({loss_rate:.1f}%)"
    return TestResult("Sequence Continuity", passed, msg, details)


# ============================================================================
# Test Suite 3: Sensor Plausibility
# ============================================================================

def test_accelerometer_stationary(recording: DataRecording) -> TestResult:
    """Validate accelerometer readings for stationary sensor.

    Stationary expectations:
    - Magnitude: 9.5 < ||a|| < 10.1 m/s^2
    - Low noise: std < 0.1 m/s^2 per axis
    """
    snapshots = recording.snapshots
    if not snapshots:
        return TestResult("Accel Stationary", False, "No data", {})

    acc_mags = [s.acc_mag for s in snapshots]
    ax_values = [s.ax for s in snapshots]
    ay_values = [s.ay for s in snapshots]
    az_values = [s.az for s in snapshots]

    mag_mean = np.mean(acc_mags)
    mag_std = np.std(acc_mags)

    ax_std = np.std(ax_values)
    ay_std = np.std(ay_values)
    az_std = np.std(az_values)

    mag_valid = 9.5 < mag_mean < 10.1
    noise_low = max(ax_std, ay_std, az_std) < 0.2

    passed = mag_valid and noise_low

    details = {
        "acc_magnitude": f"{mag_mean:.3f} +/- {mag_std:.3f} m/s^2",
        "ax_std": f"{ax_std:.4f} m/s^2",
        "ay_std": f"{ay_std:.4f} m/s^2",
        "az_std": f"{az_std:.4f} m/s^2",
        "issue": "" if passed else (
            "Magnitude not ~9.81" if not mag_valid else
            "High noise - sensor not stationary?"
        ),
    }

    msg = f"||a||={mag_mean:.2f}, noise={max(ax_std,ay_std,az_std):.3f}"
    return TestResult("Accelerometer (stationary)", passed, msg, details)


def test_gyroscope_stationary(recording: DataRecording) -> TestResult:
    """Validate gyroscope readings for stationary sensor.

    Stationary expectations:
    - Mean near zero: |mean| < 0.01 rad/s per axis
    - Low noise: std < 0.01 rad/s per axis
    - No spikes > 0.1 rad/s
    """
    snapshots = recording.snapshots
    if not snapshots:
        return TestResult("Gyro Stationary", False, "No data", {})

    gx_values = [s.gx for s in snapshots]
    gy_values = [s.gy for s in snapshots]
    gz_values = [s.gz for s in snapshots]

    gx_mean = np.mean(gx_values)
    gy_mean = np.mean(gy_values)
    gz_mean = np.mean(gz_values)

    gx_std = np.std(gx_values)
    gy_std = np.std(gy_values)
    gz_std = np.std(gz_values)

    gx_max = np.max(np.abs(gx_values))
    gy_max = np.max(np.abs(gy_values))
    gz_max = np.max(np.abs(gz_values))

    # Check bias (mean should be small)
    bias_ok = max(abs(gx_mean), abs(gy_mean), abs(gz_mean)) < 0.02

    # Check noise
    noise_ok = max(gx_std, gy_std, gz_std) < 0.02

    # Check for spikes
    no_spikes = max(gx_max, gy_max, gz_max) < 0.2

    passed = bias_ok and noise_ok and no_spikes

    # Convert to deg/s for display
    details = {
        "gx_mean": f"{np.rad2deg(gx_mean):.3f} deg/s",
        "gy_mean": f"{np.rad2deg(gy_mean):.3f} deg/s",
        "gz_mean": f"{np.rad2deg(gz_mean):.3f} deg/s",
        "gx_std": f"{np.rad2deg(gx_std):.3f} deg/s",
        "gy_std": f"{np.rad2deg(gy_std):.3f} deg/s",
        "gz_std": f"{np.rad2deg(gz_std):.3f} deg/s",
        "max_rate": f"{np.rad2deg(max(gx_max,gy_max,gz_max)):.2f} deg/s",
        "issue": "" if passed else (
            "High gyro bias" if not bias_ok else
            "High noise" if not noise_ok else
            "Spikes detected"
        ),
    }

    msg = f"bias={np.rad2deg(max(abs(gx_mean),abs(gy_mean),abs(gz_mean))):.2f} deg/s"
    return TestResult("Gyroscope (stationary)", passed, msg, details)


def test_magnetometer_stability(recording: DataRecording) -> TestResult:
    """Check magnetometer stability over time.

    Stable expectations:
    - Magnitude std < 2 uT
    - No sudden jumps > 5 uT between samples
    """
    snapshots = recording.snapshots
    if len(snapshots) < 10:
        return TestResult("Mag Stability", False, "Insufficient data", {})

    mag_mags = [s.mag_mag for s in snapshots]
    mag_std = np.std(mag_mags)

    # Check for jumps
    mag_diffs = np.abs(np.diff(mag_mags))
    max_jump = np.max(mag_diffs)
    jump_count = np.sum(mag_diffs > 5)

    stable = mag_std < 3
    no_jumps = max_jump < 10

    passed = stable and no_jumps

    details = {
        "mag_std": f"{mag_std:.2f} uT",
        "max_jump": f"{max_jump:.2f} uT",
        "jump_events": jump_count,
        "issue": "" if passed else (
            "High magnitude variance" if not stable else
            f"Sudden jumps detected ({jump_count} events)"
        ),
    }

    msg = f"std={mag_std:.1f} uT, max_jump={max_jump:.1f} uT"
    return TestResult("Magnetometer Stability", passed, msg, details)


# ============================================================================
# Test Suite 4: Yaw Drift Analysis (Critical)
# ============================================================================

def test_yaw_drift_characterization(recording: DataRecording) -> TestResult:
    """Analyze yaw drift patterns to identify root cause.

    This is the critical diagnostic for the 80 deg/10s drift issue.
    """
    snapshots = recording.snapshots
    if len(snapshots) < 100:
        return TestResult("Yaw Drift", False, "Insufficient data", {})

    yaw_values = [s.yaw for s in snapshots]
    timestamps = [s.timestamp for s in snapshots]

    # Unwrap yaw to handle 360->0 transitions
    yaw_unwrapped = np.unwrap(np.deg2rad(yaw_values))
    yaw_unwrapped = np.rad2deg(yaw_unwrapped)

    # Total drift
    yaw_start = yaw_unwrapped[0]
    yaw_end = yaw_unwrapped[-1]
    total_drift = yaw_end - yaw_start

    duration = timestamps[-1] - timestamps[0]
    drift_rate = total_drift / duration * 60  # deg/min

    # Check if drift is linear (constant rate) or accelerating
    mid_idx = len(yaw_unwrapped) // 2
    drift_first_half = yaw_unwrapped[mid_idx] - yaw_unwrapped[0]
    drift_second_half = yaw_unwrapped[-1] - yaw_unwrapped[mid_idx]

    drift_ratio = abs(drift_second_half / drift_first_half) if abs(drift_first_half) > 0.1 else 1.0

    # Drift pattern analysis
    if abs(drift_rate) < 5:
        pattern = "MINIMAL - acceptable drift"
    elif 0.8 < drift_ratio < 1.2:
        pattern = "LINEAR - suggests constant gyro bias"
    elif drift_ratio > 1.5:
        pattern = "ACCELERATING - suggests EKF divergence"
    else:
        pattern = "DECELERATING - suggests convergence"

    # Roll/pitch stability check
    roll_values = [s.roll for s in snapshots]
    pitch_values = [s.pitch for s in snapshots]
    roll_drift = np.max(roll_values) - np.min(roll_values)
    pitch_drift = np.max(pitch_values) - np.min(pitch_values)

    yaw_only = roll_drift < 5 and pitch_drift < 5 and abs(total_drift) > 10

    passed = abs(drift_rate) < 30  # Less than 30 deg/min

    details = {
        "total_drift": f"{total_drift:.1f} deg over {duration:.1f}s",
        "drift_rate": f"{drift_rate:.1f} deg/min",
        "pattern": pattern,
        "drift_ratio": f"{drift_ratio:.2f} (1.0=linear)",
        "roll_range": f"{roll_drift:.1f} deg",
        "pitch_range": f"{pitch_drift:.1f} deg",
        "yaw_only_issue": yaw_only,
        "diagnosis": (
            "Yaw-only drift with stable roll/pitch -> magnetometer issue"
            if yaw_only else
            "Multi-axis drift -> general EKF or sensor issue"
        ) if not passed else "Drift acceptable",
    }

    msg = f"{total_drift:.0f} deg in {duration:.0f}s ({drift_rate:.0f} deg/min)"
    return TestResult("Yaw Drift Analysis", passed, msg, details)


def test_gyro_integration_vs_magnetometer(recording: DataRecording) -> TestResult:
    """Compare gyro-integrated heading with magnetometer heading.

    If they diverge, indicates mag calibration or frame issue.
    """
    snapshots = recording.snapshots
    if len(snapshots) < 100:
        return TestResult("Gyro vs Mag", False, "Insufficient data", {})

    # Integrate gz to get gyro-only yaw
    gyro_yaw = 0.0
    gyro_yaws = [0.0]
    for s in snapshots[1:]:
        gyro_yaw += np.rad2deg(s.gz) * s.dt
        gyro_yaws.append(gyro_yaw)

    # Get magnetometer-derived yaw from EKF
    ekf_yaws = [s.yaw for s in snapshots]

    # Unwrap both
    gyro_yaws = np.rad2deg(np.unwrap(np.deg2rad(gyro_yaws)))
    ekf_yaws_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(ekf_yaws)))

    # Compare drift rates
    duration = snapshots[-1].timestamp - snapshots[0].timestamp
    gyro_drift = gyro_yaws[-1] - gyro_yaws[0]
    ekf_drift = ekf_yaws_unwrapped[-1] - ekf_yaws_unwrapped[0]

    gyro_rate = gyro_drift / duration * 60
    ekf_rate = ekf_drift / duration * 60

    # If gyro drift is small but EKF drift is large, mag is pulling wrong
    gyro_small = abs(gyro_rate) < 10
    ekf_large = abs(ekf_rate) > 30

    passed = not (gyro_small and ekf_large)

    details = {
        "gyro_integrated_drift": f"{gyro_drift:.1f} deg ({gyro_rate:.1f} deg/min)",
        "ekf_yaw_drift": f"{ekf_drift:.1f} deg ({ekf_rate:.1f} deg/min)",
        "diagnosis": (
            "MAGNETOMETER PULLING YAW WRONG - check mag calibration/axes!"
            if gyro_small and ekf_large else
            "Gyro bias causing drift" if abs(gyro_rate) > 30 else
            "Normal behavior"
        ),
    }

    msg = f"gyro={gyro_rate:.0f} deg/min, ekf={ekf_rate:.0f} deg/min"
    return TestResult("Gyro vs Magnetometer", passed, msg, details)


# ============================================================================
# Test Suite 5: Quaternion Validity
# ============================================================================

def test_quaternion_norm(recording: DataRecording) -> TestResult:
    """Verify quaternion remains normalized throughout."""
    snapshots = recording.snapshots
    if not snapshots:
        return TestResult("Quaternion Norm", False, "No data", {})

    q_norms = [s.q_norm for s in snapshots]
    norm_mean = np.mean(q_norms)
    norm_std = np.std(q_norms)
    norm_min = np.min(q_norms)
    norm_max = np.max(q_norms)

    # Check for NaN/Inf
    has_nan = any(np.isnan(s.qw) or np.isnan(s.qx) or
                  np.isnan(s.qy) or np.isnan(s.qz) for s in snapshots)

    norm_ok = 0.99 < norm_min and norm_max < 1.01

    passed = norm_ok and not has_nan

    details = {
        "norm_mean": f"{norm_mean:.6f}",
        "norm_std": f"{norm_std:.6f}",
        "norm_range": f"[{norm_min:.6f}, {norm_max:.6f}]",
        "has_nan": has_nan,
    }

    msg = f"norm={norm_mean:.4f} +/- {norm_std:.6f}"
    return TestResult("Quaternion Norm", passed, msg, details)


def test_euler_ranges(recording: DataRecording) -> TestResult:
    """Verify Euler angles are in valid ranges."""
    snapshots = recording.snapshots
    if not snapshots:
        return TestResult("Euler Ranges", False, "No data", {})

    roll_values = [s.roll for s in snapshots]
    pitch_values = [s.pitch for s in snapshots]
    yaw_values = [s.yaw for s in snapshots]

    roll_valid = all(-180 <= r <= 180 for r in roll_values)
    pitch_valid = all(-90 <= p <= 90 for p in pitch_values)
    yaw_valid = all(-180 <= y <= 180 for y in yaw_values)

    # Check for sudden jumps (gimbal lock symptoms)
    roll_jumps = np.max(np.abs(np.diff(roll_values)))
    pitch_jumps = np.max(np.abs(np.diff(pitch_values)))

    passed = roll_valid and pitch_valid and yaw_valid

    details = {
        "roll_range": f"[{np.min(roll_values):.1f}, {np.max(roll_values):.1f}]",
        "pitch_range": f"[{np.min(pitch_values):.1f}, {np.max(pitch_values):.1f}]",
        "yaw_range": f"[{np.min(yaw_values):.1f}, {np.max(yaw_values):.1f}]",
        "max_roll_jump": f"{roll_jumps:.1f} deg",
        "max_pitch_jump": f"{pitch_jumps:.1f} deg",
    }

    msg = "All Euler angles in valid ranges"
    return TestResult("Euler Angle Ranges", passed, msg, details)


# ============================================================================
# Main Runner
# ============================================================================

def run_all_tests(recording_file: str) -> DiagnosticReport:
    """Run all validation tests on a recording."""
    print(f"\nLoading: {recording_file}")
    recording = DataRecording.load(recording_file)

    print(f"Recording: {recording.sample_count} samples, {recording.duration_s:.1f}s")
    print(f"Description: {recording.description}")
    print(f"\n{'='*60}")
    print("RUNNING DIAGNOSTICS")
    print(f"{'='*60}\n")

    report = DiagnosticReport(recording_file)

    # Frame tests
    report.add(test_ned_frame_compliance(recording))
    report.add(test_magnetometer_frame(recording))

    # Timing tests
    report.add(test_sampling_timing(recording))
    report.add(test_sequence_continuity(recording))

    # Sensor tests
    report.add(test_accelerometer_stationary(recording))
    report.add(test_gyroscope_stationary(recording))
    report.add(test_magnetometer_stability(recording))

    # Critical yaw drift tests
    report.add(test_yaw_drift_characterization(recording))
    report.add(test_gyro_integration_vs_magnetometer(recording))

    # Quaternion tests
    report.add(test_quaternion_norm(recording))
    report.add(test_euler_ranges(recording))

    report.summary()
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run diagnostic tests on recorded IMU data")
    parser.add_argument("recording", type=str, help="Path to recording JSON file")
    args = parser.parse_args()

    if not Path(args.recording).exists():
        print(f"Error: File not found: {args.recording}")
        sys.exit(1)

    run_all_tests(args.recording)


if __name__ == "__main__":
    main()
