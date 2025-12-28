#!/usr/bin/env python3
"""Systematic root cause isolation for yaw drift.

Runs specific diagnostic tests to pinpoint the exact cause of drift.
"""

import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from data_recorder import DataRecording


def diagnose_recording(filepath: str) -> None:
    """Run comprehensive diagnostics on a recording."""
    print(f"\n{'='*70}")
    print("YAW DRIFT ROOT CAUSE ANALYSIS")
    print(f"{'='*70}")

    recording = DataRecording.load(filepath)
    snapshots = recording.snapshots

    if len(snapshots) < 100:
        print("ERROR: Insufficient data (need at least 100 samples)")
        return

    print(f"\nRecording: {len(snapshots)} samples, {recording.duration_s:.1f}s")
    print(f"Description: {recording.description}\n")

    # Run diagnostic steps
    step1_gyro_bias(snapshots)
    step2_mag_calibration(snapshots)
    step3_frame_alignment(snapshots)
    step4_timing_analysis(snapshots)
    step5_ekf_behavior(snapshots)

    # Generate plots
    generate_diagnostic_plots(snapshots, filepath)

    # Final verdict
    print_verdict(snapshots)


def step1_gyro_bias(snapshots) -> None:
    """Step 1: Analyze gyroscope bias."""
    print("\n" + "-"*60)
    print("STEP 1: GYROSCOPE BIAS ANALYSIS")
    print("-"*60)

    gx = [s.gx for s in snapshots]
    gy = [s.gy for s in snapshots]
    gz = [s.gz for s in snapshots]

    gx_mean = np.mean(gx)
    gy_mean = np.mean(gy)
    gz_mean = np.mean(gz)

    print(f"Gyro bias (mean values):")
    print(f"  gx: {np.rad2deg(gx_mean):+.4f} deg/s")
    print(f"  gy: {np.rad2deg(gy_mean):+.4f} deg/s")
    print(f"  gz: {np.rad2deg(gz_mean):+.4f} deg/s")

    # Integrate gyro to see how much drift it would cause alone
    duration = snapshots[-1].timestamp - snapshots[0].timestamp
    gyro_yaw_drift = sum(s.gz * s.dt for s in snapshots[1:])

    print(f"\nGyro-only yaw drift over {duration:.1f}s:")
    print(f"  Total: {np.rad2deg(gyro_yaw_drift):.1f} deg")
    print(f"  Rate: {np.rad2deg(gyro_yaw_drift)/duration*60:.1f} deg/min")

    if abs(np.rad2deg(gyro_yaw_drift)/duration*60) > 30:
        print("\n  >>> HIGH GYRO BIAS - This alone explains significant drift!")
        print("  >>> Recommendation: Calibrate gyro bias at startup")
    else:
        print("\n  Gyro bias is acceptable (< 30 deg/min from gyro alone)")


def step2_mag_calibration(snapshots) -> None:
    """Step 2: Analyze magnetometer calibration."""
    print("\n" + "-"*60)
    print("STEP 2: MAGNETOMETER CALIBRATION ANALYSIS")
    print("-"*60)

    mx = [s.mx for s in snapshots]
    my = [s.my for s in snapshots]
    mz = [s.mz for s in snapshots]
    mag_mags = [s.mag_mag for s in snapshots]

    print(f"Magnetometer statistics:")
    print(f"  mx: {np.mean(mx):+.1f} +/- {np.std(mx):.2f} uT")
    print(f"  my: {np.mean(my):+.1f} +/- {np.std(my):.2f} uT")
    print(f"  mz: {np.mean(mz):+.1f} +/- {np.std(mz):.2f} uT")
    print(f"  ||m||: {np.mean(mag_mags):.1f} +/- {np.std(mag_mags):.2f} uT")

    # Check for sphere vs ellipsoid (hard/soft iron)
    mx_range = np.max(mx) - np.min(mx)
    my_range = np.max(my) - np.min(my)
    mz_range = np.max(mz) - np.min(mz)

    print(f"\nMagnetic field variation (stationary):")
    print(f"  mx range: {mx_range:.2f} uT")
    print(f"  my range: {my_range:.2f} uT")
    print(f"  mz range: {mz_range:.2f} uT")

    # Compute heading from mag (horizontal projection)
    headings = []
    for s in snapshots:
        heading = np.arctan2(s.my, s.mx)
        headings.append(np.rad2deg(heading))

    heading_range = np.max(headings) - np.min(headings)
    heading_std = np.std(headings)

    print(f"\nMag-derived heading (stationary):")
    print(f"  Mean: {np.mean(headings):.1f} deg")
    print(f"  Std: {heading_std:.2f} deg")
    print(f"  Range: {heading_range:.1f} deg")

    if heading_std > 5:
        print("\n  >>> HIGH MAG HEADING VARIANCE - Magnetometer noise/disturbance!")
        print("  >>> This will cause EKF to hunt, creating apparent drift")
    else:
        print("\n  Mag heading variance acceptable")


def step3_frame_alignment(snapshots) -> None:
    """Step 3: Check frame alignment (NED compliance)."""
    print("\n" + "-"*60)
    print("STEP 3: FRAME ALIGNMENT CHECK")
    print("-"*60)

    ax_mean = np.mean([s.ax for s in snapshots])
    ay_mean = np.mean([s.ay for s in snapshots])
    az_mean = np.mean([s.az for s in snapshots])

    print(f"Accelerometer (should show gravity in +Z for NED):")
    print(f"  ax: {ax_mean:+.3f} m/s^2 (expect ~0)")
    print(f"  ay: {ay_mean:+.3f} m/s^2 (expect ~0)")
    print(f"  az: {az_mean:+.3f} m/s^2 (expect +9.81)")

    # Check which axis has gravity
    max_axis = max(abs(ax_mean), abs(ay_mean), abs(az_mean))
    if abs(az_mean) != max_axis:
        print("\n  >>> FRAME ERROR: Gravity not primarily on Z axis!")
        if abs(ax_mean) == max_axis:
            print("  >>> Gravity on X axis - 90 degree rotation needed")
        else:
            print("  >>> Gravity on Y axis - 90 degree rotation needed")
    elif az_mean < 0:
        print("\n  >>> FRAME ERROR: Z axis inverted (gravity should be +9.81)")
    else:
        print("\n  Frame alignment OK (gravity on +Z)")

    # Check magnetometer Z in northern hemisphere
    mz_mean = np.mean([s.mz for s in snapshots])
    print(f"\nMagnetometer Z (should be positive in N. hemisphere):")
    print(f"  mz: {mz_mean:+.1f} uT")

    if mz_mean < 0:
        print("\n  >>> WARNING: mz negative - check if in southern hemisphere")
        print("  >>> or magnetometer Z axis may be inverted")


def step4_timing_analysis(snapshots) -> None:
    """Step 4: Analyze timing for integration errors."""
    print("\n" + "-"*60)
    print("STEP 4: TIMING ANALYSIS")
    print("-"*60)

    dt_values = [s.dt for s in snapshots[1:]]

    dt_mean = np.mean(dt_values) * 1000
    dt_std = np.std(dt_values) * 1000
    dt_max = np.max(dt_values) * 1000
    dt_min = np.min(dt_values) * 1000

    print(f"Time step (dt) statistics:")
    print(f"  Mean: {dt_mean:.2f} ms")
    print(f"  Std: {dt_std:.2f} ms (jitter)")
    print(f"  Range: [{dt_min:.2f}, {dt_max:.2f}] ms")
    print(f"  Effective rate: {1000/dt_mean:.1f} Hz")

    # Count large gaps
    large_gaps = sum(1 for dt in dt_values if dt * 1000 > 20)
    print(f"  Large gaps (>20ms): {large_gaps}")

    if dt_std > 3:
        print("\n  >>> HIGH TIMING JITTER - May cause integration errors")
        print("  >>> Consider hardware timing improvements")
    else:
        print("\n  Timing jitter acceptable")


def step5_ekf_behavior(snapshots) -> None:
    """Step 5: Analyze EKF behavior over time."""
    print("\n" + "-"*60)
    print("STEP 5: EKF BEHAVIOR ANALYSIS")
    print("-"*60)

    # Get yaw, roll, pitch
    yaw = [s.yaw for s in snapshots]
    roll = [s.roll for s in snapshots]
    pitch = [s.pitch for s in snapshots]
    timestamps = [s.timestamp - snapshots[0].timestamp for s in snapshots]

    # Unwrap yaw
    yaw_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(yaw)))

    # Compute drift rates at different times
    n = len(yaw_unwrapped)
    if n > 200:
        # First quarter
        yaw_rate_1 = (yaw_unwrapped[n//4] - yaw_unwrapped[0]) / timestamps[n//4] * 60
        # Last quarter
        yaw_rate_4 = (yaw_unwrapped[-1] - yaw_unwrapped[3*n//4]) / (timestamps[-1] - timestamps[3*n//4]) * 60

        print(f"Yaw drift rate over time:")
        print(f"  First 25%: {yaw_rate_1:.1f} deg/min")
        print(f"  Last 25%: {yaw_rate_4:.1f} deg/min")

        if abs(yaw_rate_4) > abs(yaw_rate_1) * 1.5:
            print("\n  >>> DRIFT ACCELERATING - EKF may be diverging")
        elif abs(yaw_rate_4) < abs(yaw_rate_1) * 0.5:
            print("\n  >>> DRIFT DECELERATING - EKF may be converging")
        else:
            print("\n  >>> DRIFT CONSTANT - Likely gyro bias or mag issue")

    # Check roll/pitch stability
    roll_range = np.max(roll) - np.min(roll)
    pitch_range = np.max(pitch) - np.min(pitch)
    yaw_range = np.max(yaw_unwrapped) - np.min(yaw_unwrapped)

    print(f"\nAngle stability (peak-to-peak):")
    print(f"  Roll: {roll_range:.1f} deg")
    print(f"  Pitch: {pitch_range:.1f} deg")
    print(f"  Yaw: {yaw_range:.1f} deg")

    if roll_range < 5 and pitch_range < 5 and yaw_range > 20:
        print("\n  >>> YAW-ONLY DRIFT detected!")
        print("  >>> Roll/pitch stable = accelerometer working correctly")
        print("  >>> Yaw drifting = MAGNETOMETER IS THE PROBLEM")


def generate_diagnostic_plots(snapshots, filepath) -> None:
    """Generate diagnostic plots."""
    print("\n" + "-"*60)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("-"*60)

    timestamps = [s.timestamp - snapshots[0].timestamp for s in snapshots]
    yaw = [s.yaw for s in snapshots]
    roll = [s.roll for s in snapshots]
    pitch = [s.pitch for s in snapshots]
    yaw_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(yaw)))

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Plot 1: Euler angles over time
    ax = axes[0, 0]
    ax.plot(timestamps, roll, label='Roll', alpha=0.7)
    ax.plot(timestamps, pitch, label='Pitch', alpha=0.7)
    ax.plot(timestamps, yaw_unwrapped, label='Yaw (unwrapped)', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_title('Euler Angles Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Gyroscope readings
    ax = axes[0, 1]
    gx = [np.rad2deg(s.gx) for s in snapshots]
    gy = [np.rad2deg(s.gy) for s in snapshots]
    gz = [np.rad2deg(s.gz) for s in snapshots]
    ax.plot(timestamps, gx, label='gx', alpha=0.7)
    ax.plot(timestamps, gy, label='gy', alpha=0.7)
    ax.plot(timestamps, gz, label='gz', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular rate (deg/s)')
    ax.set_title('Gyroscope Readings')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Magnetometer magnitude
    ax = axes[1, 0]
    mag_mag = [s.mag_mag for s in snapshots]
    ax.plot(timestamps, mag_mag)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('||m|| (uT)')
    ax.set_title('Magnetometer Magnitude')
    ax.grid(True, alpha=0.3)
    ax.axhline(np.mean(mag_mag), color='r', linestyle='--', label=f'Mean: {np.mean(mag_mag):.1f}')
    ax.legend()

    # Plot 4: Magnetometer XY (heading)
    ax = axes[1, 1]
    mx = [s.mx for s in snapshots]
    my = [s.my for s in snapshots]
    ax.scatter(mx, my, c=timestamps, cmap='viridis', s=1, alpha=0.5)
    ax.set_xlabel('mx (uT)')
    ax.set_ylabel('my (uT)')
    ax.set_title('Magnetometer XY (color = time)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # Plot 5: dt distribution
    ax = axes[2, 0]
    dt_ms = [s.dt * 1000 for s in snapshots[1:]]
    ax.hist(dt_ms, bins=50, edgecolor='black')
    ax.set_xlabel('dt (ms)')
    ax.set_ylabel('Count')
    ax.set_title(f'Time Step Distribution (mean={np.mean(dt_ms):.2f}ms)')
    ax.axvline(np.mean(dt_ms), color='r', linestyle='--')
    ax.grid(True, alpha=0.3)

    # Plot 6: Gyro integration vs EKF yaw
    ax = axes[2, 1]
    gyro_yaw = [0.0]
    for s in snapshots[1:]:
        gyro_yaw.append(gyro_yaw[-1] + np.rad2deg(s.gz) * s.dt)
    gyro_yaw = np.array(gyro_yaw)
    # Subtract initial to align
    gyro_yaw = gyro_yaw - gyro_yaw[0]
    ekf_yaw = yaw_unwrapped - yaw_unwrapped[0]

    ax.plot(timestamps, gyro_yaw, label='Gyro integration', alpha=0.7)
    ax.plot(timestamps, ekf_yaw, label='EKF yaw', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle change (deg)')
    ax.set_title('Gyro Integration vs EKF Yaw')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = filepath.replace('.json', '_diagnostic.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved diagnostic plot: {plot_path}")
    plt.close()


def print_verdict(snapshots) -> None:
    """Print final diagnosis verdict."""
    print("\n" + "="*70)
    print("FINAL DIAGNOSIS")
    print("="*70)

    # Calculate key metrics
    yaw = [s.yaw for s in snapshots]
    yaw_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(yaw)))
    duration = snapshots[-1].timestamp - snapshots[0].timestamp
    total_drift = yaw_unwrapped[-1] - yaw_unwrapped[0]
    drift_rate = total_drift / duration * 60

    roll_range = np.max([s.roll for s in snapshots]) - np.min([s.roll for s in snapshots])
    pitch_range = np.max([s.pitch for s in snapshots]) - np.min([s.pitch for s in snapshots])

    gyro_yaw_drift = sum(np.rad2deg(s.gz) * s.dt for s in snapshots[1:])

    # Determine primary cause
    print(f"\nYaw drift: {total_drift:.1f} deg in {duration:.1f}s ({drift_rate:.1f} deg/min)")
    print(f"Gyro-only drift: {gyro_yaw_drift:.1f} deg")
    print(f"Roll/Pitch stability: {roll_range:.1f} / {pitch_range:.1f} deg")

    print("\n" + "-"*40)

    if abs(drift_rate) < 10:
        print("VERDICT: Drift is acceptable (<10 deg/min)")
        print("No action needed.")
    elif roll_range < 5 and pitch_range < 5:
        # Yaw-only issue
        if abs(gyro_yaw_drift) > abs(total_drift) * 0.8:
            print("VERDICT: HIGH GYRO BIAS is primary cause")
            print("\nRecommended fixes:")
            print("1. Implement gyro bias calibration at startup")
            print("2. Collect 1-2s of stationary data and subtract mean")
            print("3. Consider temperature compensation")
        else:
            print("VERDICT: MAGNETOMETER ISSUE is primary cause")
            print("\nRecommended fixes:")
            print("1. Verify magnetometer calibration (hard/soft iron)")
            print("2. Check magnetometer axis alignment with NED frame")
            print("3. Reduce EKF trust in magnetometer")
            print("4. Check for magnetic disturbances in environment")
    else:
        print("VERDICT: General EKF or sensor frame issue")
        print("\nRecommended fixes:")
        print("1. Verify sensor axis mapping to NED frame")
        print("2. Check EKF noise parameters")
        print("3. Verify sensor calibration")

    print("\n" + "="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose yaw drift from recorded data")
    parser.add_argument("recording", type=str, help="Path to recording JSON file")
    args = parser.parse_args()

    if not Path(args.recording).exists():
        print(f"Error: File not found: {args.recording}")
        sys.exit(1)

    diagnose_recording(args.recording)


if __name__ == "__main__":
    main()
