#!/usr/bin/env python3
"""Benchmark des latences EKF en conditions réelles"""

import time
import numpy as np
import imu_reader
from ekf import EKF

NUM_SAMPLES = 1000  # Nombre d'échantillons à mesurer


def benchmark():
    """Mesure les latences de chaque étape du pipeline"""

    timings = {
        "imu_read": [],
        "predict": [],
        "update": [],
        "total": [],
    }

    with imu_reader.ImuReader() as imu:
        # Initialisation
        m = imu.read()
        while m is None:
            m = imu.read()

        accel = np.array([m["ax"], m["ay"], m["az"]])
        ekf = EKF(accel_data=accel)

        print(f"Benchmark EKF - {NUM_SAMPLES} échantillons")
        print("-" * 50)

        last_time = time.time()
        count = 0

        while count < NUM_SAMPLES:
            t_start = time.perf_counter()

            # 1. Lecture IMU
            t0 = time.perf_counter()
            m = imu.read()
            t_imu = time.perf_counter() - t0

            if m is None:
                continue

            # Calcul dt
            now = time.time()
            dt = now - last_time
            last_time = now

            gyro = np.array([m["gx"], m["gy"], m["gz"]])
            accel = np.array([m["ax"], m["ay"], m["az"]])

            # 2. EKF Predict
            t0 = time.perf_counter()
            ekf.predict(gyro, dt)
            t_predict = time.perf_counter() - t0

            # 3. EKF Update
            t0 = time.perf_counter()
            ekf.update(accel)
            t_update = time.perf_counter() - t0

            # Total
            t_total = time.perf_counter() - t_start

            timings["imu_read"].append(t_imu * 1000)  # en ms
            timings["predict"].append(t_predict * 1000)
            timings["update"].append(t_update * 1000)
            timings["total"].append(t_total * 1000)

            count += 1

            if count % 100 == 0:
                print(f"  {count}/{NUM_SAMPLES}...")

    # Statistiques
    print("\n" + "=" * 50)
    print("RÉSULTATS (en ms)")
    print("=" * 50)

    for name, values in timings.items():
        arr = np.array(values)
        print(f"\n{name.upper()}:")
        print(f"  Min:    {arr.min():.3f} ms")
        print(f"  Max:    {arr.max():.3f} ms")
        print(f"  Mean:   {arr.mean():.3f} ms")
        print(f"  Median: {np.median(arr):.3f} ms")
        print(f"  Std:    {arr.std():.3f} ms")
        print(f"  P95:    {np.percentile(arr, 95):.3f} ms")
        print(f"  P99:    {np.percentile(arr, 99):.3f} ms")

    # Vérification budget temps
    print("\n" + "=" * 50)
    print("ANALYSE BUDGET TEMPS (objectif: <10ms)")
    print("=" * 50)

    total_arr = np.array(timings["total"])
    over_budget = np.sum(total_arr > 10.0)
    pct_over = 100.0 * over_budget / len(total_arr)

    print(f"Échantillons > 10ms: {over_budget}/{len(total_arr)} ({pct_over:.1f}%)")
    print(f"Fréquence max soutenue: {1000.0 / total_arr.mean():.1f} Hz")

    if total_arr.mean() < 10.0:
        print("\n✓ Budget temps respecté en moyenne")
    else:
        print("\n✗ Budget temps DÉPASSÉ - optimisation nécessaire")


if __name__ == "__main__":
    benchmark()
