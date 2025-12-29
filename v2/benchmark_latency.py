#!/usr/bin/env python3
"""
Diagnostic complet des latences dans la chaîne IMU -> EKF -> Output

Mesure chaque étape:
1. Lecture série (temps d'attente buffer)
2. Parsing paquet
3. EKF predict
4. EKF update
5. JSON serialization
6. stdout write

Identifie les goulots d'étranglement.
"""

import sys
import time
import json
import numpy as np

# La validation IMU n'est plus automatique, pas besoin de la désactiver
from imu_reader import ImuReader
from ekf import EKF

NUM_SAMPLES = 500
WARMUP = 50


class LatencyProfiler:
    def __init__(self):
        self.stages = {}

    def record(self, stage: str, duration_ms: float):
        if stage not in self.stages:
            self.stages[stage] = []
        self.stages[stage].append(duration_ms)

    def report(self):
        print("\n" + "=" * 70)
        print("PROFIL DE LATENCE PAR ÉTAPE (ms)")
        print("=" * 70)

        total_mean = 0

        for name, values in self.stages.items():
            arr = np.array(values)
            mean = arr.mean()
            total_mean += mean

            bar_len = int(mean * 10)  # 1 char = 0.1ms
            bar = "█" * min(bar_len, 50)

            print(f"\n{name}:")
            print(f"  {bar} {mean:.3f}ms")
            print(f"  min={arr.min():.3f} max={arr.max():.3f} std={arr.std():.3f} p99={np.percentile(arr, 99):.3f}")

        print("\n" + "-" * 70)
        print(f"TOTAL MOYEN: {total_mean:.3f} ms → {1000/total_mean:.1f} Hz max")
        print("-" * 70)


def benchmark_imu_read_raw():
    """Test 1: Temps de lecture brute du port série"""
    print("\n[TEST 1] Latence lecture série brute")

    import serial
    import struct

    SYNC1, SYNC2 = 0xAA, 0x55
    PACKET_SIZE = 46

    ser = serial.Serial("/dev/ttyS0", 115200, timeout=0.1)
    ser.reset_input_buffer()

    wait_times = []
    parse_times = []

    for i in range(NUM_SAMPLES + WARMUP):
        # Temps d'attente pour données disponibles
        t0 = time.perf_counter()
        while ser.in_waiting < 2 + PACKET_SIZE:
            pass  # Busy wait
        t_wait = (time.perf_counter() - t0) * 1000

        # Temps de lecture + parsing
        t0 = time.perf_counter()
        data = ser.read(ser.in_waiting)
        # Chercher sync
        idx = data.find(bytes([SYNC1, SYNC2]))
        if idx >= 0 and len(data) >= idx + 2 + PACKET_SIZE:
            payload = data[idx+2:idx+2+PACKET_SIZE]
        t_parse = (time.perf_counter() - t0) * 1000

        if i >= WARMUP:
            wait_times.append(t_wait)
            parse_times.append(t_parse)

    ser.close()

    wait_arr = np.array(wait_times)
    parse_arr = np.array(parse_times)

    print(f"  Attente données: mean={wait_arr.mean():.3f}ms max={wait_arr.max():.3f}ms")
    print(f"  Parse paquet:    mean={parse_arr.mean():.3f}ms max={parse_arr.max():.3f}ms")

    return wait_arr.mean(), parse_arr.mean()


def benchmark_imu_reader_class():
    """Test 2: Latence de ImuReader.read()"""
    print("\n[TEST 2] Latence ImuReader.read()")

    read_times = []

    with ImuReader() as imu:
        for i in range(NUM_SAMPLES + WARMUP):
            t0 = time.perf_counter()
            m = imu.read(timeout=0.5)
            t_read = (time.perf_counter() - t0) * 1000

            if m is not None and i >= WARMUP:
                read_times.append(t_read)

    arr = np.array(read_times)
    print(f"  ImuReader.read(): mean={arr.mean():.3f}ms max={arr.max():.3f}ms p99={np.percentile(arr, 99):.3f}ms")

    return arr.mean()


def benchmark_ekf():
    """Test 3: Latence EKF predict + update"""
    print("\n[TEST 3] Latence EKF")

    # Données simulées
    gyro = np.array([0.01, 0.02, 0.0])
    accel = np.array([0.1, 0.2, -9.8])
    dt = 0.01

    ekf = EKF(accel_data=accel)

    predict_times = []
    update_times = []

    for i in range(NUM_SAMPLES + WARMUP):
        t0 = time.perf_counter()
        ekf.predict(gyro, dt)
        t_predict = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        ekf.update(accel)
        t_update = (time.perf_counter() - t0) * 1000

        if i >= WARMUP:
            predict_times.append(t_predict)
            update_times.append(t_update)

    pred_arr = np.array(predict_times)
    upd_arr = np.array(update_times)

    print(f"  EKF.predict(): mean={pred_arr.mean():.3f}ms max={pred_arr.max():.3f}ms")
    print(f"  EKF.update():  mean={upd_arr.mean():.3f}ms max={upd_arr.max():.3f}ms")

    return pred_arr.mean(), upd_arr.mean()


def benchmark_json_output():
    """Test 4: Latence sérialisation JSON + stdout"""
    print("\n[TEST 4] Latence JSON + stdout")

    data = {"qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

    json_times = []
    write_times = []

    # Rediriger stdout vers /dev/null pour mesurer le temps d'écriture réel
    import os
    devnull = open(os.devnull, 'w')

    for i in range(NUM_SAMPLES + WARMUP):
        t0 = time.perf_counter()
        s = json.dumps(data)
        t_json = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        devnull.write(s + "\n")
        devnull.flush()
        t_write = (time.perf_counter() - t0) * 1000

        if i >= WARMUP:
            json_times.append(t_json)
            write_times.append(t_write)

    devnull.close()

    json_arr = np.array(json_times)
    write_arr = np.array(write_times)

    print(f"  json.dumps():  mean={json_arr.mean():.3f}ms max={json_arr.max():.3f}ms")
    print(f"  stdout write:  mean={write_arr.mean():.3f}ms max={write_arr.max():.3f}ms")

    return json_arr.mean(), write_arr.mean()


def benchmark_full_pipeline():
    """Test 5: Pipeline complet avec timestamps"""
    print("\n[TEST 5] Pipeline complet IMU → EKF → JSON")

    profiler = LatencyProfiler()

    with ImuReader() as imu:
        # Init
        m = imu.read()
        while m is None:
            m = imu.read()

        accel = np.array([m["ax"], m["ay"], m["az"]])
        ekf = EKF(accel_data=accel)

        last_time = time.time()
        count = 0

        # Mesure de la latence bout-en-bout
        loop_times = []

        while count < NUM_SAMPLES + WARMUP:
            t_loop_start = time.perf_counter()

            # 1. IMU Read
            t0 = time.perf_counter()
            m = imu.read(timeout=0.1)
            t_imu = (time.perf_counter() - t0) * 1000

            if m is None:
                continue

            # 2. Data prep
            t0 = time.perf_counter()
            now = time.time()
            dt = now - last_time
            last_time = now
            gyro = np.array([m["gx"], m["gy"], m["gz"]])
            accel = np.array([m["ax"], m["ay"], m["az"]])
            t_prep = (time.perf_counter() - t0) * 1000

            # 3. EKF Predict
            t0 = time.perf_counter()
            ekf.predict(gyro, dt)
            t_predict = (time.perf_counter() - t0) * 1000

            # 4. EKF Update
            t0 = time.perf_counter()
            ekf.update(accel)
            t_update = (time.perf_counter() - t0) * 1000

            # 5. JSON
            t0 = time.perf_counter()
            q = ekf.state[0:4]
            data = {"qw": float(q[0]), "qx": float(q[1]), "qy": float(q[2]), "qz": float(q[3])}
            s = json.dumps(data)
            t_json = (time.perf_counter() - t0) * 1000

            t_loop = (time.perf_counter() - t_loop_start) * 1000

            if count >= WARMUP:
                profiler.record("1_imu_read", t_imu)
                profiler.record("2_data_prep", t_prep)
                profiler.record("3_ekf_predict", t_predict)
                profiler.record("4_ekf_update", t_update)
                profiler.record("5_json_dumps", t_json)
                profiler.record("6_LOOP_TOTAL", t_loop)
                loop_times.append(t_loop)

            count += 1

    profiler.report()

    # Analyse des outliers
    loop_arr = np.array(loop_times)
    outliers = loop_arr[loop_arr > 15]  # > 15ms = problème
    if len(outliers) > 0:
        print(f"\n⚠ {len(outliers)} itérations > 15ms détectées!")
        print(f"  Valeurs: {outliers[:10]}...")


def benchmark_sleep_accuracy():
    """Test 6: Précision du sleep Python"""
    print("\n[TEST 6] Précision time.sleep()")

    target_ms = 10  # 100 Hz
    actual_times = []

    for i in range(100):
        t0 = time.perf_counter()
        time.sleep(target_ms / 1000)
        actual = (time.perf_counter() - t0) * 1000
        actual_times.append(actual)

    arr = np.array(actual_times)
    overshoot = arr - target_ms

    print(f"  Target: {target_ms}ms")
    print(f"  Actual: mean={arr.mean():.3f}ms std={arr.std():.3f}ms")
    print(f"  Overshoot: mean={overshoot.mean():.3f}ms max={overshoot.max():.3f}ms")

    if overshoot.mean() > 1:
        print("  ⚠ Sleep imprécis - peut causer accumulation de latence!")


def main():
    print("=" * 70)
    print("DIAGNOSTIC LATENCE - CHAÎNE IMU → VISUALISATION")
    print("=" * 70)

    try:
        benchmark_sleep_accuracy()
    except Exception as e:
        print(f"  Skip: {e}")

    try:
        benchmark_ekf()
    except Exception as e:
        print(f"  Skip: {e}")

    try:
        benchmark_json_output()
    except Exception as e:
        print(f"  Skip: {e}")

    try:
        benchmark_imu_reader_class()
    except Exception as e:
        print(f"  Skip: {e}")

    try:
        benchmark_full_pipeline()
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("RECOMMANDATIONS")
    print("=" * 70)
    print("""
Si latence > 10ms:
  - imu_read élevé     → Buffer série, réduire timeout, augmenter baudrate
  - ekf_predict élevé  → Optimiser matrices NumPy
  - ekf_update élevé   → Simplifier mesure Jacobian
  - json_dumps élevé   → Utiliser format binaire
  - sleep imprécis     → Utiliser busy-wait ou supprimer sleep
""")


if __name__ == "__main__":
    main()
