import time
import json
import sys
import numpy as np
from imu_api import ImuSoftUart
from fourati import AttitudeEstimator


#

def run_imu_fusion():
    imu = ImuSoftUart(rx_gpio=24, baudrate=57600)
    imu.open()
    
    # GAINS RÉDUITS
    estimator = AttitudeEstimator(k_q=5.0, k_b=10.0)
    
    # Phase d'initialisation
    print("Initialisation (NE PAS BOUGER)...", file=sys.stderr)
    samples = []
    while len(samples) < 100:  # ← Plus d'échantillons (100 au lieu de 50)
        m = imu.read_measurement(timeout_s=0.5)
        if m: 
            samples.append(m)
            sys.stderr.write(f"\rÉchantillons: {len(samples)}/100")
            sys.stderr.flush()
    
    print("\n\n=== STATISTIQUES INITIALISATION ===", file=sys.stderr)
    
    # Calcul des moyennes ET écarts-types
    acc_samples = np.array([[s["ax"], s["ay"], s["az"]] for s in samples])
    mag_samples = np.array([[s["mx"], s["my"], s["mz"]] for s in samples])
    gyr_samples = np.array([[s["gx"], s["gy"], s["gz"]] for s in samples])
    
    acc_mean = np.mean(acc_samples, axis=0)
    acc_std = np.std(acc_samples, axis=0)
    mag_mean = np.mean(mag_samples, axis=0)
    mag_std = np.std(mag_samples, axis=0)
    gyr_mean = np.mean(gyr_samples, axis=0)
    gyr_std = np.std(gyr_samples, axis=0)
    
    print(f"Acc  : {acc_mean} ± {acc_std}", file=sys.stderr)
    print(f"Mag  : {mag_mean} ± {mag_std}", file=sys.stderr)
    print(f"Gyro : {gyr_mean} ± {gyr_std}", file=sys.stderr)
    print(f"Norm(acc): {np.linalg.norm(acc_mean):.3f} m/s² (attendu: ~9.81)", file=sys.stderr)
    print(f"Norm(mag): {np.linalg.norm(mag_mean):.3f} µT", file=sys.stderr)
    
    # IMPORTANT : Initialiser le biais du gyroscope avec la moyenne !
    estimator.bias = gyr_mean.reshape((3,1))
    print(f"Biais gyro initial: {estimator.bias.T}", file=sys.stderr)
    
    # Initialisation du quaternion avec ILSA
    estimator.q = estimator._ilsa(acc_mean, mag_mean, num_iter=20)
    print(f"Quaternion initial: {estimator.q.T}", file=sys.stderr)
    print("="*40 + "\n", file=sys.stderr)
    
    last_time = None
    iteration = 0
    
    try:
        while True:
            m = imu.read_measurement(timeout_s=1.0)
            if m is None: continue
            
            current_time = time.time()
            if last_time is not None:
                dt = current_time - last_time
            else:
                dt = 0.01
            last_time = current_time
            
            acc = np.array([m["ax"], m["ay"], m["az"]])
            gyr = np.array([m["gx"], m["gy"], m["gz"]])
            mag = np.array([m["mx"], m["my"], m["mz"]])
            
            q, bias = estimator.update(gyr, acc, mag, dt)
            
            # Affichage tous les 10 échantillons
            iteration += 1
            if iteration % 10 == 0:
                norm_q = np.linalg.norm(q)
                sys.stderr.write(f"\rq=[{q[0,0]:+.3f}, {q[1,0]:+.3f}, {q[2,0]:+.3f}, {q[3,0]:+.3f}] |q|={norm_q:.4f} bias={bias.T}")
                sys.stderr.flush()
            
            print(json.dumps({
                "qw": float(q[0,0]), "qx": float(q[1,0]), 
                "qy": float(q[2,0]), "qz": float(q[3,0]),
            }), flush=True)
            
    except KeyboardInterrupt:
        print("\nArrêt demandé", file=sys.stderr)
    finally:
        imu.close()
