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
    
    # Initialisation du filtre
    estimator = AttitudeEstimator(k_q=100, k_b=200) 

    samples = []
    print("Initialisation (NE PAS BOUGER)...", file=sys.stderr)
    while len(samples) < 50:
        m = imu.read_measurement(timeout_s=0.5)
        if m: samples.append(m)

    acc_mean = np.mean([m["ax"], m["ay"], m["az"]], axis=1)
    mag_mean = np.mean([m["mx"], m["my"], m["mz"]], axis=1)

    # Initialiser le quaternion avec ILSA
    estimator.q = estimator._ilsa(acc_mean, mag_mean, num_iter=20)
    print("✓ Quaternion initialisé:", estimator.q.T)
    
    last_seq = 0
    print("✓ Filtre Fourati démarré.", file=sys.stderr)

    try:
        while True:
            m = imu.read_measurement(timeout_s=1.0)
            if m is None: continue

            # Gestion du temps (dt) [cite: 317]
            #dt = (m["seq"] - last_seq) * 0.010 if last_seq > 0 else 0.010
            #last_seq = m["seq"]

            # Calcul du dt avec timestamp réel
            current_time = time.time()
            if last_time is not None:
                dt = current_time - last_time
            else:
                dt = 0.01
            last_time = current_time

            # Données capteurs 
            acc = np.array([m["ax"], m["ay"], m["az"]]) # m/s^2
            gyr = np.array([m["gx"], m["gy"], m["gz"]]) # rad/s
            mag = np.array([m["mx"], m["my"], m["mz"]]) # uT

            # Mise à jour en ligne
            q = estimator.update(gyr, acc, mag, dt)

            # Affichage console pour débug
            sys.stderr.write(f"\rFourati -> qw={q[0,0]:.4f}, qx={q[1,0]:.4f}, qy={q[2,0]:.4f}, qz={q[3,0]:.4f}")
            sys.stderr.flush()
            sys.stderr.write(f"\rdt = {dt}")
            sys.stderr.flush()

            # Envoi JSON vers le serveur Web
            print(json.dumps({
                "qw": float(q[0,0]), "qx": float(q[1,0]), 
                "qy": float(q[2,0]), "qz": float(q[3,0]),
            }), flush=True)

    finally:
        imu.close()


if __name__ == "__main__":
    run_imu_fusion()