import time
import json
import sys
import numpy as np
from ahrs.filters import Fourati
from ahrs.common.orientation import ecompass
from imu_api import ImuSoftUart
from scipy.spatial.transform import Rotation as R

# ------------------------------------------------------------
# Configuration Locale (Identique à tes paramètres)
# ------------------------------------------------------------
MAG_INCLINATION = 60.2  # DIP angle deg
MAG_DECLINATION = 2.7   # Déclinaison deg

def quaternion_to_euler(q):
    """ Convention AHRS: [w, x, y, z] -> Scipy: [x, y, z, w] """
    q_scipy = [q[1], q[2], q[3], q[0]]
    r = R.from_quat(q_scipy)
    return r.as_euler('zyx', degrees=False)

def run_imu_fusion():
    imu = ImuSoftUart(rx_gpio=24, baudrate=57600)
    imu.open()

    # 1. Collecte d'initialisation pour le biais du Gyro
    samples = []
    print("Initialisation (NE PAS BOUGER)...", file=sys.stderr)
    while len(samples) < 50:
        m = imu.read_measurement(timeout_s=0.5)
        if m: samples.append(m)
    
    # Calcul du biais moyen du gyroscope
    gyro_bias = np.mean([[s["gx"], s["gy"], s["gz"]] for s in samples], axis=0)
    
    # Récupération des premières valeurs pour l'orientation initiale
    acc0 = np.mean([[s["ax"], s["ay"], s["az"]] for s in samples], axis=0)
    mag0 = np.mean([[s["mx"], s["my"], s["mz"]] for s in samples], axis=0)

    # 2. Initialisation du filtre de Fourati
    # On passe l'inclinaison magnétique (magnetic_dip)
    fourati = Fourati(
        frequency=100.0, 
        gain=0.2, 
        magnetic_dip=MAG_INCLINATION
    )

    # Calcul du quaternion initial (Convention NED pour Fourati)
    q = ecompass(acc0, mag0, frame='NED', representation='quaternion')
    
    last_seq = 0
    print("✓ Filtre Fourati démarré.", file=sys.stderr)

    try:
        while True:
            m = imu.read_measurement(timeout_s=1.0)
            if m is None: continue

            # Gestion du temps (dt)
            dt = (m["seq"] - last_seq) * 0.010 if last_seq > 0 else 0.010
            last_seq = m["seq"]

            # Préparation des données
            acc = np.array([m["ax"], m["ay"], m["az"]])
            # On retire le biais du gyro
            gyr = np.array([m["gx"], m["gy"], m["gz"]]) - gyro_bias
            mag = np.array([m["mx"], m["my"], m["mz"]])

            # --- MISE À JOUR FOURATI ---
            # L'algorithme update prend q_prec, gyr, acc, mag et dt
            q = fourati.update(q, gyr, acc, mag, dt=dt)

            # Conversion en Euler pour le monitoring
            yaw, pitch, roll = quaternion_to_euler(q)

            # Affichage console pour débug
            sys.stderr.write(f"\rFourati -> R:{np.rad2deg(roll):.1f}° P:{np.rad2deg(pitch):.1f}° Y:{np.rad2deg(yaw):.1f}°")
            sys.stderr.flush()

            # Envoi JSON vers le serveur Web
            print(json.dumps({
                "qx": float(q[1]), "qy": float(q[2]), "qz": float(q[3]), "qw": float(q[0]),
                "roll": float(np.rad2deg(roll)),
                "pitch": float(np.rad2deg(pitch)),
                "yaw": float(np.rad2deg(yaw))
            }), flush=True)

    finally:
        imu.close()

if __name__ == "__main__":
    run_imu_fusion()