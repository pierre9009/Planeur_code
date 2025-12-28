#!/usr/bin/env python3
"""
EKF avec initialisation robuste du quaternion
Moins sensible aux petites inclinaisons du capteur
"""

import time
import json
import sys
import numpy as np
from imu_api import ImuHardwareUart
from ahrs.filters import EKF
from ahrs.common.orientation import q2euler

def quaternion_from_acc_mag_robust(acc, mag, frame='NED'):
    """
    Calcul robuste du quaternion initial à partir de l'accéléromètre et magnétomètre
    
    Cette version force Z à pointer vers le bas (pour NED) même si le capteur
    est légèrement incliné, évitant les erreurs d'initialisation catastrophiques.
    
    Args:
        acc: Vecteur accélération [ax, ay, az] en m/s²
        mag: Vecteur magnétique [mx, my, mz] en µT
        frame: 'NED' ou 'ENU'
    
    Returns:
        Quaternion [w, x, y, z] normalisé
    """
    # Normaliser l'accélération
    acc_norm = acc / np.linalg.norm(acc)
    
    # En NED, la gravité pointe vers +Z (down)
    # On force le vecteur "down" à être proche de [0, 0, 1]
    if frame == 'NED':
        down = np.array([0.0, 0.0, 1.0])
    else:  # ENU
        down = np.array([0.0, 0.0, -1.0])
    
    # Vérifier que l'accélération est bien principalement vers le bas
    # Si l'inclinaison est > 30°, on garde quand même la mesure
    down_component = np.dot(acc_norm, down)
    
    if abs(down_component) < 0.866:  # cos(30°) ≈ 0.866
        print(f"⚠️  Attention: capteur incliné de {np.rad2deg(np.arccos(abs(down_component))):.1f}°")
        print(f"   Initialisation quand même avec les données disponibles")
    
    # Calculer le vecteur "Nord" à partir du magnétomètre
    # Projeter le champ magnétique dans le plan horizontal
    mag_norm = mag / np.linalg.norm(mag)
    
    # Enlever la composante verticale du champ magnétique
    mag_horizontal = mag_norm - np.dot(mag_norm, down) * down
    mag_horizontal_norm = mag_horizontal / np.linalg.norm(mag_horizontal)
    
    # Construire la base orthonormée (NED)
    # X (North) : direction du champ magnétique horizontal
    # Y (East)  : perpendiculaire à X et Z
    # Z (Down)  : vers le bas
    
    if frame == 'NED':
        z_axis = down  # Vers le bas
        x_axis = mag_horizontal_norm  # Vers le nord magnétique
        y_axis = np.cross(z_axis, x_axis)  # Vers l'est
    else:  # ENU
        z_axis = -down  # Vers le haut
        y_axis = mag_horizontal_norm  # Vers le nord magnétique
        x_axis = np.cross(y_axis, z_axis)  # Vers l'est
    
    # Normaliser (au cas où)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # Matrice de rotation (colonnes = axes de la base)
    R = np.column_stack([x_axis, y_axis, z_axis])
    
    # Conversion matrice de rotation → quaternion
    q = rotation_matrix_to_quaternion(R)
    
    return q

def rotation_matrix_to_quaternion(R):
    """
    Convertit une matrice de rotation en quaternion
    
    Args:
        R: Matrice 3x3 de rotation
    
    Returns:
        Quaternion [w, x, y, z] normalisé
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

def quaternion_identity():
    """Retourne le quaternion identité (pas de rotation)"""
    return np.array([1.0, 0.0, 0.0, 0.0])

def run_imu_fusion():
    """Programme principal avec initialisation robuste"""
    
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "EKF - INITIALISATION ROBUSTE" + " "*20 + "║")
    print("╚" + "="*68 + "╝\n")
    
    imu = ImuHardwareUart(port="/dev/ttyS0", baudrate=115200)
    imu.open()
    
    print("📊 Collecte de données initiales (moyenne sur 100 échantillons)...")
    
    # Collecter plusieurs échantillons pour moyenner
    acc_samples = []
    mag_samples = []
    
    for _ in range(100):
        m = imu.read_measurement(timeout_s=1.0)
        if m is None:
            continue
        acc_samples.append([m["ax"], m["ay"], m["az"]])
        mag_samples.append([m["mx"], m["my"], m["mz"]])
    
    if len(acc_samples) < 50:
        print("❌ Pas assez d'échantillons collectés")
        imu.close()
        return
    
    # Moyenner pour réduire le bruit
    acc0 = np.mean(acc_samples, axis=0)
    mag0 = np.mean(mag_samples, axis=0)
    
    print(f"\n✓ Données initiales (moyennées) :")
    print(f"  acc0: [{acc0[0]:7.4f}, {acc0[1]:7.4f}, {acc0[2]:7.4f}] m/s²")
    print(f"  mag0: [{mag0[0]:7.2f}, {mag0[1]:7.2f}, {mag0[2]:7.2f}] µT")
    
    # Vérifier magnitude
    acc_mag = np.linalg.norm(acc0)
    print(f"  ||acc0||: {acc_mag:.4f} m/s² (attendu: ~9.81)")
    
    if abs(acc_mag - 9.81) > 1.0:
        print("  ⚠️  Magnitude suspecte - vérifier calibration")
    
    # Calcul du quaternion initial ROBUSTE
    print(f"\n🔧 Calcul du quaternion initial (méthode robuste)...")
    q0 = quaternion_from_acc_mag_robust(acc0, mag0, frame='NED')
    
    # Alternative simple si capteur vraiment à plat : quaternion identité
    # q0 = quaternion_identity()
    # print(f"   Utilisation du quaternion identité (capteur supposé horizontal)")
    
    print(f"  q0: [{q0[0]:7.4f}, {q0[1]:7.4f}, {q0[2]:7.4f}, {q0[3]:7.4f}]")
    print(f"  ||q0||: {np.linalg.norm(q0):.6f}")
    
    # Afficher les angles d'Euler
    roll, pitch, yaw = q2euler(q0)
    print(f"\n📐 Angles d'Euler initiaux:")
    print(f"  Roll:  {np.rad2deg(roll):7.2f}°")
    print(f"  Pitch: {np.rad2deg(pitch):7.2f}°")
    print(f"  Yaw:   {np.rad2deg(yaw):7.2f}°")
    
    if abs(np.rad2deg(roll)) > 30 or abs(np.rad2deg(pitch)) > 30:
        print(f"\n⚠️  ATTENTION: Angles initiaux suspects!")
        print(f"  Le capteur semble très incliné ou il y a un problème d'axes.")
        print(f"  Continuer quand même ? (y/n)")
        
        response = input().strip().lower()
        if response != 'y':
            print("Arrêt.")
            imu.close()
            return
    
    # Initialisation de l'EKF
    print(f"\n🚀 Initialisation EKF...")
    ekf = EKF(
        gyr=np.zeros((1, 3)),
        acc=acc0.reshape((1, 3)),
        mag=mag0.reshape((1, 3)),
        frequency=100,
        q0=q0,
        frame='NED'
    )
    
    print(f"✓ EKF initialisé\n")
    print("="*70)
    print("DÉMARRAGE FUSION DE DONNÉES")
    print("="*70)
    print("Ctrl+C pour arrêter\n")
    
    last_time = time.time()
    q = q0.copy()
    iteration = 0
    
    try:
        while True:
            m = imu.read_measurement(timeout_s=1.0)
            if m is None:
                continue
            
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            acc = np.array([m["ax"], m["ay"], m["az"]], dtype=float)
            gyr = np.array([m["gx"], m["gy"], m["gz"]], dtype=float)
            mag = np.array([m["mx"], m["my"], m["mz"]], dtype=float)
            
            # Mise à jour EKF
            q = ekf.update(q=q, gyr=gyr, acc=acc, mag=mag, dt=dt)
            
            # Conversion en Euler
            roll, pitch, yaw = q2euler(q)
            
            # Vérifier validité quaternion
            q_norm = np.linalg.norm(q)
            if abs(q_norm - 1.0) > 0.1:
                print(f"\n❌ DIVERGENCE: ||q||={q_norm:.4f}")
                break
            
            # Affichage périodique
            if iteration % 10 == 0:
                print(json.dumps({
                    "qw": float(q[0]), "qx": float(q[1]), 
                    "qy": float(q[2]), "qz": float(q[3]),
                    "roll": float(np.rad2deg(roll)),
                    "pitch": float(np.rad2deg(pitch)),
                    "yaw": float(np.rad2deg(yaw)),
                    "ax": float(m["ax"]), "ay": float(m["ay"]), "az": float(m["az"]),
                    "gx": float(m["gx"]), "gy": float(m["gy"]), "gz": float(m["gz"]),
                    "mx": float(m["mx"]), "my": float(m["my"]), "mz": float(m["mz"]),
                    "dt": float(dt * 1000),
                    "q_norm": float(q_norm)
                }), flush=True)
            
            iteration += 1
            
    except KeyboardInterrupt:
        print("\n\nArrêt demandé")
    
    finally:
        stats = imu.get_stats()
        print(f"\n{'='*70}")
        print("STATISTIQUES FINALES")
        print(f"{'='*70}")
        print(f"Itérations: {iteration}")
        print(f"Paquets reçus: {stats['total_packets']}")
        print(f"Erreurs CRC: {stats['bad_crc']}")
        print(f"Timeouts: {stats['timeouts']}")
        imu.close()

if __name__ == "__main__":
    run_imu_fusion()