#!/usr/bin/env python3
"""
Validation systématique de l'EKF - Debugging étape par étape
Chaque étape DOIT passer avant de continuer à la suivante
"""

import time
import json
import sys
import numpy as np
from imu_api import ImuSoftUart
from ahrs.filters import EKF
from ahrs.common.orientation import am2q, q2euler

# ============================================================================
# ÉTAPE 1: VALIDATION COMMUNICATION
# ============================================================================
def step1_validate_communication():
    """
    Objectif: Vérifier que la communication UART fonctionne de manière stable
    Critères de succès:
    - Aucun timeout pendant 10 secondes
    - Séquence incrémentale sans saut
    - Valeurs dans les plages attendues
    """
    print("\n" + "="*70)
    print("ÉTAPE 1: VALIDATION COMMUNICATION UART")
    print("="*70)
    print("⏱️  Durée: 10 secondes")
    print("📍 Gardez le capteur immobile\n")
    
    imu = ImuSoftUart(rx_gpio=24, baudrate=80000)
    imu.open()
    
    try:
        timeouts = 0
        seq_errors = 0
        last_seq = None
        samples = []
        
        start_time = time.time()
        while time.time() - start_time < 10:
            m = imu.read_measurement(timeout_s=0.1)
            
            if m is None:
                timeouts += 1
                print("⚠️  Timeout détecté!")
                continue
            
            # Vérifier séquence
            if last_seq is not None:
                expected = (last_seq + 1) % (2**32)
                if m["seq"] != expected:
                    seq_errors += 1
                    print(f"⚠️  Saut de séquence: {last_seq} -> {m['seq']}")
            
            last_seq = m["seq"]
            samples.append(m)
            
            # Affichage progression
            if len(samples) % 50 == 0:
                print(f"✓ {len(samples)} échantillons reçus (seq={m['seq']})", end='\r')
        
        print(f"\n\n📊 RÉSULTATS ÉTAPE 1:")
        print(f"   Échantillons reçus: {len(samples)}")
        print(f"   Timeouts: {timeouts}")
        print(f"   Erreurs de séquence: {seq_errors}")
        print(f"   Fréquence moyenne: {len(samples)/10:.1f} Hz")
        
        # Validation
        success = timeouts == 0 and seq_errors == 0 and len(samples) > 900
        
        if success:
            print("   ✅ ÉTAPE 1 VALIDÉE - Communication stable")
        else:
            print("   ❌ ÉTAPE 1 ÉCHOUÉE - Problème de communication")
            if timeouts > 0:
                print("      → Vérifier câblage UART")
            if seq_errors > 0:
                print("      → Vérifier pull-up et baud rate")
            return False, None
        
        return True, samples
        
    finally:
        imu.close()


# ============================================================================
# ÉTAPE 2: VALIDATION PLAGES DE VALEURS
# ============================================================================
def step2_validate_ranges(samples):
    """
    Objectif: Vérifier que les valeurs sont physiquement plausibles
    Critères de succès:
    - Magnitude accéléromètre ≈ 9.81 m/s² (±1 m/s²)
    - Gyroscope proche de 0 au repos (< 0.1 rad/s)
    - Magnétomètre dans [-100, 100] µT
    - Pas de NaN, Inf
    """
    print("\n" + "="*70)
    print("ÉTAPE 2: VALIDATION PLAGES DE VALEURS")
    print("="*70)
    
    acc_array = np.array([[s["ax"], s["ay"], s["az"]] for s in samples])
    gyr_array = np.array([[s["gx"], s["gy"], s["gz"]] for s in samples])
    mag_array = np.array([[s["mx"], s["my"], s["mz"]] for s in samples])
    
    # Vérifier NaN/Inf
    has_nan = np.any(np.isnan(acc_array)) or np.any(np.isnan(gyr_array)) or np.any(np.isnan(mag_array))
    has_inf = np.any(np.isinf(acc_array)) or np.any(np.isinf(gyr_array)) or np.any(np.isinf(mag_array))
    
    print(f"\n1. VÉRIFICATION NaN/Inf:")
    print(f"   NaN détectés: {'OUI ❌' if has_nan else 'NON ✅'}")
    print(f"   Inf détectés: {'OUI ❌' if has_inf else 'NON ✅'}")
    
    if has_nan or has_inf:
        print("   ❌ DONNÉES INVALIDES - Problème de parsing ou capteur")
        return False
    
    # Accéléromètre
    acc_mags = np.linalg.norm(acc_array, axis=1)
    acc_mag_mean = np.mean(acc_mags)
    acc_mag_std = np.std(acc_mags)
    
    print(f"\n2. ACCÉLÉROMÈTRE (au repos):")
    print(f"   Magnitude moyenne: {acc_mag_mean:.3f} m/s² (attendu: 9.81)")
    print(f"   Écart-type: {acc_mag_std:.4f} m/s²")
    
    acc_ok = 8.8 < acc_mag_mean < 10.8 and acc_mag_std < 0.5
    print(f"   {'✅' if acc_ok else '❌'} Plage {'OK' if acc_ok else 'INVALIDE'}")
    
    # Gyroscope
    gyr_mean = np.mean(gyr_array, axis=0)
    gyr_std = np.std(gyr_array, axis=0)
    gyr_mag = np.linalg.norm(gyr_mean)
    
    print(f"\n3. GYROSCOPE (au repos):")
    print(f"   Moyenne: x={gyr_mean[0]:.5f}, y={gyr_mean[1]:.5f}, z={gyr_mean[2]:.5f} rad/s")
    print(f"   Écart-type: x={gyr_std[0]:.5f}, y={gyr_std[1]:.5f}, z={gyr_std[2]:.5f} rad/s")
    print(f"   ||Moyenne||: {gyr_mag:.5f} rad/s")
    
    gyr_ok = gyr_mag < 0.1 and np.all(gyr_std < 0.05)
    print(f"   {'✅' if gyr_ok else '⚠️ '} Plage {'OK' if gyr_ok else 'ATTENTION - Possible drift'}")
    
    # Magnétomètre
    mag_mean = np.mean(mag_array, axis=0)
    mag_mag = np.linalg.norm(mag_mean)
    
    print(f"\n4. MAGNÉTOMÈTRE:")
    print(f"   Moyenne: x={mag_mean[0]:.2f}, y={mag_mean[1]:.2f}, z={mag_mean[2]:.2f} µT")
    print(f"   ||Magnitude||: {mag_mag:.2f} µT (attendu: 20-60 µT)")
    
    mag_ok = 10 < mag_mag < 100 and np.all(np.abs(mag_mean) < 100)
    print(f"   {'✅' if mag_ok else '❌'} Plage {'OK' if mag_ok else 'INVALIDE'}")
    
    # Validation globale
    success = acc_ok and mag_ok and not has_nan and not has_inf
    
    print(f"\n📊 RÉSULTATS ÉTAPE 2:")
    if success:
        print("   ✅ ÉTAPE 2 VALIDÉE - Toutes les valeurs plausibles")
    else:
        print("   ❌ ÉTAPE 2 ÉCHOUÉE - Valeurs hors plage")
        if not acc_ok:
            print("      → Recalibrer accéléromètre")
        if not mag_ok:
            print("      → Recalibrer magnétomètre (hard-iron/soft-iron)")
    
    return success


# ============================================================================
# ÉTAPE 3: VALIDATION INITIALISATION EKF
# ============================================================================
def step3_validate_initialization():
    """
    Objectif: Vérifier que l'initialisation de l'EKF est correcte
    Critères de succès:
    - Quaternion initial normalisé (norme = 1)
    - Angles d'Euler cohérents avec orientation réelle
    - Pas d'erreur lors de l'initialisation
    """
    print("\n" + "="*70)
    print("ÉTAPE 3: VALIDATION INITIALISATION EKF")
    print("="*70)
    print("📍 Positionnez le capteur à PLAT sur une table\n")
    
    imu = ImuSoftUart(rx_gpio=24, baudrate=80000)
    imu.open()
    
    try:
        # Lecture initiale
        m = imu.read_measurement(timeout_s=1.0)
        if m is None:
            print("❌ Impossible de lire les données pour initialisation")
            return False, None, None, None
        
        acc0 = np.array([m["ax"], m["ay"], m["az"]], dtype=float)
        mag0 = np.array([m["mx"], m["my"], m["mz"]], dtype=float)
        
        print(f"1. DONNÉES INITIALES:")
        print(f"   acc0: [{acc0[0]:.3f}, {acc0[1]:.3f}, {acc0[2]:.3f}] m/s²")
        print(f"   mag0: [{mag0[0]:.2f}, {mag0[1]:.2f}, {mag0[2]:.2f}] µT")
        
        # Calcul quaternion initial
        print(f"\n2. CALCUL QUATERNION INITIAL (am2q):")
        try:
            q0 = am2q(a=acc0, m=mag0, frame='NED')
            q0 = q0.flatten()
            print(f"   q0: [{q0[0]:.4f}, {q0[1]:.4f}, {q0[2]:.4f}, {q0[3]:.4f}]")
        except Exception as e:
            print(f"   ❌ ERREUR lors du calcul: {e}")
            return False, None, None, None
        
        # Vérifier normalisation
        q0_norm = np.linalg.norm(q0)
        print(f"   ||q0||: {q0_norm:.6f} (attendu: 1.0)")
        
        if abs(q0_norm - 1.0) > 0.01:
            print(f"   ❌ Quaternion mal normalisé!")
            return False, None, None, None
        
        # Conversion en Euler
        roll, pitch, yaw = q2euler(q0)
        roll_deg = np.rad2deg(roll)
        pitch_deg = np.rad2deg(pitch)
        yaw_deg = np.rad2deg(yaw)
        
        print(f"\n3. ANGLES D'EULER INITIAUX:")
        print(f"   Roll:  {roll_deg:7.2f}° (attendu: ~0°)")
        print(f"   Pitch: {pitch_deg:7.2f}° (attendu: ~0°)")
        print(f"   Yaw:   {yaw_deg:7.2f}° (variable)")
        
        # Vérifier que roll/pitch sont proches de 0 (capteur plat)
        if abs(roll_deg) > 10 or abs(pitch_deg) > 10:
            print(f"   ⚠️  ATTENTION: Capteur non horizontal!")
            print(f"      → Vérifier orientation physique")
            print(f"      → Ou problème d'axes X/Y/Z")
        
        # Initialisation EKF
        print(f"\n4. INITIALISATION EKF:")
        try:
            ekf = EKF(
                gyr=np.zeros((1, 3)),
                acc=acc0.reshape((1, 3)),
                mag=mag0.reshape((1, 3)),
                frequency=100,
                q0=q0,
                frame='NED'
            )
            print(f"   ✅ EKF initialisé avec succès")
        except Exception as e:
            print(f"   ❌ ERREUR lors de l'initialisation: {e}")
            return False, None, None, None
        
        print(f"\n📊 RÉSULTATS ÉTAPE 3:")
        print(f"   ✅ ÉTAPE 3 VALIDÉE - EKF initialisé correctement")
        
        return True, ekf, q0, imu
        
    except Exception as e:
        print(f"❌ Exception inattendue: {e}")
        imu.close()
        return False, None, None, None


# ============================================================================
# ÉTAPE 4: VALIDATION STABILITÉ COURT TERME
# ============================================================================
def step4_validate_short_term_stability(ekf, q0, imu):
    """
    Objectif: Vérifier que l'EKF reste stable sur 10 secondes (capteur immobile)
    Critères de succès:
    - Quaternion reste proche de q0
    - Drift < 5° sur 10 secondes
    - Pas de divergence soudaine
    """
    print("\n" + "="*70)
    print("ÉTAPE 4: VALIDATION STABILITÉ COURT TERME")
    print("="*70)
    print("⏱️  Durée: 10 secondes")
    print("📍 GARDEZ LE CAPTEUR TOTALEMENT IMMOBILE\n")
    
    input("Appuyez sur Entrée pour commencer...")
    
    q = q0.copy()
    last_time = time.time()
    samples = []
    
    start_time = time.time()
    iteration = 0
    
    try:
        while time.time() - start_time < 10:
            m = imu.read_measurement(timeout_s=0.1)
            if m is None:
                print("⚠️  Timeout!")
                continue
            
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            acc = np.array([m["ax"], m["ay"], m["az"]], dtype=float)
            gyr = np.array([m["gx"], m["gy"], m["gz"]], dtype=float)
            mag = np.array([m["mx"], m["my"], m["mz"]], dtype=float)
            
            # ⚠️ POINT CRITIQUE: Update EKF
            q_prev = q.copy()
            q = ekf.update(q=q, gyr=gyr, acc=acc, mag=mag, dt=dt)
            
            # Vérifier quaternion valide
            q_norm = np.linalg.norm(q)
            if abs(q_norm - 1.0) > 0.1:
                print(f"\n❌ DIVERGENCE DÉTECTÉE à t={time.time()-start_time:.2f}s")
                print(f"   ||q|| = {q_norm:.4f} (devrait être 1.0)")
                print(f"   q = [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
                print(f"   acc = [{acc[0]:.3f}, {acc[1]:.3f}, {acc[2]:.3f}]")
                print(f"   gyr = [{gyr[0]:.5f}, {gyr[1]:.5f}, {gyr[2]:.5f}]")
                print(f"   mag = [{mag[0]:.2f}, {mag[1]:.2f}, {mag[2]:.2f}]")
                print(f"   dt = {dt:.4f}s")
                return False
            
            roll, pitch, yaw = q2euler(q)
            
            samples.append({
                'time': time.time() - start_time,
                'q': q.copy(),
                'q_norm': q_norm,
                'roll': np.rad2deg(roll),
                'pitch': np.rad2deg(pitch),
                'yaw': np.rad2deg(yaw),
                'dt': dt,
                'acc': acc.copy(),
                'gyr': gyr.copy()
            })
            
            # Affichage toutes les 50 itérations
            if iteration % 50 == 0:
                print(f"t={time.time()-start_time:5.1f}s | "
                      f"R={np.rad2deg(roll):6.2f}° | "
                      f"P={np.rad2deg(pitch):6.2f}° | "
                      f"Y={np.rad2deg(yaw):6.2f}° | "
                      f"||q||={q_norm:.4f}", end='\r')
            
            iteration += 1
        
        print("\n")
        
        # Analyse
        roll0, pitch0, yaw0 = q2euler(q0)
        rollf, pitchf, yawf = q2euler(q)
        
        drift_roll = abs(np.rad2deg(rollf - roll0))
        drift_pitch = abs(np.rad2deg(pitchf - pitch0))
        drift_yaw = abs(np.rad2deg(yawf - yaw0))
        
        print(f"📊 RÉSULTATS ÉTAPE 4:")
        print(f"   Itérations réussies: {len(samples)}")
        print(f"   Drift Roll:  {drift_roll:.2f}°")
        print(f"   Drift Pitch: {drift_pitch:.2f}°")
        print(f"   Drift Yaw:   {drift_yaw:.2f}°")
        
        # Critères de succès
        max_drift = max(drift_roll, drift_pitch, drift_yaw)
        success = max_drift < 5.0
        
        if success:
            print(f"   ✅ ÉTAPE 4 VALIDÉE - Drift < 5° sur 10s")
        else:
            print(f"   ❌ ÉTAPE 4 ÉCHOUÉE - Drift excessif ({max_drift:.2f}°)")
            print(f"\n🔍 DIAGNOSTIC:")
            
            # Analyser la cause
            dt_array = np.array([s['dt'] for s in samples])
            dt_mean = np.mean(dt_array)
            dt_std = np.std(dt_array)
            
            print(f"   dt moyen: {dt_mean*1000:.2f} ms (écart-type: {dt_std*1000:.2f} ms)")
            
            if dt_std > dt_mean * 0.5:
                print(f"   ⚠️  dt très variable → Problème de timing")
            
            gyr_norms = [np.linalg.norm(s['gyr']) for s in samples]
            if np.mean(gyr_norms) > 0.05:
                print(f"   ⚠️  Gyroscope actif au repos → Recalibration nécessaire")
            
            if drift_roll > 5 or drift_pitch > 5:
                print(f"   ⚠️  Drift Roll/Pitch → Problème d'axes ou calibration accéléro")
        
        return success, samples
        
    except Exception as e:
        print(f"\n❌ Exception durant l'exécution: {e}")
        import traceback
        traceback.print_exc()
        return False, None


# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================
def main():
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "VALIDATION SYSTÉMATIQUE EKF" + " "*26 + "║")
    print("╚" + "="*68 + "╝")
    print("\nCe programme va valider chaque composant un par un.")
    print("Chaque étape DOIT réussir avant de passer à la suivante.\n")
    
    input("Appuyez sur Entrée pour commencer...")
    
    # ÉTAPE 1: Communication
    success, samples = step1_validate_communication()
    if not success:
        print("\n🛑 ARRÊT: Résolvez les problèmes de communication d'abord")
        return
    
    # ÉTAPE 2: Plages de valeurs
    success = step2_validate_ranges(samples)
    if not success:
        print("\n🛑 ARRÊT: Calibrez les capteurs avant de continuer")
        return
    
    # ÉTAPE 3: Initialisation
    success, ekf, q0, imu = step3_validate_initialization()
    if not success:
        print("\n🛑 ARRÊT: Problème d'initialisation EKF")
        return
    
    # ÉTAPE 4: Stabilité court terme
    try:
        success, samples = step4_validate_short_term_stability(ekf, q0, imu)
        
        if success:
            print("\n" + "="*70)
            print("🎉 TOUTES LES ÉTAPES VALIDÉES!")
            print("="*70)
            print("\nVotre EKF est correctement configuré et stable.")
            print("Vous pouvez maintenant l'utiliser en production.\n")
        else:
            print("\n" + "="*70)
            print("⚠️  ÉTAPE 4 ÉCHOUÉE - EKF instable")
            print("="*70)
            print("\nLe problème est dans la configuration ou les paramètres de l'EKF.")
            print("Vérifiez les messages de diagnostic ci-dessus.\n")
    
    finally:
        imu.close()


if __name__ == "__main__":
    main()