import time
import json
import numpy as np
from imu_api import ImuSoftUart
from ahrs.filters import EKF
from ahrs.common.orientation import am2q, q2euler

class IMUDiagnostic:
    """Diagnostic pour différencier drift axes vs calibration"""
    
    def __init__(self, imu):
        self.imu = imu
        self.static_samples = []
        self.rotation_samples = []
        
    def test_static_alignment(self, duration_s=10):
        """
        Test 1: Capteur IMMOBILE horizontal
        
        Problème AXES:
        - Quaternion diverge rapidement (< 30s)
        - Dérive dans une direction constante
        - Roll/Pitch dérivent même sans mouvement
        
        Problème CALIBRATION:
        - Quaternion reste stable ou dérive lentement (> 1min)
        - Bruit aléatoire, pas de direction constante
        - Magnitude de l'accéléromètre != 9.81 m/s²
        """
        print(f"\n{'='*60}")
        print("TEST 1: ALIGNEMENT STATIQUE (capteur immobile)")
        print(f"{'='*60}")
        print("⏱️  Durée: 10 secondes")
        print("📍 Position: Capteur à plat, immobile sur table")
        print("\nCollecte des données...\n")
        
        samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration_s:
            m = self.imu.read_measurement(timeout_s=1.0)
            if m is None:
                continue
                
            acc = np.array([m["ax"], m["ay"], m["az"]])
            gyr = np.array([m["gx"], m["gy"], m["gz"]])
            mag = np.array([m["mx"], m["my"], m["mz"]])
            
            samples.append({
                'time': time.time() - start_time,
                'acc': acc,
                'gyr': gyr,
                'mag': mag
            })
            
            # Affichage en temps réel
            acc_mag = np.linalg.norm(acc)
            gyr_mag = np.linalg.norm(gyr)
            print(f"t={time.time()-start_time:5.1f}s | "
                  f"||acc||={acc_mag:6.3f} m/s² | "
                  f"||gyr||={gyr_mag:7.5f} rad/s", end='\r')
            
            time.sleep(0.01)
        
        print("\n")
        self._analyze_static_test(samples)
        return samples
    
    def _analyze_static_test(self, samples):
        """Analyse les résultats du test statique"""
        acc_array = np.array([s['acc'] for s in samples])
        gyr_array = np.array([s['gyr'] for s in samples])
        
        # 1. Magnitude accéléromètre
        acc_mags = np.linalg.norm(acc_array, axis=1)
        acc_mean = np.mean(acc_mags)
        acc_std = np.std(acc_mags)
        
        print("📊 RÉSULTATS:")
        print(f"\n1. MAGNITUDE ACCÉLÉROMÈTRE:")
        print(f"   Moyenne: {acc_mean:.4f} m/s² (attendu: 9.81 m/s²)")
        print(f"   Écart-type: {acc_std:.4f} m/s²")
        
        if abs(acc_mean - 9.81) > 0.3:
            print("   ⚠️  ALERTE: Calibration accéléromètre probablement mauvaise!")
        else:
            print("   ✅ Magnitude correcte")
        
        # 2. Gyroscope au repos
        gyr_mean = np.mean(gyr_array, axis=0)
        gyr_std = np.std(gyr_array, axis=0)
        gyr_mag_mean = np.linalg.norm(gyr_mean)
        
        print(f"\n2. GYROSCOPE AU REPOS:")
        print(f"   Moyenne: x={gyr_mean[0]:.5f}, y={gyr_mean[1]:.5f}, z={gyr_mean[2]:.5f} rad/s")
        print(f"   Écart-type: x={gyr_std[0]:.5f}, y={gyr_std[1]:.5f}, z={gyr_std[2]:.5f} rad/s")
        print(f"   ||moyenne||: {gyr_mag_mean:.5f} rad/s")
        
        if gyr_mag_mean > 0.02:
            print(f"   ⚠️  ALERTE: Bias gyroscope élevé ({gyr_mag_mean:.5f} rad/s)")
            print("   → Recalibration gyroscope recommandée")
        else:
            print("   ✅ Gyroscope bien calibré")
        
        # 3. Direction de la gravité
        acc_mean_vec = np.mean(acc_array, axis=0)
        dominant_axis = np.argmax(np.abs(acc_mean_vec))
        axis_names = ['X', 'Y', 'Z']
        
        print(f"\n3. ORIENTATION GRAVITÉ:")
        print(f"   Vecteur moyen: [{acc_mean_vec[0]:.3f}, {acc_mean_vec[1]:.3f}, {acc_mean_vec[2]:.3f}]")
        print(f"   Axe dominant: {axis_names[dominant_axis]} = {acc_mean_vec[dominant_axis]:.3f} m/s²")
        
        if dominant_axis != 2:
            print(f"   ⚠️  ALERTE: Gravité sur axe {axis_names[dominant_axis]} au lieu de Z!")
            print("   → Problème d'orientation des axes probable")
        elif acc_mean_vec[2] < 0:
            print("   ⚠️  ALERTE: Z négatif (attendu positif en NED)")
            print("   → Axe Z probablement inversé")
        else:
            print("   ✅ Gravité correctement alignée sur +Z (NED)")
    
    def test_rotation_consistency(self, axis='z', duration_s=5):
        """
        Test 2: Rotation lente autour d'un axe
        
        Problème AXES:
        - Mauvais axe détecte la rotation (ex: gyr_y au lieu de gyr_z)
        - Quaternion diverge rapidement après rotation
        - Roll/Pitch/Yaw incohérents avec le mouvement
        
        Problème CALIBRATION:
        - Bon axe détecte rotation, mais magnitude incorrecte
        - Drift lent après arrêt de rotation
        - Intégration gyro != angle réel
        """
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_idx = axis_map[axis.lower()]
        axis_names = ['X', 'Y', 'Z']
        
        print(f"\n{'='*60}")
        print(f"TEST 2: ROTATION AUTOUR DE {axis_names[axis_idx]}")
        print(f"{'='*60}")
        print("⏱️  Durée: 5 secondes")
        print(f"🔄 Action: Rotation LENTE et CONSTANTE autour de {axis_names[axis_idx]}")
        print("\nAppuyez sur Entrée quand prêt...")
        input()
        print("\nDÉBUT dans 3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("TOURNEZ MAINTENANT!\n")
        
        samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration_s:
            m = self.imu.read_measurement(timeout_s=1.0)
            if m is None:
                continue
                
            gyr = np.array([m["gx"], m["gy"], m["gz"]])
            samples.append({
                'time': time.time() - start_time,
                'gyr': gyr
            })
            
            print(f"t={time.time()-start_time:4.1f}s | "
                  f"gx={gyr[0]:7.4f} | gy={gyr[1]:7.4f} | gz={gyr[2]:7.4f} rad/s", 
                  end='\r')
            
            time.sleep(0.01)
        
        print("\n\nARRÊTEZ LA ROTATION!")
        self._analyze_rotation_test(samples, axis_idx, axis_names[axis_idx])
        return samples
    
    def _analyze_rotation_test(self, samples, expected_axis, axis_name):
        """Analyse les résultats du test de rotation"""
        gyr_array = np.array([s['gyr'] for s in samples])
        times = np.array([s['time'] for s in samples])
        
        # Trouver quel axe a la plus grande activité
        gyr_std = np.std(gyr_array, axis=0)
        detected_axis = np.argmax(gyr_std)
        axis_names = ['X', 'Y', 'Z']
        
        print("\n📊 RÉSULTATS:")
        print(f"\n1. DÉTECTION D'AXE:")
        print(f"   Axe attendu: {axis_name}")
        print(f"   Écart-type par axe:")
        print(f"     X: {gyr_std[0]:.5f} rad/s")
        print(f"     Y: {gyr_std[1]:.5f} rad/s")
        print(f"     Z: {gyr_std[2]:.5f} rad/s")
        print(f"   Axe détecté: {axis_names[detected_axis]}")
        
        if detected_axis != expected_axis:
            print(f"   ❌ ERREUR CRITIQUE: Rotation détectée sur {axis_names[detected_axis]} au lieu de {axis_name}!")
            print("   → Problème d'ORIENTATION DES AXES")
            return False
        else:
            print(f"   ✅ Bon axe détecté")
        
        # Intégration pour estimer l'angle total
        gyr_on_axis = gyr_array[:, expected_axis]
        angles = np.cumsum(gyr_on_axis * np.gradient(times))
        total_angle_deg = np.rad2deg(angles[-1])
        
        print(f"\n2. INTÉGRATION GYROSCOPE:")
        print(f"   Angle total: {total_angle_deg:.1f}°")
        print(f"   Vitesse moyenne: {np.mean(np.abs(gyr_on_axis)):.4f} rad/s")
        
        return True
    
    def test_ekf_stability(self, duration_s=30):
        """
        Test 3: Stabilité EKF sur longue durée (immobile)
        
        Problème AXES:
        - Divergence rapide (< 30s)
        - Dérive unidirectionnelle constante
        - Quaternion explose ou devient invalide
        
        Problème CALIBRATION:
        - Dérive lente et aléatoire
        - Quaternion reste valide
        - Erreur s'accumule progressivement
        """
        print(f"\n{'='*60}")
        print("TEST 3: STABILITÉ EKF (30 secondes immobile)")
        print(f"{'='*60}")
        print("⏱️  Durée: 30 secondes")
        print("📍 Position: Capteur à plat, TOTALEMENT immobile")
        print("\nDémarrage de l'EKF...\n")
        
        # Initialisation
        m = self.imu.read_measurement(timeout_s=1.0)
        acc0 = np.array([m["ax"], m["ay"], m["az"]], dtype=float)
        mag0 = np.array([m["mx"], m["my"], m["mz"]], dtype=float)
        q0 = am2q(a=acc0, m=mag0, frame='NED').flatten()
        
        ekf = EKF(gyr=np.zeros((1,3)), acc=acc0.reshape((1,3)), 
                  mag=mag0.reshape((1,3)), frequency=100, q0=q0, frame='NED')
        
        samples = []
        q = q0
        last_time = time.time()
        start_time = time.time()
        
        while time.time() - start_time < duration_s:
            m = self.imu.read_measurement(timeout_s=1.0)
            if m is None:
                continue
            
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            acc = np.array([m["ax"], m["ay"], m["az"]], dtype=float)
            gyr = np.array([m["gx"], m["gy"], m["gz"]], dtype=float)
            mag = np.array([m["mx"], m["my"], m["mz"]], dtype=float)
            
            q = ekf.update(q=q, gyr=gyr, acc=acc, mag=mag, dt=dt)
            
            # Conversion en angles d'Euler
            roll, pitch, yaw = q2euler(q)
            
            samples.append({
                'time': time.time() - start_time,
                'q': q.copy(),
                'roll': np.rad2deg(roll),
                'pitch': np.rad2deg(pitch),
                'yaw': np.rad2deg(yaw)
            })
            
            print(f"t={time.time()-start_time:5.1f}s | "
                  f"R={np.rad2deg(roll):6.2f}° | "
                  f"P={np.rad2deg(pitch):6.2f}° | "
                  f"Y={np.rad2deg(yaw):6.2f}°", end='\r')
            
            time.sleep(0.01)
        
        print("\n")
        self._analyze_ekf_stability(samples)
        return samples
    
    def _analyze_ekf_stability(self, samples):
        """Analyse la stabilité de l'EKF"""
        rolls = np.array([s['roll'] for s in samples])
        pitchs = np.array([s['pitch'] for s in samples])
        yaws = np.array([s['yaw'] for s in samples])
        times = np.array([s['time'] for s in samples])
        
        # Drift total
        roll_drift = abs(rolls[-1] - rolls[0])
        pitch_drift = abs(pitchs[-1] - pitchs[0])
        yaw_drift = abs(yaws[-1] - yaws[0])
        
        # Vitesse de drift (deg/min)
        duration_min = times[-1] / 60
        roll_drift_rate = roll_drift / duration_min
        pitch_drift_rate = pitch_drift / duration_min
        yaw_drift_rate = yaw_drift / duration_min
        
        print("📊 RÉSULTATS:")
        print(f"\n1. DRIFT TOTAL ({times[-1]:.1f}s):")
        print(f"   Roll:  {roll_drift:6.2f}° (taux: {roll_drift_rate:.2f}°/min)")
        print(f"   Pitch: {pitch_drift:6.2f}° (taux: {pitch_drift_rate:.2f}°/min)")
        print(f"   Yaw:   {yaw_drift:6.2f}° (taux: {yaw_drift_rate:.2f}°/min)")
        
        # Diagnostic
        max_drift = max(roll_drift, pitch_drift, yaw_drift)
        max_drift_rate = max(roll_drift_rate, pitch_drift_rate, yaw_drift_rate)
        
        print(f"\n2. DIAGNOSTIC:")
        
        if max_drift > 10 and times[-1] < 20:
            print("   ❌ ERREUR CRITIQUE: Divergence rapide (> 10° en < 20s)")
            print("   → Problème d'ORIENTATION DES AXES très probable")
            print("   → Les axes X/Y/Z ne correspondent pas au frame NED")
        elif max_drift_rate > 5:
            print("   ⚠️  Drift élevé (> 5°/min)")
            print("   → Problème de CALIBRATION probable")
            print("   → Vérifier: biais gyroscope, soft-iron magnéto")
        elif max_drift_rate > 2:
            print("   ⚠️  Drift modéré (2-5°/min)")
            print("   → Calibration acceptable mais améliorable")
        else:
            print("   ✅ Drift acceptable (< 2°/min)")
            print("   → Axes et calibration OK")
        
        # Vérifier quaternion valide
        qs = np.array([s['q'] for s in samples])
        q_norms = np.linalg.norm(qs, axis=1)
        q_norm_error = np.max(np.abs(q_norms - 1.0))
        
        print(f"\n3. VALIDITÉ QUATERNION:")
        print(f"   Erreur max de norme: {q_norm_error:.6f}")
        
        if q_norm_error > 0.01:
            print("   ⚠️  Quaternion s'éloigne de la norme unitaire")
            print("   → Possible problème numérique ou axes incorrects")
        else:
            print("   ✅ Quaternion reste valide")


def run_full_diagnostic():
    """Lance tous les tests de diagnostic"""
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "DIAGNOSTIC IMU - AXES vs CALIBRATION" + " "*11 + "║")
    print("╚" + "="*58 + "╝")
    
    imu = ImuSoftUart(rx_gpio=24, baudrate=80000)
    imu.open()
    
    try:
        diag = IMUDiagnostic(imu)
        
        # Test 1: Statique
        print("\n📋 Nous allons faire 3 tests pour identifier le problème")
        input("\nAppuyez sur Entrée pour commencer...")
        diag.test_static_alignment(duration_s=10)
        
        # Test 2: Rotation
        input("\nAppuyez sur Entrée pour le test de rotation...")
        diag.test_rotation_consistency(axis='z', duration_s=5)
        
        # Test 3: Stabilité EKF
        input("\nAppuyez sur Entrée pour le test de stabilité EKF...")
        diag.test_ekf_stability(duration_s=30)
        
        print("\n" + "="*60)
        print("DIAGNOSTIC TERMINÉ")
        print("="*60)
        
    finally:
        imu.close()


if __name__ == "__main__":
    run_full_diagnostic()