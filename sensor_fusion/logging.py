import time
import json
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from imu_api import ImuSoftUart


class ImuLogger:
    """Logger pour enregistrer les données IMU avec horodatage précis"""
    
    def __init__(self, log_dir="imu_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Nom du fichier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"imu_data_{timestamp}.log"
        self.metadata_file = self.log_dir / f"imu_metadata_{timestamp}.json"
        
        self.start_time = None
        self.sample_count = 0
        
    def start_logging(self):
        """Démarre l'enregistrement"""
        self.start_time = time.time()
        print(f"📝 Enregistrement démarré : {self.log_file}", file=sys.stderr)
        
    def log_sample(self, timestamp, seq, acc, gyr, mag, temp):
        """
        Enregistre un échantillon avec horodatage
        
        Format : timestamp_abs, dt, seq, ax, ay, az, gx, gy, gz, mx, my, mz, temp
        """
        if self.start_time is None:
            self.start_logging()
        
        # Temps absolu depuis le début
        time_abs = timestamp - self.start_time
        
        # Écriture dans le fichier (format CSV)
        with open(self.log_file, 'a') as f:
            line = (f"{time_abs:.6f},"
                   f"{seq},"
                   f"{acc[0]:.6f},{acc[1]:.6f},{acc[2]:.6f},"
                   f"{gyr[0]:.6f},{gyr[1]:.6f},{gyr[2]:.6f},"
                   f"{mag[0]:.6f},{mag[1]:.6f},{mag[2]:.6f},"
                   f"{temp:.2f}\n")
            f.write(line)
        
        self.sample_count += 1
        
    def save_metadata(self, info):
        """Sauvegarde les métadonnées de la session"""
        metadata = {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "sample_count": self.sample_count,
            "duration_seconds": time.time() - self.start_time,
            "log_file": str(self.log_file),
            **info
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n📊 Métadonnées sauvegardées : {self.metadata_file}", file=sys.stderr)
        print(f"   Échantillons : {self.sample_count}", file=sys.stderr)
        print(f"   Durée : {metadata['duration_seconds']:.1f}s", file=sys.stderr)


def run_imu_logger(duration_seconds=None, output_stats=True):
    """
    Enregistre les données IMU dans un fichier .log
    
    Args:
        duration_seconds: Durée d'enregistrement (None = infini)
        output_stats: Afficher les statistiques en temps réel
    """
    imu = ImuSoftUart(rx_gpio=24, baudrate=57600)
    imu.open()
    
    logger = ImuLogger()
    
    # Phase d'initialisation pour calculer les statistiques
    print("Initialisation (collecte de 100 échantillons)...", file=sys.stderr)
    init_samples = []
    while len(init_samples) < 100:
        m = imu.read_measurement(timeout_s=0.5)
        if m:
            init_samples.append(m)
            sys.stderr.write(f"\rÉchantillons: {len(init_samples)}/100")
            sys.stderr.flush()
    
    # Calcul des statistiques d'initialisation
    acc_samples = np.array([[s["ax"], s["ay"], s["az"]] for s in init_samples])
    mag_samples = np.array([[s["mx"], s["my"], s["mz"]] for s in init_samples])
    gyr_samples = np.array([[s["gx"], s["gy"], s["gz"]] for s in init_samples])
    
    acc_mean = np.mean(acc_samples, axis=0)
    acc_std = np.std(acc_samples, axis=0)
    mag_mean = np.mean(mag_samples, axis=0)
    mag_std = np.std(mag_samples, axis=0)
    gyr_mean = np.mean(gyr_samples, axis=0)
    gyr_std = np.std(gyr_samples, axis=0)
    
    # Calcul de l'inclinaison magnétique
    mag_horizontal = np.sqrt(mag_mean[0]**2 + mag_mean[1]**2)
    mag_inclination = np.degrees(np.arctan2(mag_mean[2], mag_horizontal))
    mag_azimuth = np.degrees(np.arctan2(mag_mean[0], mag_mean[1]))
    
    if output_stats:
        print("\n\n=== STATISTIQUES INITIALISATION ===", file=sys.stderr)
        print(f"Accéléromètre:", file=sys.stderr)
        print(f"  Moyenne : {acc_mean}", file=sys.stderr)
        print(f"  Écart-type : {acc_std}", file=sys.stderr)
        print(f"  Norme : {np.linalg.norm(acc_mean):.3f} m/s²", file=sys.stderr)
        print(f"\nMagnétomètre:", file=sys.stderr)
        print(f"  Moyenne : {mag_mean}", file=sys.stderr)
        print(f"  Écart-type : {mag_std}", file=sys.stderr)
        print(f"  Norme : {np.linalg.norm(mag_mean):.3f} µT", file=sys.stderr)
        print(f"  Inclinaison : {mag_inclination:.1f}°", file=sys.stderr)
        print(f"  Azimut : {mag_azimuth:.1f}°", file=sys.stderr)
        print(f"\nGyroscope:", file=sys.stderr)
        print(f"  Moyenne : {gyr_mean}", file=sys.stderr)
        print(f"  Écart-type : {gyr_std}", file=sys.stderr)
        print("="*40 + "\n", file=sys.stderr)
    
    # Métadonnées de la session
    metadata = {
        "initialization": {
            "acc_mean": acc_mean.tolist(),
            "acc_std": acc_std.tolist(),
            "mag_mean": mag_mean.tolist(),
            "mag_std": mag_std.tolist(),
            "mag_inclination_deg": float(mag_inclination),
            "mag_azimuth_deg": float(mag_azimuth),
            "gyr_mean": gyr_mean.tolist(),
            "gyr_std": gyr_std.tolist(),
        }
    }
    
    # Écriture de l'en-tête du fichier log
    with open(logger.log_file, 'w') as f:
        f.write("# IMU Data Log\n")
        f.write(f"# Date: {datetime.now().isoformat()}\n")
        f.write(f"# Format: time_abs(s), seq, ax, ay, az, gx, gy, gz, mx, my, mz, temp\n")
        f.write("time_abs,seq,ax,ay,az,gx,gy,gz,mx,my,mz,temp\n")
    
    logger.start_logging()
    
    print("🔴 ENREGISTREMENT EN COURS", file=sys.stderr)
    if duration_seconds:
        print(f"   Durée: {duration_seconds}s", file=sys.stderr)
    print("   Appuyez sur Ctrl+C pour arrêter\n", file=sys.stderr)
    
    last_time = None
    last_display_time = time.time()
    dt_history = []
    
    try:
        while True:
            m = imu.read_measurement(timeout_s=1.0)
            if m is None:
                continue
            
            current_time = time.time()
            
            # Calcul du dt
            if last_time is not None:
                dt = current_time - last_time
                dt_history.append(dt)
                # Garder seulement les 100 derniers dt
                if len(dt_history) > 100:
                    dt_history.pop(0)
            else:
                dt = 0.01
            last_time = current_time
            
            # Enregistrement
            acc = np.array([m["ax"], m["ay"], m["az"]])
            gyr = np.array([m["gx"], m["gy"], m["gz"]])
            mag = np.array([m["mx"], m["my"], m["mz"]])
            temp = m.get("temp", 0.0)
            
            logger.log_sample(current_time, m["seq"], acc, gyr, mag, temp)
            
            # Affichage toutes les 0.5s
            if time.time() - last_display_time > 0.5:
                dt_mean = np.mean(dt_history) if dt_history else 0.01
                dt_std = np.std(dt_history) if dt_history else 0.0
                hz_actual = 1.0 / dt_mean if dt_mean > 0 else 0
                
                elapsed = current_time - logger.start_time
                sys.stderr.write(
                    f"\r⏱️  {elapsed:.1f}s | "
                    f"Échantillons: {logger.sample_count} | "
                    f"dt: {dt_mean*1000:.1f}±{dt_std*1000:.1f}ms | "
                    f"Hz: {hz_actual:.1f}"
                )
                sys.stderr.flush()
                last_display_time = time.time()
            
            # Arrêt si durée atteinte
            if duration_seconds and (current_time - logger.start_time) >= duration_seconds:
                break
                
    except KeyboardInterrupt:
        print("\n\n⏹️  Arrêt demandé", file=sys.stderr)
    finally:
        # Sauvegarde des métadonnées finales
        if dt_history:
            metadata["sampling"] = {
                "dt_mean_ms": float(np.mean(dt_history) * 1000),
                "dt_std_ms": float(np.std(dt_history) * 1000),
                "dt_min_ms": float(np.min(dt_history) * 1000),
                "dt_max_ms": float(np.max(dt_history) * 1000),
                "frequency_hz": float(1.0 / np.mean(dt_history)),
            }
        
        logger.save_metadata(metadata)
        imu.close()
        
        print(f"\n✅ Enregistrement terminé : {logger.log_file}", file=sys.stderr)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enregistrement des données IMU")
    parser.add_argument("-d", "--duration", type=float, default=None,
                       help="Durée d'enregistrement en secondes (défaut: infini)")
    parser.add_argument("--no-stats", action="store_true",
                       help="Ne pas afficher les statistiques d'initialisation")
    
    args = parser.parse_args()
    
    run_imu_logger(
        duration_seconds=args.duration,
        output_stats=not args.no_stats
    )