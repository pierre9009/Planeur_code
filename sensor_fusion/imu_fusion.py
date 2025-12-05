import serial
import time
import math
import numpy as np
import json
import sys

from ahrs.filters import Fourati
from ahrs.common.orientation import acc2q


def parse_packet(line: str):
    try:
        if not line.startswith("D,"):
            return None

        parts = line.strip().split(",")
        if len(parts) != 12:
            return None

        return {
            "roll_comp":  float(parts[1]),
            "pitch_comp": float(parts[2]),
            "gx_deg":     float(parts[3]),
            "gy_deg":     float(parts[4]),
            "gz_deg":     float(parts[5]),
            "ax":         float(parts[6]),
            "ay":         float(parts[7]),
            "az":         float(parts[8]),
            "mx":         float(parts[9]),
            "my":         float(parts[10]),
            "mz":         float(parts[11]),
        }
    except ValueError:
        return None


def quaternion_to_euler_zyx(q):
    """
    Convertit un quaternion [w, x, y, z] en angles d'Euler (roll, pitch, yaw)
    Convention: ZYX (Yaw-Pitch-Roll) intrinsèque
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Roll (rotation autour de X)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (rotation autour de Y)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Gimbal lock
    else:
        pitch = math.asin(sinp)
    
    # Yaw (rotation autour de Z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def run_imu_fusion():
    """
    Processus principal de fusion IMU.
    Envoie les données d'orientation sur stdout en JSON.
    """
    
    ser = serial.Serial(
        port="/dev/serial0",
        baudrate=230400,
        timeout=1.0,
    )

    time.sleep(0.2)
    print("Port série ouvert", file=sys.stderr)
    print("\n" + "="*70, file=sys.stderr)
    print("INITIALISATION - Placez le capteur À PLAT pendant 3 secondes...", file=sys.stderr)
    print("="*70 + "\n", file=sys.stderr)

    # Attendre et nettoyer le buffer
    time.sleep(1)
    for _ in range(10):
        ser.readline()
    
    # Diagnostic et initialisation
    diag_samples = []
    for _ in range(30):
        line = ser.readline().decode(errors="ignore").strip()
        data = parse_packet(line)
        if data:
            diag_samples.append(data)
    
    if not diag_samples:
        print("⚠️  ERREUR: Pas de données reçues!", file=sys.stderr)
        return

    # Calcul de la moyenne pour l'initialisation
    avg_ax = np.mean([d["ax"] for d in diag_samples])
    avg_ay = np.mean([d["ay"] for d in diag_samples])
    avg_az = np.mean([d["az"] for d in diag_samples])
    
    print(f"Accéléromètre moyen (à plat):", file=sys.stderr)
    print(f"  X: {avg_ax:.2f} m/s²", file=sys.stderr)
    print(f"  Y: {avg_ay:.2f} m/s²", file=sys.stderr)
    print(f"  Z: {avg_az:.2f} m/s²", file=sys.stderr)
    
    # Déterminer l'orientation du capteur
    if abs(avg_az) > 8.0:  # L'axe Z est vertical
        if avg_az > 0:
            print("→ Capteur à plat, Z pointe VERS LE HAUT", file=sys.stderr)
            z_sign = 1
        else:
            print("→ Capteur à plat, Z pointe VERS LE BAS", file=sys.stderr)
            z_sign = -1
    else:
        print(f"⚠️  ATTENTION: Le capteur n'est pas à plat!", file=sys.stderr)
        print(f"   Gravité mesurée: {math.sqrt(avg_ax**2 + avg_ay**2 + avg_az**2):.2f} m/s²", file=sys.stderr)
        z_sign = 1
    
    print("="*70 + "\n", file=sys.stderr)

    sample_freq = 50.0
    magnetic_dip_deg = 64.0  # Inclinaison magnétique à Paris

    # Configuration Fourati avec gain plus élevé
    fourati = Fourati(
        frequency=sample_freq,
        magnetic_dip=magnetic_dip_deg,
        gain=1.0,  # Gain augmenté pour plus de réactivité
    )

    # Initialisation du quaternion avec l'accéléromètre moyen
    init_acc = np.array([avg_ax, avg_ay, avg_az * z_sign])
    init_acc = init_acc / np.linalg.norm(init_acc)
    q = acc2q(init_acc)
    q = q / np.linalg.norm(q)
    
    print(f"Quaternion initial: [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]", file=sys.stderr)
    
    # Test de conversion
    test_roll, test_pitch, test_yaw = quaternion_to_euler_zyx(q)
    print(f"Angles initiaux: R={math.degrees(test_roll):.2f}° P={math.degrees(test_pitch):.2f}° Y={math.degrees(test_yaw):.2f}°", file=sys.stderr)
    print(file=sys.stderr)

    last_t = time.time()

    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            now = time.time()
            dt = now - last_t
            if dt <= 0.0:
                dt = 1.0 / sample_freq
            last_t = now

            data = parse_packet(line)
            if data is None:
                continue

            # Gyroscope en rad/s (GARDER L'ORDRE ORIGINAL)
            gx_rad = data["gx_deg"] * math.pi / 180.0
            gy_rad = data["gy_deg"] * math.pi / 180.0
            gz_rad = data["gz_deg"] * math.pi / 180.0
            gyr = np.array([gx_rad, gy_rad, gz_rad * z_sign], dtype=float)

            # Accéléromètre (GARDER L'ORDRE ORIGINAL)
            acc = np.array([data["ax"], data["ay"], data["az"] * z_sign], dtype=float)
            
            # NORMALISATION CRITIQUE
            acc_norm = np.linalg.norm(acc)
            if acc_norm > 0.1:
                acc = acc / acc_norm
            else:
                continue  # Skip ce sample si l'accéléromètre est invalide

            # Magnétomètre µT -> mT (GARDER L'ORDRE ORIGINAL)
            mag = np.array([data["mx"], data["my"], data["mz"] * z_sign], dtype=float) / 1000.0
            
            # NORMALISATION CRITIQUE
            mag_norm = np.linalg.norm(mag)
            if mag_norm > 0.001:
                mag = mag / mag_norm
            else:
                # Si le magnétomètre est invalide, utiliser seulement acc + gyro
                mag = None

            # Mise à jour Fourati
            if mag is not None:
                q = fourati.update(
                    q=q,
                    gyr=gyr,
                    acc=acc,
                    mag=mag,
                    dt=dt,
                )
            else:
                # Mode sans magnétomètre (IMU 6 axes)
                q = fourati.update(
                    q=q,
                    gyr=gyr,
                    acc=acc,
                    mag=np.array([1.0, 0.0, 0.0]),  # Magnétomètre fictif
                    dt=dt,
                )

            # Normalisation du quaternion
            q = q / np.linalg.norm(q)

            # Conversion en angles d'Euler (convention ZYX)
            roll_f, pitch_f, yaw_f = quaternion_to_euler_zyx(q)
            
            roll_f_deg  = math.degrees(roll_f)
            pitch_f_deg = math.degrees(pitch_f)
            yaw_f_deg   = math.degrees(yaw_f)

            # Envoi des données via stdout en JSON
            orientation_data = {
                'roll': roll_f_deg,
                'pitch': pitch_f_deg,
                'yaw': yaw_f_deg,
                'roll_comp': data["roll_comp"],
                'pitch_comp': data["pitch_comp"],
                'dt': dt * 1000.0
            }
            
            # Envoyer JSON sur stdout
            print(json.dumps(orientation_data), flush=True)

            # Log sur stderr
            print(
                f"COMP R={data['roll_comp']:7.2f} P={data['pitch_comp']:7.2f} | "
                f"FOURATI R={roll_f_deg:7.2f} P={pitch_f_deg:7.2f} Y={yaw_f_deg:7.2f} | "
                f"dt={dt*1000:.1f}ms",
                file=sys.stderr
            )

        except Exception as e:
            print("Erreur IMU fusion:", e, file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)


if __name__ == '__main__':
    run_imu_fusion()