# gps_base.py
import serial
import time
import RPi.GPIO as GPIO

PORT = "/dev/serial0"
BAUDRATE = 9600

EN_PIN = 0  # attention: GPIO0 en mode BCM

GPIO.setmode(GPIO.BCM)
GPIO.setup(EN_PIN, GPIO.OUT)

# Activer le GPS (EN à l'état haut)
GPIO.output(EN_PIN, GPIO.HIGH)

# Commande UBX pour activer GGA (Enable GPGGA) - datasheet
ENABLE_GGA = bytes.fromhex("B5 62 06 01 03 00 F0 00 01 FB 10")

# Activer RMC (Enable GPRMC) - datasheet
ENABLE_RMC = bytes.fromhex("B5 62 06 01 03 00 F0 04 01 FF 18")

# Définir output à 1 Hz - datasheet
SET_RATE_1HZ = bytes.fromhex("B5 62 06 08 06 00 E8 03 01 00 01 00 01 39")

# Cold Start - datasheet 8.5 Other Settings
COLD_START = bytes.fromhex("B5 62 06 04 04 00 FF FF 02 00 0E 61")


def send_command(ser, cmd, label):
    """Envoie une commande GPS et confirme."""
    ser.write(cmd)
    ser.flush()
    print(f"[GPS] Commande envoyée : {label}")
    time.sleep(0.2)


def init_gps(ser, cold_start=True):
    print("Initialisation du module GPS...")
    time.sleep(0.5)  # >= 300 ms pour laisser le module démarrer

    if cold_start:
        print("[GPS] Cold start")
        send_command(ser, COLD_START, "COLD START")
        time.sleep(2.0)

    # Config des phrases NMEA et de la fréquence
    send_command(ser, ENABLE_GGA, "Enable GGA")
    send_command(ser, ENABLE_RMC, "Enable RMC")
    send_command(ser, SET_RATE_1HZ, "Set rate 1 Hz")

    print("\n--- Lecture NMEA activée ---\n")


def open_gps(cold_start=True):
    """Ouvre le port série et initialise le GPS, renvoie l'objet serial."""
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    init_gps(ser, cold_start=cold_start)
    return ser


def nmea_to_decimal_latlon(lat_str, lat_hemi, lon_str, lon_hemi):
    """Convertit lat / lon NMEA (ddmm.mmmm / dddmm.mmmm) en décimal."""
    if not lat_str or not lon_str:
        return None, None

    try:
        # latitude: ddmm.mmmm
        lat_deg = int(lat_str[0:2])
        lat_min = float(lat_str[2:])
        lat = lat_deg + lat_min / 60.0
        if lat_hemi == "S":
            lat = -lat

        # longitude: dddmm.mmmm
        lon_deg = int(lon_str[0:3])
        lon_min = float(lon_str[3:])
        lon = lon_deg + lon_min / 60.0
        if lon_hemi == "W":
            lon = -lon

        return lat, lon
    except (ValueError, IndexError):
        return None, None


def wait_for_fix(ser, verbose=True):
    """
    Bloque jusqu'à recevoir une trame RMC valide.
    Renvoie (lat, lon) en degrés décimaux.
    """
    print("[GPS] Attente d'un fix valide (RMC statut A)...")

    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue

        if verbose:
            print(">>", line)

        # On cherche les trames RMC
        if line.startswith("$GPRMC") or line.startswith("$GNRMC"):
            parts = line.split(",")
            if len(parts) < 7:
                continue

            status = parts[2]  # champ statut: A = actif, V = void
            lat_str = parts[3]
            lat_hemi = parts[4]
            lon_str = parts[5]
            lon_hemi = parts[6]

            if status != "A":
                continue  # pas encore de fix

            lat, lon = nmea_to_decimal_latlon(lat_str, lat_hemi, lon_str, lon_hemi)
            if lat is not None and lon is not None:
                print(f"[GPS] Fix valide, lat={lat}, lon={lon}")
                return lat, lon


def read_loop():
    """Boucle simple de debug qui affiche tout ce qui sort du GPS."""
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    try:
        init_gps(ser, cold_start=True)
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                print(">>", line)
    except KeyboardInterrupt:
        print("\nArrêt")
    finally:
        ser.close()
        GPIO.output(EN_PIN, GPIO.LOW)
        GPIO.cleanup()


if __name__ == "__main__":
    # mode debug: on lit brute
    read_loop()
