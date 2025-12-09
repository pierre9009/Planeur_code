import serial
import time
import RPi.GPIO as GPIO

PORT = "/dev/serial0"
BAUDRATE = 9600

EN_PIN = 0

GPIO.setmode(GPIO.BCM)
GPIO.setup(EN_PIN, GPIO.OUT)

# Activer le GPS (EN à l'état haut)
GPIO.output(EN_PIN, GPIO.HIGH)

# Commande UBX pour activer GGA (Enable GPGGA) - datasheet
# b5 62 06 01 03 00 f0 00 01 fb 10
ENABLE_GGA = bytes.fromhex("B5 62 06 01 03 00 F0 00 01 FB 10")

# Activer RMC (Enable GPRMC) - datasheet
# b5 62 06 01 03 00 f0 04 01 ff 18
ENABLE_RMC = bytes.fromhex("B5 62 06 01 03 00 F0 04 01 FF 18")

# Définir output à 1 Hz - datasheet
# B5 62 06 08 06 00 E8 03 01 00 01 00 01 39
SET_RATE_1HZ = bytes.fromhex("B5 62 06 08 06 00 E8 03 01 00 01 00 01 39")

# Cold Start - datasheet 8.5 Other Settings
# B5 62 06 04 04 00 FF FF 02 00 0E 61
COLD_START = bytes.fromhex("B5 62 06 04 04 00 FF FF 02 00 0E 61")


def send_command(ser, cmd, label):
    """Envoie une commande GPS et confirme."""
    ser.write(cmd)
    ser.flush()
    print(f"[GPS] Commande envoyée : {label}")
    time.sleep(0.2)


def init_gps(ser, cold_start=True):
    print("Initialisation du module GPS...")

    # La doc dit que le module met environ 300 ms à démarrer
    time.sleep(0.5)

    if cold_start:
        print("[GPS] Cold start")
        send_command(ser, COLD_START, "COLD START")
        # Le module redémarre, on lui laisse un peu de temps
        time.sleep(2.0)

    # Config des phrases NMEA et de la fréquence
    send_command(ser, ENABLE_GGA, "Enable GGA")
    send_command(ser, ENABLE_RMC, "Enable RMC")
    send_command(ser, SET_RATE_1HZ, "Set rate 1 Hz")

    print("\n--- Lecture NMEA activée ---\n")


def main():
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)

    try:
        # Init avec cold start au démarrage
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
    main()
