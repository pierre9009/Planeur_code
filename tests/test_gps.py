import serial
import time
import RPi.GPIO as GPIO

PORT = "/dev/serial0"
BAUDRATE = 9600

EN_PIN = 11


GPIO.setmode(GPIO.BCM)
GPIO.setup(EN_PIN, GPIO.OUT)

# Activer le GPS
GPIO.output(EN_PIN, GPIO.HIGH)

# Commande UBX pour activer GGA (page 16 – Enable GPGGA)
ENABLE_GGA = bytes.fromhex("B5 62 06 01 03 00 F0 00 01 FB 11")

# Activer RMC (page 16 – Enable GPRMC)
ENABLE_RMC = bytes.fromhex("B5 62 06 01 03 00 F0 04 01 FF 18")

# Définir output à 1 Hz (exemple)
SET_RATE_1HZ = bytes.fromhex("B5 62 06 08 06 00 E8 03 01 00 01 00 01 39")


def send_command(ser, cmd, label):
    """Envoie une commande GPS et confirme."""
    ser.write(cmd)
    ser.flush()
    print(f"[GPS] Commande envoyée : {label}")
    time.sleep(0.2)


def main():
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print("Initialisation du module GPS...")

    # Attendre que le module soit initialisé
    time.sleep(0.5)

    # Envoyer les commandes d’activation
    send_command(ser, ENABLE_GGA, "GGA")
    send_command(ser, ENABLE_RMC, "RMC")
    send_command(ser, SET_RATE_1HZ, "RATE 1Hz")

    print("\n--- Lecture NMEA activée ---\n")

    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                print(">>", line)

    except KeyboardInterrupt:
        print("\nArrêt")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
