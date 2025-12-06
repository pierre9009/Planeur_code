import serial
import time

def main():
    ser = serial.Serial(
        port="/dev/serial0",   # verifie avec ls -l /dev/serial*
        baudrate=230400,       # doit matcher Serial1.begin(230400)
        timeout=1.0,
    )

    time.sleep(0.2)
    print("Lecture brute de /dev/serial0...")

    while True:
        try:
            line = ser.readline()
            if not line:
                continue
            try:
                txt = line.decode(errors="ignore").strip()
            except UnicodeDecodeError:
                continue

            print(repr(txt))   # afficher avec quotes pour voir les caracteres

        except KeyboardInterrupt:
            print("\nStop.")
            ser.close()
            break
        except Exception as e:
            print("Erreur:", e)

if __name__ == "__main__":
    main()
