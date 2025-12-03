import serial
import time

def parse_packet(line):
    """
    Parse a packet of the form:
    D,roll,pitch,gx,gy,gz,ax,ay,az
    Returns a dict.
    """
    try:
        if not line.startswith("D,"):
            return None

        parts = line.strip().split(',')
        if len(parts) != 10:
            return None

        return {
            "roll":  float(parts[1]),
            "pitch": float(parts[2]),
            "gx":    float(parts[3]),
            "gy":    float(parts[4]),
            "gz":    float(parts[5]),
            "ax":    float(parts[6]),
            "ay":    float(parts[7]),
            "az":    float(parts[8])
        }
    except:
        return None


def main():
    # Ouvre l'UART du Pi a 230400 bauds
    ser = serial.Serial(
        port="/dev/serial0",
        baudrate=230400,
        timeout=1
    )

    time.sleep(0.2)
    print("Lecture UART en cours...")

    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            data = parse_packet(line)
            if data is None:
                continue

            print(
                f"Roll={data['roll']:.2f}  "
                f"Pitch={data['pitch']:.2f}  "
                f"G=({data['gx']:.2f},{data['gy']:.2f},{data['gz']:.2f})  "
                f"A=({data['ax']:.2f},{data['ay']:.2f},{data['az']:.2f})"
            )

        except KeyboardInterrupt:
            print("\nFin du programme")
            ser.close()
            break
        except Exception as e:
            print("Erreur:", e)


if __name__ == "__main__":
    main()
