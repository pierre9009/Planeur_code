#!/usr/bin/env python3
import argparse
import serial
import sys
import time
import re

def parse_args():
    parser = argparse.ArgumentParser(
        description="Logger IMU sur UART vers un .log compatible imu-calib"
    )
    parser.add_argument(
        "-p", "--port", required=True,default="/dev/serial0",
        help="Port serie, par exemple COM3 ou /dev/ttyUSB0"
    )
    parser.add_argument(
        "-b", "--baudrate", type=int, default=230400,
        help="Baudrate serie (defaut: 115200)"
    )
    parser.add_argument(
        "-o", "--output", default="imu0.log",
        help="Fichier de sortie .log (defaut: imu0.log)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Afficher aussi les valeurs sur le terminal"
    )
    return parser.parse_args()

# Expression reguliere pour extraire des floats dans une ligne
FLOAT_RE = re.compile(
    r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
)

def extract_six_floats(line):
    """
    Essaie d'extraire exactement 6 flottants dans la ligne.
    Renvoie une liste de 6 floats ou None.
    """
    matches = FLOAT_RE.findall(line)
    if len(matches) < 6:
        return None
    # On prend les 6 premiers seulement
    try:
        values = [float(x) for x in matches[:6]]
    except ValueError:
        return None
    return values

def main():
    args = parse_args()

    print(f"Ouverture du port {args.port} a {args.baudrate} bauds")
    try:
        ser = serial.Serial(
            args.port,
            args.baudrate,
            timeout=1
        )
    except serial.SerialException as e:
        print(f"Erreur ouverture port serie: {e}")
        sys.exit(1)

    # Petit delai pour laisser l'Arduino rebooter si besoin
    time.sleep(2.0)
    ser.reset_input_buffer()

    print(f"Log des donnees dans {args.output}")
    print("Ctrl+C pour arreter")

    with open(args.output, "w") as f:
        try:
            while True:
                line_bytes = ser.readline()
                if not line_bytes:
                    continue

                try:
                    line = line_bytes.decode("utf-8", errors="ignore").strip()
                except UnicodeDecodeError:
                    continue

                if not line:
                    continue

                values = extract_six_floats(line)
                if values is None:
                    # Ligne pas exploitable, tu peux enlever ce print si ca spam trop
                    # print("Ligne ignoree:", line)
                    continue

                ax, ay, az, gx, gy, gz = values

                # Ecriture au format: ax ay az gx gy gz
                f.write(f"{ax} {ay} {az} {gx} {gy} {gz}\n")
                f.flush()

                if args.show:
                    print(f"{ax:.6f} {ay:.6f} {az:.6f}  {gx:.6f} {gy:.6f} {gz:.6f}")

        except KeyboardInterrupt:
            print("\nArret demande par l'utilisateur")

    ser.close()
    print("Port serie ferme, log termine")

if __name__ == "__main__":
    main()
