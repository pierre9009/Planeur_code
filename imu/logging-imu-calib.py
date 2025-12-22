#!/usr/bin/env python3
import argparse
import struct
import time
import pigpio
import sys

SYNC1 = 0xAA
SYNC2 = 0x55

FMT = "<I10fH"
PKT_SIZE = struct.calcsize(FMT)

def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    crc = init
    for b in data:
        crc ^= (b << 8) & 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc

def parse_args():
    p = argparse.ArgumentParser(
        description="Logger IMU binaire (UART soft pigpio) vers .log imu-calib"
    )
    p.add_argument(
        "--rx-gpio", type=int, default=24,
        help="GPIO BCM pour RX UART soft (defaut: 24)"
    )
    p.add_argument(
        "-b", "--baudrate", type=int, default=57600,
        help="Baudrate UART (defaut: 57600)"
    )
    p.add_argument(
        "-o", "--output", default="imu0.log",
        help="Fichier .log de sortie (defaut: imu0.log)"
    )
    p.add_argument(
        "--show", action="store_true",
        help="Afficher aussi les valeurs sur le terminal"
    )
    return p.parse_args()

def main():
    args = parse_args()

    pi = pigpio.pi()
    if not pi.connected:
        print("Erreur: pigpiod non actif")
        sys.exit(1)

    RX_GPIO = args.rx_gpio

    pi.set_mode(RX_GPIO, pigpio.INPUT)
    pi.set_pull_up_down(RX_GPIO, pigpio.PUD_OFF)

    try:
        pi.bb_serial_read_close(RX_GPIO)
    except pigpio.error:
        pass

    rc = pi.bb_serial_read_open(RX_GPIO, args.baudrate, 8)
    if rc != 0:
        print(f"Erreur ouverture UART soft rc={rc}")
        pi.stop()
        sys.exit(1)

    pi.bb_serial_read(RX_GPIO)  # flush

    print(f"Logging IMU sur GPIO{RX_GPIO} @ {args.baudrate} bauds")
    print(f"Fichier: {args.output}")
    print("Ctrl+C pour arreter")

    buf = bytearray()
    frames = 0
    bad = 0

    with open(args.output, "w") as f:
        try:
            while True:
                count, data = pi.bb_serial_read(RX_GPIO)
                if count > 0:
                    buf.extend(data)

                    while True:
                        idx = buf.find(bytes([SYNC1, SYNC2]))
                        if idx < 0:
                            if len(buf) > 1:
                                buf[:] = buf[-1:]
                            break

                        if idx > 0:
                            del buf[:idx]

                        if len(buf) < 2 + PKT_SIZE:
                            break

                        payload = bytes(buf[2:2 + PKT_SIZE])
                        del buf[:2 + PKT_SIZE]

                        (
                            seq,
                            ax, ay, az,
                            gx, gy, gz,
                            *_,
                            rx_crc
                        ) = struct.unpack(FMT, payload)

                        calc = crc16_ccitt(payload[:-2])
                        if rx_crc != calc:
                            bad += 1
                            continue

                        frames += 1

                        # Format imu-calib: ax ay az gx gy gz
                        f.write(f"{ax} {ay} {az} {gx} {gy} {gz}\n")

                        if args.show:
                            print(
                                f"{frames:6d}  "
                                f"{ax:+.6f} {ay:+.6f} {az:+.6f}  "
                                f"{gx:+.6f} {gy:+.6f} {gz:+.6f}"
                            )

                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\nArret utilisateur")

    pi.bb_serial_read_close(RX_GPIO)
    pi.stop()

    print(f"Log termine: {frames} trames OK, {bad} trames CRC ignorees")

if __name__ == "__main__":
    main()

#python imu_logger.py --rx-gpio 24 -b 57600 -o imu0.log --show