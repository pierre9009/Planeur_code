import time
import struct
import pigpio

RX_GPIO = 23
BAUD = 115200  # commence à 115200, puis monte à 230400 si tout est propre

SYNC1 = 0xAA
SYNC2 = 0x55

FMT = "<I10fH"
PKT_SIZE = struct.calcsize(FMT)  # 46

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

def main():
    pi = pigpio.pi()
    if not pi.connected:
        print("pigpio not connected. Is pigpiod running?")
        return

    pi.set_mode(RX_GPIO, pigpio.INPUT)
    pi.set_pull_up_down(RX_GPIO, pigpio.PUD_UP)

    pi.bb_serial_read_close(RX_GPIO)
    pi.bb_serial_read_open(RX_GPIO, BAUD, 8)

    print(f"Soft UART RX on GPIO{RX_GPIO} @ {BAUD} baud, payload={PKT_SIZE} bytes")

    buf = bytearray()
    last_seq = None
    bad_crc = 0
    total = 0

    try:
        while True:
            count, data = pi.bb_serial_read(RX_GPIO)
            if count > 0:
                buf.extend(data)

                while True:
                    # Cherche la sync
                    idx = buf.find(bytes([SYNC1, SYNC2]))
                    if idx < 0:
                        # garde juste le dernier octet au cas où c'est 0xAA
                        if len(buf) > 1:
                            buf[:] = buf[-1:]
                        break

                    # jette ce qu'il y a avant la sync
                    if idx > 0:
                        del buf[:idx]

                    # on a au moins 2 octets de sync
                    if len(buf) < 2 + PKT_SIZE:
                        break

                    payload = bytes(buf[2:2 + PKT_SIZE])
                    del buf[:2 + PKT_SIZE]

                    seq, ax, ay, az, gx, gy, gz, mx, my, mz, tempC, rx_crc = struct.unpack(FMT, payload)
                    calc = crc16_ccitt(payload[:-2])

                    total += 1
                    if rx_crc != calc:
                        bad_crc += 1
                        print(f"CRC fail seq={seq} rx=0x{rx_crc:04X} calc=0x{calc:04X} bad={bad_crc}/{total}")
                        continue

                    if last_seq is not None and seq != last_seq + 1:
                        print(f"Seq jump {last_seq} -> {seq}")
                    last_seq = seq

                    print(
                        f"{seq}\t"
                        f"{ax:.4f}\t{ay:.4f}\t{az:.4f}\t"
                        f"{gx:.4f}\t{gy:.4f}\t{gz:.4f}\t"
                        f"{mx:.2f}\t{my:.2f}\t{mz:.2f}\t"
                        f"{tempC:.2f}"
                    )

            time.sleep(0.001)

    finally:
        pi.bb_serial_read_close(RX_GPIO)
        pi.stop()

if __name__ == "__main__":
    main()
