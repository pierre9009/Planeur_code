import time
import struct
import pigpio

RX_GPIO = 23
BAUD = 115200

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

def main():
    pi = pigpio.pi()
    if not pi.connected:
        print("pigpio not connected. Start pigpiod: sudo systemctl start pigpiod")
        return

    pi.set_mode(RX_GPIO, pigpio.INPUT)
    pi.set_pull_up_down(RX_GPIO, pigpio.PUD_UP)

    # Close only if already open; pigpio throws otherwise
    try:
        pi.bb_serial_read_close(RX_GPIO)
    except pigpio.error:
        pass

    rc = pi.bb_serial_read_open(RX_GPIO, BAUD, 8)
    if rc != 0:
        print(f"bb_serial_read_open failed rc={rc}")
        pi.stop()
        return

    # Clear any old buffered bytes
    pi.bb_serial_read(RX_GPIO)

    print(f"Soft UART RX on GPIO{RX_GPIO} @ {BAUD} baud, payload={PKT_SIZE} bytes")

    buf = bytearray()
    last_seq = None
    bad_crc = 0
    total = 0
    last_rx_time = time.time()

    try:
        while True:
            count, data = pi.bb_serial_read(RX_GPIO)
            if count > 0:
                last_rx_time = time.time()
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

            else:
                if time.time() - last_rx_time > 2.0:
                    print("No RX data (check wiring, baud, and level shifting on Arduino TX -> Pi RX)")
                    last_rx_time = time.time()

            time.sleep(0.001)

    finally:
        try:
            pi.bb_serial_read_close(RX_GPIO)
        except pigpio.error:
            pass
        pi.stop()

if __name__ == "__main__":
    main()
