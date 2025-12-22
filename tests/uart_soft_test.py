import time
import struct
import pigpio

RX_GPIO = 24
BAUD = 57600

SYNC1 = 0xAA
SYNC2 = 0x55

FMT = "<I10fH"
PKT_SIZE = struct.calcsize(FMT)

STAT_PERIOD_S = 1.0

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

    try:
        pi.bb_serial_read_close(RX_GPIO)
    except pigpio.error:
        pass

    rc = pi.bb_serial_read_open(RX_GPIO, BAUD, 8)
    if rc != 0:
        print(f"bb_serial_read_open failed rc={rc}")
        pi.stop()
        return

    pi.bb_serial_read(RX_GPIO)  # flush

    print(f"Soft UART RX on GPIO{RX_GPIO} @ {BAUD} baud")

    buf = bytearray()
    last_seq = None
    total = 0
    bad_crc = 0
    jumps = 0

    last_stat_t = time.perf_counter()

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

                    seq, *rest, rx_crc = struct.unpack(FMT, payload)
                    calc = crc16_ccitt(payload[:-2])

                    total += 1

                    if rx_crc != calc:
                        bad_crc += 1
                        continue

                    if last_seq is not None and seq != last_seq + 1:
                        jumps += 1

                    last_seq = seq

            now = time.perf_counter()
            if now - last_stat_t >= STAT_PERIOD_S:
                err_rate = (bad_crc / total * 100.0) if total > 0 else 0.0
                print(
                    f"frames={total}  "
                    f"crc_err={bad_crc}  "
                    f"seq_jumps={jumps}  "
                    f"err_rate={err_rate:.2f}%"
                )
                last_stat_t = now

            time.sleep(0.001)

    finally:
        try:
            pi.bb_serial_read_close(RX_GPIO)
        except pigpio.error:
            pass
        pi.stop()

if __name__ == "__main__":
    main()
