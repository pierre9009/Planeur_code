import time
import struct
import pigpio

RX_GPIO = 24
BAUD = 115200

SYNC1 = 0xAA
SYNC2 = 0x55

# uint32 + 10 floats + uint16
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
        print("pigpio not connected")
        return

    pi.set_mode(RX_GPIO, pigpio.INPUT)
    pi.set_pull_up_down(RX_GPIO, pigpio.PUD_UP)

    try:
        pi.bb_serial_read_close(RX_GPIO)
    except pigpio.error:
        pass

    pi.bb_serial_read_open(RX_GPIO, BAUD, 8)
    pi.bb_serial_read(RX_GPIO)  # flush

    print(f"Listening on GPIO{RX_GPIO} @ {BAUD} baud")

    buf = bytearray()

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

                    seq, ax, ay, az, gx, gy, gz, mx, my, mz, tempC, rx_crc = struct.unpack(FMT, payload)

                    if crc16_ccitt(payload[:-2]) != rx_crc:
                        continue

                    print(
                        f"seq={seq:6d} | "
                        f"A=({ax:+.3f}, {ay:+.3f}, {az:+.3f}) m/s² | "
                        f"G=({gx:+.3f}, {gy:+.3f}, {gz:+.3f}) rad/s | "
                        f"M=({mx:+.2f}, {my:+.2f}, {mz:+.2f}) uT | "
                        f"T={tempC:.2f} C"
                    )

            time.sleep(0.001)

    finally:
        try:
            pi.bb_serial_read_close(RX_GPIO)
        except pigpio.error:
            pass
        pi.stop()

if __name__ == "__main__":
    main()
