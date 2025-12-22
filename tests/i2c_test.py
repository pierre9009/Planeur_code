#!/usr/bin/env python3
import time
import struct
from smbus2 import SMBus, i2c_msg

I2C_BUS = 1
ARDUINO_ADDR = 0x42
HZ = 100.0
PERIOD_S = 1.0 / HZ

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

def read_packet(bus: SMBus) -> bytes:
    rd = i2c_msg.read(ARDUINO_ADDR, PKT_SIZE)
    bus.i2c_rdwr(rd)
    return bytes(rd)

def unpack_packet(pkt: bytes):
    seq, ax, ay, az, gx, gy, gz, mx, my, mz, tempC, rx_crc = struct.unpack(FMT, pkt)
    calc_crc = crc16_ccitt(pkt[:-2])
    ok = (calc_crc == rx_crc)
    return ok, seq, (ax, ay, az), (gx, gy, gz), (mx, my, mz), tempC, rx_crc, calc_crc

def main():
    print(f"Reading {PKT_SIZE} bytes from 0x{ARDUINO_ADDR:02X} at {HZ:.1f} Hz")

    with SMBus(I2C_BUS) as bus:
        next_t = time.perf_counter()
        last_seq = None
        bad_crc = 0
        total = 0

        while True:
            next_t += PERIOD_S

            try:
                pkt = read_packet(bus)
                ok, seq, acc, gyr, mag, tempC, rx_crc, calc_crc = unpack_packet(pkt)
                total += 1

                if not ok:
                    bad_crc += 1
                    print(f"CRC fail seq={seq} rx=0x{rx_crc:04X} calc=0x{calc_crc:04X} bad={bad_crc}/{total}")
                else:
                    if last_seq is not None and seq != (last_seq + 1):
                        print(f"Seq jump {last_seq} -> {seq}")
                    last_seq = seq

                    ax, ay, az = acc
                    gx, gy, gz = gyr
                    mx, my, mz = mag

                    print(
                        f"{seq}\t"
                        f"{ax:.4f}\t{ay:.4f}\t{az:.4f}\t"
                        f"{gx:.4f}\t{gy:.4f}\t{gz:.4f}\t"
                        f"{mx:.2f}\t{my:.2f}\t{mz:.2f}\t"
                        f"{tempC:.2f}"
                    )

            except OSError as e:
                print(f"I2C error: {e}")

            now = time.perf_counter()
            sleep_s = next_t - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = now

if __name__ == "__main__":
    main()
