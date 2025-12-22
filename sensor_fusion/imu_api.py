#!/usr/bin/env python3
import time
import struct
import pigpio

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

class ImuSoftUart:
    def __init__(self, rx_gpio: int = 24, baudrate: int = 57600):
        self.rx_gpio = rx_gpio
        self.baudrate = baudrate
        self.pi = None
        self.buf = bytearray()
        self.bad_crc = 0
        self.total = 0

    def open(self):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio not connected. Start pigpiod (sudo systemctl start pigpiod)")

        self.pi.set_mode(self.rx_gpio, pigpio.INPUT)
        self.pi.set_pull_up_down(self.rx_gpio, pigpio.PUD_OFF)

        try:
            self.pi.bb_serial_read_close(self.rx_gpio)
        except pigpio.error:
            pass

        rc = self.pi.bb_serial_read_open(self.rx_gpio, self.baudrate, 8)
        if rc != 0:
            self.pi.stop()
            self.pi = None
            raise RuntimeError(f"bb_serial_read_open failed rc={rc}")

        self.pi.bb_serial_read(self.rx_gpio)

    def close(self):
        if self.pi is None:
            return
        try:
            self.pi.bb_serial_read_close(self.rx_gpio)
        except pigpio.error:
            pass
        self.pi.stop()
        self.pi = None

    def _feed(self):
        count, data = self.pi.bb_serial_read(self.rx_gpio)
        if count > 0:
            self.buf.extend(data)

    def read_measurement(self, timeout_s: float = 1.0):
        t0 = time.perf_counter()
        while True:
            self._feed()

            while True:
                idx = self.buf.find(bytes([SYNC1, SYNC2]))
                if idx < 0:
                    if len(self.buf) > 1:
                        self.buf[:] = self.buf[-1:]
                    break

                if idx > 0:
                    del self.buf[:idx]

                if len(self.buf) < 2 + PKT_SIZE:
                    break

                payload = bytes(self.buf[2:2 + PKT_SIZE])
                del self.buf[:2 + PKT_SIZE]

                self.total += 1
                rx_crc = struct.unpack_from("<H", payload, PKT_SIZE - 2)[0]
                calc = crc16_ccitt(payload[:-2])
                if rx_crc != calc:
                    self.bad_crc += 1
                    continue

                seq, ax, ay, az, gx, gy, gz, mx, my, mz, tempC, _ = struct.unpack(FMT, payload)
                return {
                    "seq": int(seq),
                    "ax": float(ax), "ay": float(ay), "az": float(az), # m/s²
                    "gx": float(gx), "gy": float(gy), "gz": float(gz), #rad/s
                    "mx": float(mx), "my": float(my), "mz": float(mz), #uT
                    "tempC": float(tempC),
                    "ts": time.time(),
                }

            if time.perf_counter() - t0 > timeout_s:
                return None

            time.sleep(0.001)
