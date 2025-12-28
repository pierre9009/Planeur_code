"""UART communication for IMU data reception."""

import struct
import time
import logging
from typing import Optional

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

from core.types import ImuReading, SensorStats
from core.config import Config

logger = logging.getLogger(__name__)

SYNC1 = 0xAA
SYNC2 = 0x55
PACKET_FORMAT = "<I10fH"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)


class UartError(Exception):
    """Base exception for UART communication errors."""
    pass


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    """Calculate CRC-16-CCITT checksum.

    Args:
        data: Bytes to checksum.
        init: Initial CRC value.

    Returns:
        16-bit CRC value.
    """
    crc = init
    for b in data:
        crc ^= (b << 8) & 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


class ImuUart:
    """Hardware UART interface for IMU data reception.

    Handles binary protocol with sync bytes, CRC validation,
    and packet parsing. Thread-safe for single reader.
    """

    def __init__(self, config: Config):
        """Initialize UART interface.

        Args:
            config: System configuration with UART settings.

        Raises:
            ImportError: If pyserial is not installed.
        """
        if not SERIAL_AVAILABLE:
            raise ImportError("pyserial is required for UART communication")

        self._config = config
        self._port = config.uart.port
        self._baudrate = config.uart.baudrate
        self._timeout = config.uart.timeout_s
        self._write_timeout = config.uart.write_timeout_s

        self._serial: Optional[serial.Serial] = None
        self._buffer = bytearray()
        self._stats = SensorStats()
        self._is_open = False

    def open(self) -> None:
        """Open serial connection.

        Raises:
            UartError: If connection cannot be established.
        """
        if self._is_open:
            return

        try:
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self._timeout,
                write_timeout=self._write_timeout,
            )
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
            time.sleep(0.1)
            self._is_open = True
            logger.info("UART opened: %s @ %d baud", self._port, self._baudrate)

        except serial.SerialException as e:
            raise UartError(f"Failed to open {self._port}: {e}") from e

    def close(self) -> None:
        """Close serial connection."""
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
            self._serial = None
        self._is_open = False
        logger.info("UART closed")

    def read_measurement(self, timeout_s: float = 1.0) -> Optional[ImuReading]:
        """Read a complete IMU measurement.

        Blocks until a valid packet is received or timeout expires.

        Args:
            timeout_s: Maximum time to wait for a packet.

        Returns:
            ImuReading if successful, None on timeout.

        Raises:
            UartError: If connection is not open.
        """
        if not self._is_open or self._serial is None:
            raise UartError("UART not open")

        deadline = time.perf_counter() + timeout_s

        while True:
            self._feed()
            reading = self._try_parse_packet()
            if reading is not None:
                return reading

            if time.perf_counter() > deadline:
                self._stats.timeouts += 1
                return None

            time.sleep(0.001)

    def _feed(self) -> None:
        """Read available data from serial port into buffer."""
        if self._serial is not None and self._serial.in_waiting > 0:
            data = self._serial.read(self._serial.in_waiting)
            self._buffer.extend(data)

    def _try_parse_packet(self) -> Optional[ImuReading]:
        """Try to parse a packet from the buffer.

        Returns:
            ImuReading if valid packet found, None otherwise.
        """
        while True:
            idx = self._buffer.find(bytes([SYNC1, SYNC2]))

            if idx < 0:
                if len(self._buffer) > 1:
                    self._buffer[:] = self._buffer[-1:]
                return None

            if idx > 0:
                del self._buffer[:idx]

            if len(self._buffer) < 2 + PACKET_SIZE:
                return None

            payload = bytes(self._buffer[2:2 + PACKET_SIZE])
            del self._buffer[:2 + PACKET_SIZE]

            self._stats.total_packets += 1

            rx_crc = struct.unpack_from("<H", payload, PACKET_SIZE - 2)[0]
            calc_crc = crc16_ccitt(payload[:-2])

            if rx_crc != calc_crc:
                self._stats.crc_errors += 1
                logger.debug("CRC error: received 0x%04X, expected 0x%04X", rx_crc, calc_crc)
                continue

            self._stats.valid_packets += 1
            return self._decode_packet(payload)

    def _decode_packet(self, payload: bytes) -> ImuReading:
        """Decode a validated packet into ImuReading.

        Args:
            payload: Raw packet bytes (without sync bytes).

        Returns:
            Parsed IMU reading.
        """
        seq, ax, ay, az, gx, gy, gz, mx, my, mz, temp_c, _ = struct.unpack(
            PACKET_FORMAT, payload
        )

        return ImuReading(
            seq=int(seq),
            timestamp=time.time(),
            ax=float(ax),
            ay=float(ay),
            az=float(az),
            gx=float(gx),
            gy=float(gy),
            gz=float(gz),
            mx=float(mx),
            my=float(my),
            mz=float(mz),
            temperature=float(temp_c),
        )

    @property
    def stats(self) -> SensorStats:
        """Get communication statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset communication statistics."""
        self._stats = SensorStats()

    @property
    def is_open(self) -> bool:
        """Check if connection is open."""
        return self._is_open

    def __enter__(self) -> "ImuUart":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


class MockImuUart:
    """Mock UART interface for testing.

    Generates synthetic IMU data for unit tests and development
    without hardware.
    """

    def __init__(self, config: Config):
        """Initialize mock UART.

        Args:
            config: System configuration.
        """
        self._config = config
        self._stats = SensorStats()
        self._seq = 0
        self._is_open = False
        self._sample_rate = config.sensor.sample_rate_hz
        self._last_time = 0.0

    def open(self) -> None:
        """Simulate opening connection."""
        self._is_open = True
        self._last_time = time.time()
        logger.info("Mock UART opened")

    def close(self) -> None:
        """Simulate closing connection."""
        self._is_open = False
        logger.info("Mock UART closed")

    def read_measurement(self, timeout_s: float = 1.0) -> Optional[ImuReading]:
        """Generate synthetic IMU reading.

        Simulates a stationary IMU with gravity pointing down
        and a nominal magnetic field.

        Args:
            timeout_s: Ignored in mock.

        Returns:
            Synthetic IMU reading.
        """
        if not self._is_open:
            raise UartError("Mock UART not open")

        dt = 1.0 / self._sample_rate
        time.sleep(dt * 0.9)

        self._seq += 1
        self._stats.total_packets += 1
        self._stats.valid_packets += 1

        import numpy as np
        noise_acc = np.random.normal(0, 0.01, 3)
        noise_gyr = np.random.normal(0, 0.001, 3)
        noise_mag = np.random.normal(0, 0.1, 3)

        return ImuReading(
            seq=self._seq,
            timestamp=time.time(),
            ax=noise_acc[0],
            ay=noise_acc[1],
            az=9.81 + noise_acc[2],
            gx=noise_gyr[0],
            gy=noise_gyr[1],
            gz=noise_gyr[2],
            mx=20.0 + noise_mag[0],
            my=5.0 + noise_mag[1],
            mz=45.0 + noise_mag[2],
            temperature=25.0,
        )

    @property
    def stats(self) -> SensorStats:
        """Get mock statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = SensorStats()

    @property
    def is_open(self) -> bool:
        """Check if mock is open."""
        return self._is_open

    def __enter__(self) -> "MockImuUart":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
