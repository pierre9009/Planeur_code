"""Tests for UART communication."""

import pytest
import struct

from core.config import Config
from communication.uart import crc16_ccitt, MockImuUart, SYNC1, SYNC2, PACKET_FORMAT


class TestCrc16:
    """Tests for CRC-16-CCITT function."""

    def test_empty_data(self):
        """Empty data should return initial value."""
        crc = crc16_ccitt(b"", init=0xFFFF)
        assert crc == 0xFFFF

    def test_known_value(self):
        """CRC should match known test value."""
        data = b"123456789"
        crc = crc16_ccitt(data)
        assert crc == 0x29B1

    def test_different_init(self):
        """Different init value should give different result."""
        data = b"test"
        crc1 = crc16_ccitt(data, init=0xFFFF)
        crc2 = crc16_ccitt(data, init=0x0000)
        assert crc1 != crc2

    def test_deterministic(self):
        """Same input should give same output."""
        data = b"deterministic test"
        crc1 = crc16_ccitt(data)
        crc2 = crc16_ccitt(data)
        assert crc1 == crc2


class TestMockImuUart:
    """Tests for MockImuUart class."""

    def test_open_close(self, config):
        """Mock should open and close without error."""
        mock = MockImuUart(config)

        assert not mock.is_open
        mock.open()
        assert mock.is_open
        mock.close()
        assert not mock.is_open

    def test_context_manager(self, config):
        """Mock should work as context manager."""
        with MockImuUart(config) as mock:
            assert mock.is_open
        assert not mock.is_open

    def test_read_measurement(self, config):
        """Mock should return valid readings."""
        with MockImuUart(config) as mock:
            reading = mock.read_measurement()

            assert reading is not None
            assert reading.seq == 1
            assert abs(reading.az - 9.81) < 0.1
            assert reading.temperature == 25.0

    def test_sequential_readings(self, config):
        """Sequential readings should have incrementing sequence numbers."""
        with MockImuUart(config) as mock:
            r1 = mock.read_measurement()
            r2 = mock.read_measurement()
            r3 = mock.read_measurement()

            assert r1.seq == 1
            assert r2.seq == 2
            assert r3.seq == 3

    def test_stats_updated(self, config):
        """Stats should be updated after readings."""
        with MockImuUart(config) as mock:
            for _ in range(10):
                mock.read_measurement()

            stats = mock.stats
            assert stats.total_packets == 10
            assert stats.valid_packets == 10
            assert stats.crc_errors == 0

    def test_reset_stats(self, config):
        """Reset should clear statistics."""
        with MockImuUart(config) as mock:
            for _ in range(10):
                mock.read_measurement()

            mock.reset_stats()
            stats = mock.stats
            assert stats.total_packets == 0

    def test_readings_have_noise(self, config):
        """Mock readings should include noise."""
        with MockImuUart(config) as mock:
            readings = [mock.read_measurement() for _ in range(100)]

            ax_values = [r.ax for r in readings]
            assert max(ax_values) != min(ax_values)


class TestPacketFormat:
    """Tests for packet format constants."""

    def test_packet_size(self):
        """Packet size should be correct."""
        expected_size = struct.calcsize(PACKET_FORMAT)
        assert expected_size == 46

    def test_sync_bytes(self):
        """Sync bytes should be correct."""
        assert SYNC1 == 0xAA
        assert SYNC2 == 0x55

    def test_format_unpacking(self):
        """Packet format should unpack correctly."""
        seq = 12345
        data = [0.1, 0.2, 9.81, 0.01, 0.02, 0.03, 20.0, 5.0, 45.0, 25.0]
        crc = 0x1234

        packet = struct.pack(PACKET_FORMAT, seq, *data, crc)
        unpacked = struct.unpack(PACKET_FORMAT, packet)

        assert unpacked[0] == seq
        assert abs(unpacked[3] - 9.81) < 1e-6
        assert unpacked[-1] == crc
