"""Communication module for IMU sensor fusion."""

from communication.uart import ImuUart, UartError

__all__ = ["ImuUart", "UartError"]
