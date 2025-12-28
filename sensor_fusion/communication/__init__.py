"""Communication module for IMU sensor fusion."""

from .uart import ImuUart, UartError

__all__ = ["ImuUart", "UartError"]
