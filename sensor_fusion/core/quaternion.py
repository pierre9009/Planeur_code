"""Quaternion operations and utilities."""

import numpy as np
from numpy.typing import NDArray

from .types import Quaternion, EulerAngles


class QuaternionOps:
    """Static methods for quaternion operations."""

    @staticmethod
    def from_rotation_matrix(R: NDArray[np.float64]) -> Quaternion:
        """Convert rotation matrix to quaternion.

        Uses Shepperd's method for numerical stability.

        Args:
            R: 3x3 rotation matrix.

        Returns:
            Unit quaternion representing the same rotation.
        """
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        q = Quaternion(w=w, x=x, y=y, z=z)
        return q.normalized()

    @staticmethod
    def to_euler(q: Quaternion) -> EulerAngles:
        """Convert quaternion to Euler angles (ZYX convention).

        Args:
            q: Unit quaternion.

        Returns:
            Euler angles in radians.
        """
        w, x, y, z = q.w, q.x, q.y, q.z

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return EulerAngles(roll=float(roll), pitch=float(pitch), yaw=float(yaw))

    @staticmethod
    def from_acc_mag(
        acc: NDArray[np.float64],
        mag: NDArray[np.float64],
        frame: str = "NED"
    ) -> Quaternion:
        """Compute initial quaternion from accelerometer and magnetometer.

        This method constructs an orthonormal frame from gravity and
        magnetic field vectors, suitable for EKF initialization.

        Args:
            acc: Accelerometer reading [ax, ay, az] in m/s^2.
            mag: Magnetometer reading [mx, my, mz] in uT.
            frame: Reference frame, 'NED' or 'ENU'.

        Returns:
            Initial orientation quaternion.
        """
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-6:
            return Quaternion.identity()
        acc_unit = acc / acc_norm

        mag_norm = np.linalg.norm(mag)
        if mag_norm < 1e-6:
            return Quaternion.identity()
        mag_unit = mag / mag_norm

        if frame == "NED":
            down = np.array([0.0, 0.0, 1.0])
        else:
            down = np.array([0.0, 0.0, -1.0])

        mag_horizontal = mag_unit - np.dot(mag_unit, down) * down
        mag_h_norm = np.linalg.norm(mag_horizontal)
        if mag_h_norm < 1e-6:
            return Quaternion.identity()
        mag_horizontal = mag_horizontal / mag_h_norm

        if frame == "NED":
            z_axis = down
            x_axis = mag_horizontal
            y_axis = np.cross(z_axis, x_axis)
        else:
            z_axis = -down
            y_axis = mag_horizontal
            x_axis = np.cross(y_axis, z_axis)

        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        R = np.column_stack([x_axis, y_axis, z_axis])
        return QuaternionOps.from_rotation_matrix(R)

    @staticmethod
    def multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
        """Multiply two quaternions (Hamilton product).

        Args:
            q1: First quaternion.
            q2: Second quaternion.

        Returns:
            Product quaternion q1 * q2.
        """
        w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
        x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y
        y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x
        z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
        return Quaternion(w=w, x=x, y=y, z=z)

    @staticmethod
    def conjugate(q: Quaternion) -> Quaternion:
        """Compute quaternion conjugate.

        Args:
            q: Input quaternion.

        Returns:
            Conjugate quaternion.
        """
        return Quaternion(w=q.w, x=-q.x, y=-q.y, z=-q.z)

    @staticmethod
    def angle_between(q1: Quaternion, q2: Quaternion) -> float:
        """Compute rotation angle between two quaternions.

        Args:
            q1: First quaternion.
            q2: Second quaternion.

        Returns:
            Angle in radians.
        """
        q1_conj = QuaternionOps.conjugate(q1)
        q_diff = QuaternionOps.multiply(q2, q1_conj)
        angle = 2.0 * np.arccos(np.clip(abs(q_diff.w), -1.0, 1.0))
        return float(angle)

    @staticmethod
    def slerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
        """Spherical linear interpolation between quaternions.

        Args:
            q1: Start quaternion.
            q2: End quaternion.
            t: Interpolation parameter [0, 1].

        Returns:
            Interpolated quaternion.
        """
        dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z

        if dot < 0:
            q2 = Quaternion(w=-q2.w, x=-q2.x, y=-q2.y, z=-q2.z)
            dot = -dot

        if dot > 0.9995:
            w = q1.w + t * (q2.w - q1.w)
            x = q1.x + t * (q2.x - q1.x)
            y = q1.y + t * (q2.y - q1.y)
            z = q1.z + t * (q2.z - q1.z)
            return Quaternion(w=w, x=x, y=y, z=z).normalized()

        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)

        s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0

        w = s1 * q1.w + s2 * q2.w
        x = s1 * q1.x + s2 * q2.x
        y = s1 * q1.y + s2 * q2.y
        z = s1 * q1.z + s2 * q2.z

        return Quaternion(w=w, x=x, y=y, z=z)
