"""Tests for quaternion operations."""

import pytest
import numpy as np

from core.types import Quaternion, EulerAngles
from core.quaternion import QuaternionOps


class TestQuaternion:
    """Tests for Quaternion dataclass."""

    def test_identity(self):
        """Identity quaternion should have correct values."""
        q = Quaternion.identity()
        assert q.w == 1.0
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0

    def test_from_array(self):
        """Quaternion should be created from numpy array."""
        arr = np.array([0.7071, 0.0, 0.7071, 0.0])
        q = Quaternion.from_array(arr)

        assert abs(q.w - 0.7071) < 1e-4
        assert abs(q.x - 0.0) < 1e-4
        assert abs(q.y - 0.7071) < 1e-4
        assert abs(q.z - 0.0) < 1e-4

    def test_to_array(self):
        """Quaternion should convert to numpy array."""
        q = Quaternion(w=0.7071, x=0.0, y=0.7071, z=0.0)
        arr = q.to_array()

        assert isinstance(arr, np.ndarray)
        assert len(arr) == 4
        assert abs(arr[0] - 0.7071) < 1e-4
        assert abs(arr[2] - 0.7071) < 1e-4

    def test_norm_identity(self):
        """Identity quaternion should have unit norm."""
        q = Quaternion.identity()
        assert abs(q.norm - 1.0) < 1e-10

    def test_norm_non_unit(self):
        """Non-unit quaternion should have correct norm."""
        q = Quaternion(w=2.0, x=0.0, y=0.0, z=0.0)
        assert abs(q.norm - 2.0) < 1e-10

    def test_is_valid_unit(self):
        """Unit quaternion should be valid."""
        q = Quaternion.identity()
        assert q.is_valid()

    def test_is_valid_non_unit(self):
        """Non-unit quaternion should be invalid."""
        q = Quaternion(w=2.0, x=0.0, y=0.0, z=0.0)
        assert not q.is_valid()

    def test_is_valid_with_tolerance(self):
        """Near-unit quaternion should be valid within tolerance."""
        q = Quaternion(w=0.995, x=0.0, y=0.0, z=0.0)
        assert q.is_valid(tolerance=0.01)
        assert not q.is_valid(tolerance=0.001)

    def test_normalized(self):
        """Normalized quaternion should have unit norm."""
        q = Quaternion(w=2.0, x=2.0, y=2.0, z=2.0)
        q_norm = q.normalized()

        assert abs(q_norm.norm - 1.0) < 1e-10

    def test_normalized_zero_quaternion(self):
        """Zero quaternion should normalize to identity."""
        q = Quaternion(w=0.0, x=0.0, y=0.0, z=0.0)
        q_norm = q.normalized()

        assert q_norm.w == 1.0


class TestQuaternionOps:
    """Tests for QuaternionOps static methods."""

    def test_to_euler_identity(self):
        """Identity quaternion should give zero Euler angles."""
        q = Quaternion.identity()
        euler = QuaternionOps.to_euler(q)

        assert abs(euler.roll) < 1e-10
        assert abs(euler.pitch) < 1e-10
        assert abs(euler.yaw) < 1e-10

    def test_to_euler_roll_90(self):
        """90 degree roll should give correct Euler angles."""
        angle = np.deg2rad(90)
        q = Quaternion(
            w=np.cos(angle / 2),
            x=np.sin(angle / 2),
            y=0.0,
            z=0.0,
        )
        euler = QuaternionOps.to_euler(q)

        assert abs(euler.roll_deg - 90) < 1.0
        assert abs(euler.pitch) < 0.1
        assert abs(euler.yaw) < 0.1

    def test_to_euler_yaw_45(self):
        """45 degree yaw should give correct Euler angles."""
        angle = np.deg2rad(45)
        q = Quaternion(
            w=np.cos(angle / 2),
            x=0.0,
            y=0.0,
            z=np.sin(angle / 2),
        )
        euler = QuaternionOps.to_euler(q)

        assert abs(euler.roll) < 0.1
        assert abs(euler.pitch) < 0.1
        assert abs(euler.yaw_deg - 45) < 1.0

    def test_from_rotation_matrix_identity(self):
        """Identity matrix should give identity quaternion."""
        R = np.eye(3)
        q = QuaternionOps.from_rotation_matrix(R)

        assert abs(q.w - 1.0) < 1e-6
        assert abs(q.x) < 1e-6
        assert abs(q.y) < 1e-6
        assert abs(q.z) < 1e-6

    def test_from_rotation_matrix_90_z(self):
        """90 degree Z rotation matrix should give correct quaternion."""
        R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        q = QuaternionOps.from_rotation_matrix(R)
        euler = QuaternionOps.to_euler(q)

        assert abs(euler.yaw_deg - 90) < 1.0

    def test_from_acc_mag_ned_level(self):
        """Level sensor in NED frame should give identity-like quaternion."""
        acc = np.array([0.0, 0.0, 9.81])
        mag = np.array([20.0, 0.0, 45.0])

        q = QuaternionOps.from_acc_mag(acc, mag, frame="NED")
        euler = QuaternionOps.to_euler(q)

        assert abs(euler.roll_deg) < 5.0
        assert abs(euler.pitch_deg) < 5.0

    def test_multiply_identity(self):
        """Multiplying by identity should not change quaternion."""
        q = Quaternion(w=0.7071, x=0.0, y=0.7071, z=0.0)
        identity = Quaternion.identity()

        result = QuaternionOps.multiply(q, identity)

        assert abs(result.w - q.w) < 1e-6
        assert abs(result.x - q.x) < 1e-6
        assert abs(result.y - q.y) < 1e-6
        assert abs(result.z - q.z) < 1e-6

    def test_multiply_inverse(self):
        """Multiplying quaternion by conjugate should give identity."""
        q = Quaternion(w=0.7071067811865476, x=0.0, y=0.7071067811865476, z=0.0)
        q_conj = QuaternionOps.conjugate(q)
        result = QuaternionOps.multiply(q, q_conj)

        assert abs(result.w - 1.0) < 1e-6
        assert abs(result.x) < 1e-6
        assert abs(result.y) < 1e-6
        assert abs(result.z) < 1e-6

    def test_conjugate(self):
        """Conjugate should negate imaginary parts."""
        q = Quaternion(w=1.0, x=2.0, y=3.0, z=4.0)
        q_conj = QuaternionOps.conjugate(q)

        assert q_conj.w == q.w
        assert q_conj.x == -q.x
        assert q_conj.y == -q.y
        assert q_conj.z == -q.z

    def test_angle_between_same(self):
        """Angle between identical quaternions should be zero."""
        q = Quaternion.identity()
        angle = QuaternionOps.angle_between(q, q)

        assert abs(angle) < 1e-6

    def test_angle_between_90_deg(self):
        """Angle between quaternions should be computed correctly."""
        q1 = Quaternion.identity()
        angle_rad = np.deg2rad(90)
        q2 = Quaternion(
            w=np.cos(angle_rad / 2),
            x=0.0,
            y=0.0,
            z=np.sin(angle_rad / 2),
        )

        angle = QuaternionOps.angle_between(q1, q2)
        assert abs(np.rad2deg(angle) - 90) < 1.0

    def test_slerp_endpoints(self):
        """Slerp at t=0 and t=1 should return endpoints."""
        q1 = Quaternion.identity()
        q2 = Quaternion(w=0.7071, x=0.0, y=0.7071, z=0.0)

        result0 = QuaternionOps.slerp(q1, q2, 0.0)
        assert abs(result0.w - q1.w) < 1e-6
        assert abs(result0.y - q1.y) < 1e-6

        result1 = QuaternionOps.slerp(q1, q2, 1.0)
        assert abs(result1.w - q2.w) < 1e-6
        assert abs(result1.y - q2.y) < 1e-6

    def test_slerp_midpoint(self):
        """Slerp at t=0.5 should return midpoint rotation."""
        q1 = Quaternion.identity()
        angle = np.deg2rad(90)
        q2 = Quaternion(
            w=np.cos(angle / 2),
            x=0.0,
            y=0.0,
            z=np.sin(angle / 2),
        )

        result = QuaternionOps.slerp(q1, q2, 0.5)
        euler = QuaternionOps.to_euler(result)

        assert abs(euler.yaw_deg - 45) < 2.0


class TestEulerAngles:
    """Tests for EulerAngles dataclass."""

    def test_deg_conversion(self):
        """Degree properties should convert correctly."""
        euler = EulerAngles(
            roll=np.deg2rad(30),
            pitch=np.deg2rad(45),
            yaw=np.deg2rad(60),
        )

        assert abs(euler.roll_deg - 30) < 1e-6
        assert abs(euler.pitch_deg - 45) < 1e-6
        assert abs(euler.yaw_deg - 60) < 1e-6
