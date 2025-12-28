"""Extended Kalman Filter with quaternion and gyro bias estimation.

State vector: [qw, qx, qy, qz, bias_x, bias_y, bias_z] (7 dimensions)

The filter estimates both orientation (as quaternion) and gyroscope bias
simultaneously, providing adaptive bias correction that tracks sensor drift.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional


class AdaptiveBiasEKF:
    """EKF with quaternion and gyro bias estimation.

    State: [q(4), bias(3)]
    - Quaternion represents orientation in [w,x,y,z] format
    - Bias represents gyroscope drift (rad/s)

    Process model:
    - Quaternion: kinematic equation with corrected gyro
    - Bias: random walk (slowly varying)

    Measurement model:
    - Accelerometer: gravity direction observation
    - Magnetometer: Earth magnetic field observation (when valid)
    """

    # Reference vectors (NED frame)
    GRAVITY_REF = np.array([0.0, 0.0, 9.81])  # Gravity points down in NED

    def __init__(
        self,
        q0: NDArray[np.float64],
        initial_bias: NDArray[np.float64],
        q_quat: float = 1e-4,
        q_bias: float = 1e-7,
        r_acc: float = 0.1,
        r_mag: float = 1.0,
        mag_ref: Optional[NDArray[np.float64]] = None
    ):
        """Initialize EKF.

        Args:
            q0: Initial quaternion [w,x,y,z], will be normalized.
            initial_bias: Initial bias estimate [bx,by,bz] in rad/s.
            q_quat: Process noise for quaternion states.
            q_bias: Process noise for bias states (random walk).
            r_acc: Accelerometer measurement noise variance.
            r_mag: Magnetometer measurement noise variance.
            mag_ref: Reference magnetic field vector in body frame at t=0.
                     If None, will be set on first valid mag update.
        """
        # Normalize initial quaternion
        q0_norm = q0 / np.linalg.norm(q0)

        # State vector: [qw, qx, qy, qz, bias_x, bias_y, bias_z]
        self.x = np.concatenate([q0_norm, initial_bias.copy()])

        # Covariance matrix (7x7)
        self.P = np.diag([
            0.1, 0.1, 0.1, 0.1,  # Quaternion uncertainty
            0.01, 0.01, 0.01     # Bias uncertainty
        ])

        # Process noise parameters
        self.Q_quat = q_quat
        self.Q_bias = q_bias

        # Measurement noise parameters
        self.R_acc = r_acc
        self.R_mag = r_mag

        # Reference magnetic field (set on first valid reading if None)
        self.mag_ref = mag_ref

        # Statistics
        self.update_count = 0

    def predict(self, gyro_measured: NDArray[np.float64], dt: float) -> None:
        """Prediction step using gyroscope measurement.

        Propagates quaternion using kinematic equation and assumes
        bias follows a random walk (no change in mean).

        Args:
            gyro_measured: Raw gyro reading [gx,gy,gz] in rad/s.
            dt: Time step in seconds.
        """
        if dt <= 0:
            return

        # Extract state components
        q = self.x[0:4].copy()
        bias = self.x[4:7].copy()

        # Correct gyro with current bias estimate
        omega = gyro_measured - bias

        # Quaternion kinematics: q_dot = 0.5 * q ⊗ [0, omega]
        # Using first-order integration
        omega_norm = np.linalg.norm(omega)

        if omega_norm > 1e-10:
            # Rotation quaternion from angular velocity
            theta = omega_norm * dt * 0.5
            axis = omega / omega_norm
            dq = np.array([
                np.cos(theta),
                axis[0] * np.sin(theta),
                axis[1] * np.sin(theta),
                axis[2] * np.sin(theta)
            ])

            # Quaternion multiplication: q_new = q ⊗ dq
            q_new = self._quat_multiply(q, dq)
        else:
            q_new = q.copy()

        # Normalize quaternion
        q_new = q_new / np.linalg.norm(q_new)

        # Update state (bias unchanged in prediction)
        self.x[0:4] = q_new
        # self.x[4:7] unchanged (random walk assumption)

        # Compute Jacobian of state transition
        F = self._compute_transition_jacobian(q, omega, dt)

        # Process noise matrix
        Q = self._compute_process_noise(dt)

        # Propagate covariance: P = F @ P @ F.T + Q
        self.P = F @ self.P @ F.T + Q

        # Ensure symmetry and positive definiteness
        self.P = self._ensure_spd(self.P)

    def update(
        self,
        acc: NDArray[np.float64],
        mag: NDArray[np.float64],
        use_mag: bool
    ) -> None:
        """Measurement update step.

        Args:
            acc: Accelerometer reading [ax,ay,az] in m/s².
            mag: Magnetometer reading [mx,my,mz] in µT.
            use_mag: Whether magnetometer reading is valid.
        """
        # Always update with accelerometer (gravity observation)
        self._update_accelerometer(acc)

        # Conditionally update with magnetometer
        if use_mag:
            if self.mag_ref is None:
                # Set reference magnetic field on first valid reading
                self._set_magnetic_reference(mag)
            else:
                self._update_magnetometer(mag)

        # Normalize quaternion after update
        q_norm = np.linalg.norm(self.x[0:4])
        if q_norm > 1e-10:
            self.x[0:4] = self.x[0:4] / q_norm

        # Ensure positive definiteness
        self.P = self._ensure_spd(self.P)

        self.update_count += 1

    def _update_accelerometer(self, acc: NDArray[np.float64]) -> None:
        """Update state using accelerometer measurement.

        The accelerometer measures gravity in the body frame.
        We compare this to the expected gravity direction based on
        current orientation estimate.
        """
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-6:
            return

        # Normalize accelerometer (we only care about direction)
        acc_unit = acc / acc_norm

        # Current quaternion estimate
        q = self.x[0:4]

        # Expected gravity direction in body frame
        # g_body = q* ⊗ g_world ⊗ q
        g_expected = self._rotate_vector_by_quat_inverse(self.GRAVITY_REF, q)
        g_expected = g_expected / np.linalg.norm(g_expected)

        # Measurement residual (innovation)
        z = acc_unit
        h = g_expected
        y = z - h

        # Measurement Jacobian (3x7)
        H = self._compute_acc_jacobian(q)

        # Measurement noise (scaled by deviation from 1g)
        gravity_deviation = abs(acc_norm - 9.81) / 9.81
        R_scaled = self.R_acc * (1.0 + 10.0 * gravity_deviation)
        R = np.eye(3) * R_scaled

        # Kalman gain: K = P @ H.T @ (H @ P @ H.T + R)^-1
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.pinv(S)

        # State update: x = x + K @ y
        dx = K @ y
        self._apply_state_update(dx)

        # Covariance update: P = (I - K @ H) @ P
        I = np.eye(7)
        self.P = (I - K @ H) @ self.P

    def _update_magnetometer(self, mag: NDArray[np.float64]) -> None:
        """Update state using magnetometer measurement.

        The magnetometer measures Earth's magnetic field in body frame.
        We compare to reference field rotated by current orientation.
        """
        mag_norm = np.linalg.norm(mag)
        if mag_norm < 1e-6 or self.mag_ref is None:
            return

        # Normalize magnetometer
        mag_unit = mag / mag_norm

        # Current quaternion estimate
        q = self.x[0:4]

        # Expected magnetic field direction in body frame
        # Project to horizontal plane for yaw observation
        mag_expected = self._rotate_vector_by_quat_inverse(self.mag_ref, q)
        mag_expected_norm = np.linalg.norm(mag_expected)
        if mag_expected_norm < 1e-6:
            return
        mag_expected = mag_expected / mag_expected_norm

        # Measurement residual
        z = mag_unit
        h = mag_expected
        y = z - h

        # Measurement Jacobian (3x7)
        H = self._compute_mag_jacobian(q)

        # Measurement noise
        R = np.eye(3) * self.R_mag

        # Kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.pinv(S)

        # State update
        dx = K @ y
        self._apply_state_update(dx)

        # Covariance update
        I = np.eye(7)
        self.P = (I - K @ H) @ self.P

    def _set_magnetic_reference(self, mag: NDArray[np.float64]) -> None:
        """Set the reference magnetic field vector in world frame.

        Transforms the body-frame measurement to world frame using
        current orientation estimate, then projects to horizontal.
        """
        mag_norm = np.linalg.norm(mag)
        if mag_norm < 1e-6:
            return

        q = self.x[0:4]

        # Transform to world frame
        mag_world = self._rotate_vector_by_quat(mag, q)

        # Project to horizontal plane (NED: remove Z component)
        mag_horizontal = np.array([mag_world[0], mag_world[1], 0.0])
        mag_h_norm = np.linalg.norm(mag_horizontal)

        if mag_h_norm > 1e-6:
            # Normalize to unit vector pointing North
            self.mag_ref = mag_horizontal / mag_h_norm * mag_norm
        else:
            # Fallback: use full vector
            self.mag_ref = mag_world

    def _apply_state_update(self, dx: NDArray[np.float64]) -> None:
        """Apply state update with quaternion correction.

        For quaternion states, we apply a small rotation rather than
        direct addition to maintain unit norm constraint.
        """
        # Quaternion correction (first 4 elements of dx)
        dq = dx[0:4]

        # Apply as small rotation: q_new = q + 0.5 * q ⊗ [0, dq_vec]
        # For small corrections, we can approximate
        q = self.x[0:4]
        q_new = q + dq
        q_new = q_new / np.linalg.norm(q_new)

        self.x[0:4] = q_new

        # Bias correction (direct addition)
        self.x[4:7] = self.x[4:7] + dx[4:7]

    def _compute_transition_jacobian(
        self,
        q: NDArray[np.float64],
        omega: NDArray[np.float64],
        dt: float
    ) -> NDArray[np.float64]:
        """Compute Jacobian of state transition.

        Returns 7x7 matrix F = ∂f/∂x where f is the state transition function.
        """
        F = np.eye(7)

        # Quaternion-quaternion block (4x4)
        # ∂q_new/∂q ≈ I + 0.5*dt*Omega(omega)
        wx, wy, wz = omega
        Omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        F[0:4, 0:4] = np.eye(4) + 0.5 * dt * Omega

        # Quaternion-bias block (4x3)
        # ∂q_new/∂bias = -0.5*dt*Q(q) where Q maps bias to quaternion derivative
        qw, qx, qy, qz = q
        Q_bias = np.array([
            [-qx, -qy, -qz],
            [qw, -qz, qy],
            [qz, qw, -qx],
            [-qy, qx, qw]
        ])
        F[0:4, 4:7] = -0.5 * dt * Q_bias

        # Bias-bias block: identity (random walk)
        # Already set by np.eye(7)

        return F

    def _compute_process_noise(self, dt: float) -> NDArray[np.float64]:
        """Compute process noise covariance matrix."""
        Q = np.zeros((7, 7))

        # Quaternion process noise (scaled by dt)
        Q[0:4, 0:4] = np.eye(4) * self.Q_quat * dt

        # Bias random walk noise (scaled by dt)
        Q[4:7, 4:7] = np.eye(3) * self.Q_bias * dt

        return Q

    def _compute_acc_jacobian(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute Jacobian of accelerometer measurement model.

        H = ∂h/∂x where h(x) is expected gravity direction in body frame.

        For gravity g_world = [0, 0, 1] (normalized), the expected gravity
        in body frame is g_body = R(q)^T @ g_world = third column of R^T.

        g_body = [2(xz + wy), 2(yz - wx), 1 - 2(x² + y²)]

        Returns 3x7 matrix.
        """
        qw, qx, qy, qz = q
        H = np.zeros((3, 7))

        # ∂g_body/∂q for g_world = [0, 0, 1]
        # g[0] = 2(xz + wy)
        H[0, 0] = 2 * qy   # ∂g[0]/∂w
        H[0, 1] = 2 * qz   # ∂g[0]/∂x
        H[0, 2] = 2 * qw   # ∂g[0]/∂y
        H[0, 3] = 2 * qx   # ∂g[0]/∂z

        # g[1] = 2(yz - wx)
        H[1, 0] = -2 * qx  # ∂g[1]/∂w
        H[1, 1] = -2 * qw  # ∂g[1]/∂x
        H[1, 2] = 2 * qz   # ∂g[1]/∂y
        H[1, 3] = 2 * qy   # ∂g[1]/∂z

        # g[2] = 1 - 2(x² + y²)
        H[2, 0] = 0        # ∂g[2]/∂w
        H[2, 1] = -4 * qx  # ∂g[2]/∂x
        H[2, 2] = -4 * qy  # ∂g[2]/∂y
        H[2, 3] = 0        # ∂g[2]/∂z

        # Bias has no effect on accelerometer measurement
        # H[:, 4:7] remains zero

        return H

    def _compute_mag_jacobian(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute Jacobian of magnetometer measurement model.

        Returns 3x7 matrix.
        """
        # Similar structure to accelerometer Jacobian
        # but with magnetic reference instead of gravity
        H = np.zeros((3, 7))

        if self.mag_ref is None:
            return H

        qw, qx, qy, qz = q
        mx, my, mz = self.mag_ref

        # ∂(R.T @ m_ref)/∂q
        H[0, 0] = 2*(qw*mx - qz*my + qy*mz)
        H[0, 1] = 2*(qx*mx + qy*my + qz*mz)
        H[0, 2] = 2*(-qy*mx + qx*my + qw*mz)
        H[0, 3] = 2*(-qz*mx - qw*my + qx*mz)

        H[1, 0] = 2*(qz*mx + qw*my - qx*mz)
        H[1, 1] = 2*(qy*mx - qx*my - qw*mz)
        H[1, 2] = 2*(qx*mx + qy*my + qz*mz)
        H[1, 3] = 2*(qw*mx - qz*my + qy*mz)

        H[2, 0] = 2*(-qy*mx + qx*my + qw*mz)
        H[2, 1] = 2*(qz*mx + qw*my - qx*mz)
        H[2, 2] = 2*(-qw*mx + qz*my - qy*mz)
        H[2, 3] = 2*(qx*mx + qy*my + qz*mz)

        return H

    def _quat_multiply(
        self,
        q1: NDArray[np.float64],
        q2: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Quaternion multiplication (Hamilton product)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def _rotate_vector_by_quat(
        self,
        v: NDArray[np.float64],
        q: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Rotate vector by quaternion: v' = q ⊗ v ⊗ q*"""
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        v_quat = np.array([0, v[0], v[1], v[2]])

        temp = self._quat_multiply(q, v_quat)
        result = self._quat_multiply(temp, q_conj)

        return result[1:4]

    def _rotate_vector_by_quat_inverse(
        self,
        v: NDArray[np.float64],
        q: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Rotate vector by inverse quaternion: v' = q* ⊗ v ⊗ q"""
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        v_quat = np.array([0, v[0], v[1], v[2]])

        temp = self._quat_multiply(q_conj, v_quat)
        result = self._quat_multiply(temp, q)

        return result[1:4]

    def _ensure_spd(self, P: NDArray[np.float64]) -> NDArray[np.float64]:
        """Ensure matrix is symmetric positive definite."""
        # Symmetrize
        P = 0.5 * (P + P.T)

        # Ensure positive eigenvalues
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals = np.maximum(eigvals, 1e-10)
        P = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return P

    def get_quaternion(self) -> NDArray[np.float64]:
        """Get current quaternion estimate [w,x,y,z]."""
        q = self.x[0:4].copy()
        return q / np.linalg.norm(q)

    def get_bias(self) -> NDArray[np.float64]:
        """Get current gyro bias estimate [bx,by,bz] in rad/s."""
        return self.x[4:7].copy()

    def get_covariance(self) -> NDArray[np.float64]:
        """Get current state covariance matrix."""
        return self.P.copy()

    def get_quaternion_uncertainty(self) -> float:
        """Get quaternion uncertainty (trace of quaternion covariance)."""
        return float(np.trace(self.P[0:4, 0:4]))

    def get_bias_uncertainty(self) -> NDArray[np.float64]:
        """Get bias uncertainty (diagonal of bias covariance)."""
        return np.sqrt(np.diag(self.P[4:7, 4:7]))
