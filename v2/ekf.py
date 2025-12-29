"""
Extended Kalman Filter (EKF) for quaternion-based orientation estimation.
Python/NumPy implementation based on Rust version.
"""

import numpy as np
from scipy.spatial.transform import Rotation

GRAVITY = 9.81  # Gravitational constant (m/s^2)


class EKF:
    """
    Extended Kalman Filter for orientation estimation using quaternions.
    State vector: [q0, q1, q2, q3, bx, by, bz] (quaternion + gyro bias)
    """

    def __init__(self, accel_data: np.ndarray = None):
        """
        Create a new EKF instance.
        
        Args:
            accel_data: Optional initial accelerometer data [ax, ay, az] to compute initial orientation.
        """
        if accel_data is not None:
            # Normalize accelerometer vector
            norm = np.linalg.norm(accel_data)
            ax = accel_data[0] / norm
            ay = accel_data[1] / norm
            az = -accel_data[2] / norm

            # Calculate quaternion from accelerometer data
            q0 = np.sqrt(1.0 + az) / 2.0
            q1 = -ay / (2.0 * q0)
            q2 = ax / (2.0 * q0)
            q3 = 0.0  # Yaw is zero since accelerometer cannot calculate yaw

            # Normalize quaternion
            q_norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
            q0, q1, q2, q3 = q0/q_norm, q1/q_norm, q2/q_norm, q3/q_norm
        else:
            # Default to identity quaternion
            q0, q1, q2, q3 = 1.0, 0.0, 0.0, 0.0

        # Initialize state vector [q0, q1, q2, q3, bx, by, bz]
        self.state = np.array([q0, q1, q2, q3, 0.0, 0.0, 0.0])

        # Initial state covariance (7x7)
        self.covariance = np.eye(7) * 1.0

        # Process noise Q (7x7)
        self.process_noise = np.diag([0.05, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01])

        # Measurement noise R (3x3)
        self.measurement_noise = np.diag([0.1, 0.1, 0.1])

    def predict(self, gyro: np.ndarray, dt: float):
        """
        EKF Predict Step: Propagates state and covariance forward using gyro data.
        
        Args:
            gyro: Gyroscope measurements [wx, wy, wz] in rad/s.
            dt: Time step in seconds.
        """
        # 1. Subtract estimated bias from raw gyro measurements
        bias = self.state[4:7]
        omega = gyro - bias

        # 2. Integrate quaternion using angular velocity
        q = self.state[0:4]
        omega_matrix = self._omega_matrix(omega)
        q_dot = 0.5 * omega_matrix @ q
        q_new = q + q_dot * dt

        # 3. Update state with new quaternion
        self.state[0:4] = q_new
        self._normalize_quaternion()

        # 4. Compute Jacobian of motion model
        f_jacobian = self._compute_f_jacobian(gyro, dt)

        # 5. Propagate uncertainty: P' = FPF^T + Q
        self.covariance = f_jacobian @ self.covariance @ f_jacobian.T + self.process_noise

        # 6. Lock yaw axis
        self._lock_yaw()

    def update(self, accel: np.ndarray):
        """
        EKF Update Step: Corrects the prediction using accelerometer data.
        
        Args:
            accel: Accelerometer measurements [ax, ay, az] in m/s^2.
        """
        # 1. Compute expected gravity vector in sensor frame
        gravity = np.array([0.0, 0.0, -GRAVITY])
        q = self.state[0:4]
        r_transpose = self._quaternion_to_rotation_matrix(q).T
        accel_expected = r_transpose @ gravity

        # 2. Compute innovation (measurement residual)
        z = accel
        innovation = z - accel_expected

        # 3. Compute Jacobian of measurement model
        h_jacobian = self._compute_h_jacobian(q)

        # 4. Compute innovation covariance: S = HPH^T + R
        s = h_jacobian @ self.covariance @ h_jacobian.T + self.measurement_noise

        # 5. Compute Kalman gain: K = PH^T * S^(-1)
        try:
            s_inv = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            print("Warning: Skipping EKF update — non-invertible innovation covariance.")
            return

        k = self.covariance @ h_jacobian.T @ s_inv

        # 6. Update state: x = x + K(z - h(x))
        self.state = self.state + k @ innovation

        # 7. Update covariance: P = (I - KH)P
        self.covariance = (np.eye(7) - k @ h_jacobian) @ self.covariance

        # 8. Normalize quaternion after update
        self._normalize_quaternion()

        # 9. Re-lock yaw axis
        self._lock_yaw()

    def _compute_f_jacobian(self, gyro: np.ndarray, dt: float) -> np.ndarray:
        """Compute the dynamic Jacobian (∂f/∂x)."""
        q0, q1, q2, q3 = self.state[0:4]
        bx, by, bz = self.state[4:7]

        p = gyro[0] - bx
        q = gyro[1] - by
        r = gyro[2] - bz

        f = np.eye(7)

        # Quaternion dynamics wrt quaternion
        f[0, 1] = -p * dt
        f[0, 2] = -q * dt
        f[0, 3] = -r * dt

        f[1, 0] = p * dt
        f[1, 2] = r * dt
        f[1, 3] = -q * dt

        f[2, 0] = q * dt
        f[2, 1] = -r * dt
        f[2, 3] = p * dt

        f[3, 0] = r * dt
        f[3, 1] = q * dt
        f[3, 2] = -p * dt

        # Quaternion dynamics wrt bias
        f[0, 4] = 0.5 * q1 * dt
        f[0, 5] = 0.5 * q2 * dt
        f[0, 6] = 0.5 * q3 * dt

        f[1, 4] = -0.5 * q0 * dt
        f[1, 5] = -0.5 * q3 * dt
        f[1, 6] = 0.5 * q2 * dt

        f[2, 4] = 0.5 * q3 * dt
        f[2, 5] = -0.5 * q0 * dt
        f[2, 6] = -0.5 * q1 * dt

        f[3, 4] = -0.5 * q2 * dt
        f[3, 5] = 0.5 * q1 * dt
        f[3, 6] = -0.5 * q0 * dt

        return f

    def _compute_h_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute the measurement Jacobian (∂h/∂x)."""
        q0, q1, q2, q3 = q

        h = np.zeros((3, 7))

        h[0, 0] = 2.0 * (-GRAVITY * q2)
        h[0, 1] = 2.0 * (GRAVITY * q3)
        h[0, 2] = 2.0 * (-GRAVITY * q0)
        h[0, 3] = 2.0 * (GRAVITY * q1)

        h[1, 0] = 2.0 * (GRAVITY * q1)
        h[1, 1] = 2.0 * (GRAVITY * q0)
        h[1, 2] = 2.0 * (GRAVITY * q3)
        h[1, 3] = 2.0 * (GRAVITY * q2)

        h[2, 0] = 2.0 * (GRAVITY * q0)
        h[2, 1] = 2.0 * (-GRAVITY * q1)
        h[2, 2] = 2.0 * (-GRAVITY * q2)
        h[2, 3] = 2.0 * (-GRAVITY * q3)

        return h

    def _normalize_quaternion(self):
        """Normalize the quaternion part of the state."""
        q = self.state[0:4]
        norm = np.linalg.norm(q)
        if norm > 0.0:
            self.state[0:4] = q / norm

    @staticmethod
    def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        q0, q1, q2, q3 = q

        return np.array([
            [1.0 - 2.0*(q2*q2 + q3*q3), 2.0*(q1*q2 - q0*q3), 2.0*(q1*q3 + q0*q2)],
            [2.0*(q1*q2 + q0*q3), 1.0 - 2.0*(q1*q1 + q3*q3), 2.0*(q2*q3 - q0*q1)],
            [2.0*(q1*q3 - q0*q2), 2.0*(q2*q3 + q0*q1), 1.0 - 2.0*(q1*q1 + q2*q2)]
        ])

    @staticmethod
    def _omega_matrix(omega: np.ndarray) -> np.ndarray:
        """Compute omega matrix for quaternion dynamics."""
        return np.array([
            [0.0, -omega[0], -omega[1], -omega[2]],
            [omega[0], 0.0, omega[2], -omega[1]],
            [omega[1], -omega[2], 0.0, omega[0]],
            [omega[2], omega[1], -omega[0], 0.0]
        ])

    def _remove_yaw_from_quaternion(self):
        """Remove yaw component from quaternion."""
        # Convert state quaternion to scipy Rotation
        # Note: scipy uses [x, y, z, w] format, our state uses [w, x, y, z]
        q_scipy = [self.state[1], self.state[2], self.state[3], self.state[0]]
        rot = Rotation.from_quat(q_scipy)

        # Get euler angles (roll, pitch, yaw)
        euler = rot.as_euler('xyz')

        # Create new rotation with yaw = 0
        new_rot = Rotation.from_euler('xyz', [euler[0], euler[1], 0.0])
        new_q = new_rot.as_quat()  # [x, y, z, w]

        # Update state (convert back to [w, x, y, z])
        self.state[0] = new_q[3]  # w
        self.state[1] = new_q[0]  # x
        self.state[2] = new_q[1]  # y
        self.state[3] = new_q[2]  # z

    def _lock_yaw(self):
        """Lock yaw axis to ensure stability with only accelerometer."""
        self.state[6] = 0.0
        self.covariance[6, 6] = 0.0
        self._remove_yaw_from_quaternion()

    def get_state(self) -> np.ndarray:
        """Get the fully updated state vector."""
        return self.state.copy()


# Example usage
if __name__ == "__main__":
    # Initialize with accelerometer data
    initial_accel = np.array([0.1, 0.2, -9.8])
    ekf = EKF(accel_data=initial_accel)

    print("Initial state:", ekf.get_state())

    # Simulate a few steps
    dt = 0.01  # 10ms
    for i in range(10):
        gyro = np.array([0.01, 0.02, 0.0])  # Small angular velocity
        accel = np.array([0.1, 0.2, -9.8])   # Gravity + small offset

        ekf.predict(gyro, dt)
        ekf.update(accel)

    print("Final state:", ekf.get_state())
    print("Quaternion (w,x,y,z):", ekf.state[0:4])
    print("Gyro bias:", ekf.state[4:7])