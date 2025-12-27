import numpy as np

def quat_mul(q, r):
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def quat_norm(q):
    return q / np.linalg.norm(q)

def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]
    ])

class AttitudeEKF:
    def __init__(self):
        self.x = np.zeros(7)
        self.x[0] = 1.0
        self.P = np.eye(7) * 0.01

        self.Q = np.eye(7) * 1e-5
        self.Q[4:7,4:7] *= 1e-3

        self.R_acc = np.eye(3) * 0.05
        self.R_mag = np.eye(3) * 0.1

    def predict(self, gyro, dt):
        q = self.x[:4]
        b = self.x[4:]

        omega = gyro - b
        dq = np.hstack(([1.0], 0.5 * omega * dt))
        q = quat_mul(q, dq)
        self.x[:4] = quat_norm(q)

        F = np.eye(7)
        F[:4,4:] = -0.5 * dt

        self.P = F @ self.P @ F.T + self.Q

    def update_acc(self, acc):
        acc = acc / np.linalg.norm(acc)
        q = self.x[:4]
        R = quat_to_rot(q)

        g_pred = R.T @ np.array([0, 0, 1])
        y = acc - g_pred

        H = np.zeros((3,7))
        H[:,:4] = 0.5

        S = H @ self.P @ H.T + self.R_acc
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        self.P = (np.eye(7) - K @ H) @ self.P
        self.x[:4] = quat_norm(self.x[:4])

    def update_mag(self, mag):
        mag = mag / np.linalg.norm(mag)
        q = self.x[:4]
        R = quat_to_rot(q)

        m_ref = np.array([1, 0, 0])
        m_pred = R.T @ m_ref
        y = mag - m_pred

        H = np.zeros((3,7))
        H[:,:4] = 0.5

        S = H @ self.P @ H.T + self.R_mag
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        self.P = (np.eye(7) - K @ H) @ self.P
        self.x[:4] = quat_norm(self.x[:4])

    def update(self, gyro, acc, mag, dt):
        self.predict(gyro, dt)
        self.update_acc(acc)
        self.update_mag(mag)
        return self.x[:4].copy(), self.x[4:].copy()
