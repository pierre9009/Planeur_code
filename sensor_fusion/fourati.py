import numpy as np


class AttitudeEstimator:
    def __init__(self, k_q=100.0, k_b=200.0, alpha=0.33):
        # Paramètres de l'observateur [cite: 318, 319]

        self.k_q = k_q  # Gain pour le quaternion
        self.k_b = k_b  # Gain pour le biais du gyroscope
        self.alpha = alpha  # Vitesse de convergence ILSA [cite: 225]
        
        # États initialisés [cite: 351]
        self.q = np.array([[1.0], [0.0], [0.0], [0.0]])  # Quaternion estimé (w, x, y, z)
        self.bias = np.zeros((3,1))  # Biais du gyroscope estimé

        self.tau = 80 # seconde
        
        # Vecteurs de référence (Navigation frame F_I) [cite: 121, 160]
        self.g_ref = np.array([0.0, 0.0, 1.0])  # Gravité normalisée en quaternion
        self.m_ref = np.array([0.5, 0.0, np.sqrt(3)/2])  # Champ magnétique normalisé (dip angle 60°) en quaternion

    def _skew(self, v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def _quaternion_prod(self, p, q):
        """Produit de deux quaternions"""
        w1 = p[0,0]; x1 = p[1,0]; y1 = p[2,0]; z1 = p[3,0]
        w2 = q[0,0]; x2 = q[1,0]; y2 = q[2,0]; z2 = q[3,0]

        A = np.array([[w1,-x1,-y1, -z1],
                  [x1, w1, -z1, y1],
                  [y1, z1, w1, -x1],
                  [z1, -y1, x1, w1]])

        B = np.array([[w2], [x2], [y2], [z2]])
        product_result = A@B
        return product_result
    

    def _ilsa(self, acc, mag, num_iter=8):
        """Iterated Least Squares Algorithm pour obtenir q_m [cite: 215-227]."""
        q_est = self.q.copy()
        # Normalisation des mesures
        f = acc / np.linalg.norm(acc)
        h = mag / np.linalg.norm(mag)

        f = np.array([[0],
                    [f[0]],
                    [f[1]],
                    [f[2]]])
        h = np.array([[0],
                      [h[0]],
                      [h[1]],
                      [h[2]]])
        
        for _ in range(num_iter):
            #Estimated gvect

            q_est_inv = np.array([[q_est[0,0]], [-q_est[1,0]], [-q_est[2,0]], [-q_est[3,0]]])
            tempo = self._quaternion_prod(q_est,f)
            g_vect = self._quaternion_prod(tempo,q_est_inv)

            #Estimated hvect

            tempo = self._quaternion_prod(q_est,h)
            h_vect = self._quaternion_prod(tempo,q_est_inv)

            #Navigation error
            delta_g_vect = np.array([[self.g_ref[0]-g_vect[1,0]],[self.g_ref[1]-g_vect[2,0]],[self.g_ref[2]-g_vect[3,0]]])
            delta_h_vect = np.array([[self.m_ref[0]-h_vect[1,0]],[self.m_ref[1]-h_vect[2,0]],[self.m_ref[2]-h_vect[3,0]]])
            z = np.vstack((delta_h_vect, delta_g_vect))

            #Compute the observation matrix
            step1 = -2 * self._skew(self.m_ref)
            step2 = -2 * self._skew(self.g_ref)
            O = np.hstack((step1.T, step2.T)).T

            #calculate pseudo inverse

            O_star = np.linalg.inv(O.T@O)@O.T

            #Compute qe
            qe = self.alpha*O_star@z
            qe = np.vstack((np.array([[1.0]]), qe))
            qe /= np.linalg.norm(qe)

            q_est = self._quaternion_prod(qe,q_est)
            q_est = q_est/np.linalg.norm(q_est)
        return q_est 

    def update(self, gyro, acc, mag, dt):
        """Mise à jour de l'estimation à chaque pas de temps (Online)."""
        gyro = gyro.reshape((3,1))

        q_m = self._ilsa(acc, mag)
        q_inv = np.array([[self.q[0,0]], [-self.q[1,0]], [-self.q[2,0]], [-self.q[3,0]]])
        q_er = self._quaternion_prod(q_inv, q_m)
        q_vect_er = np.array([
                                [q_er[1,0]],
                                [q_er[2,0]],
                                [q_er[3,0]],
                            ])
        
        # dynamique quaternion eq 12
        q0 = self.q[0,0]
        q1 = self.q[1,0]
        q2 = self.q[2,0]
        q3 = self.q[3,0]
        quater_tempo = (0.5)*np.array([[-q1, -q2, -q3],
                                       [q0, -q3, q2],
                                       [q3, q0 , -q1],
                                       [-q2, q1, q0]])
        dq = quater_tempo@(gyro - self.bias + self.k_q*q_vect_er)
        
        #Dynamique du biais

        Ninv = (1/self.tau)*np.eye(3,3)

        dbias = -Ninv@self.bias - self.k_b * q_vect_er

        #euler approx
        self.q = dq * dt + self.q
        self.q /= np.linalg.norm(self.q)

        self.bias = dbias * dt + self.bias

        return self.q
    
