import numpy as np
import matplotlib.pyplot as plt


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
        self.g_ref = np.array([0.0, 0.0, 1.0])  # Gravité normalisée
        self.m_ref = np.array([0.5, 0.0, np.sqrt(3)/2])  # Champ magnétique normalisé (dip angle 60°)

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
    

    def _ilsa(self, acc, mag, num_iter=5):
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

        return self.q, self.bias
    
def simulate_and_plot():
    dt = 0.01
    T = 50.0
    steps = int(T/dt)
    time = np.linspace(0, T, steps)

    # 1. Génération de la Vérité Terrain (Ground Truth)
    q_true = np.zeros((4, steps))
    q_true[:, 0] = [1.0, 0.0, 0.0, 0.0] # Init théorique 
    
    # Biais réel du gyroscope (simulé comme un processus de Gauss-Markov)
    bias_true = np.zeros((3, steps))
    bias_true[:, 0] = [-2.0, 1.0, 0.5] # Biais initaux 

    # Profil de vitesse angulaire du papier (Eq. 14) [cite: 327, 329]
    def get_omega(t):
        if t <= 25:
            return np.array([-0.8*np.sin(1.2*t), 1.1*np.cos(0.5*t), 0.4*np.sin(0.3*t)])
        else:
            return np.array([1.3*np.sin(1.4*t), -0.6*np.cos(-0.3*t), 0.3*np.sin(0.5*t)])

    # Intégration de la trajectoire réelle
    for i in range(1, steps):
        t = time[i-1]
        w = get_omega(t)
        q = q_true[:, i-1]
        mat = 0.5 * np.array([[-q[1], -q[2], -q[3]],
                              [q[0], -q[3], q[2]],
                              [q[3], q[0], -q[1]],
                              [-q[2], q[1], q[0]]])
        q_true[:, i] = q + (mat @ w) * dt
        q_true[:, i] /= np.linalg.norm(q_true[:, i])
        # Évolution lente du biais
        bias_true[:, i] = bias_true[:, i-1] * (1 - dt/80.0)

    # 2. Simulation des Capteurs (avec bruit) 
    gyro_meas = np.zeros((3, steps))
    acc_meas = np.zeros((3, steps))
    mag_meas = np.zeros((3, steps))
    
    g_nav = np.array([0, 0, 1])
    m_nav = np.array([0.5, 0, np.sqrt(3)/2])

    for i in range(steps):
        q = q_true[:, i]
        # Matrice de rotation (Navigation vers Corps)
        q0, q1, q2, q3 = q
        R = np.array([
            [2*(q0**2+q1**2)-1, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)],
            [2*(q1*q2-q0*q3), 2*(q0**2+q2**2)-1, 2*(q0*q1+q2*q3)],
            [2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), 2*(q0**2+q3**2)-1]
        ])

        gyro_meas[:, i] = get_omega(time[i]) + bias_true[:, i] + np.random.normal(0, 0.01, 3)
        acc_meas[:, i] = R @ g_nav + np.random.normal(0, 0.002, 3)
        mag_meas[:, i] = R @ m_nav + np.random.normal(0, 0.007, 3)
        print(f"\rFourati -> gyro={gyro_meas[:, i]}, acc={acc_meas[:, i]}, mag={mag_meas[:, i]}")

    # 3. Exécution de l'Estimateur
    # Initialisation de l'observateur avec erreur (Table 1) 
    estimator = AttitudeEstimator(k_q=100, k_b=200)
    q_est_hist = np.zeros((4, steps))
    bias_est_hist = np.zeros((3, steps))

    for i in range(steps):
        qe, be = estimator.update(gyro_meas[:, i], acc_meas[:, i], mag_meas[:, i], dt)
        q_est_hist[:, i] = qe.flatten()
        bias_est_hist[:, i] = be.flatten()

    # 4. Affichage des Résultats
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for i in range(4):
        axes[i].plot(time, q_true[i, :], 'k--', label='Théorique' if i==0 else "")
        axes[i].plot(time, q_est_hist[i, :], 'r', label='Estimé' if i==0 else "")
        axes[i].set_ylabel(f'q{i}')
    axes[0].set_title('Suivi des Quaternions (Convergence de l\'observateur)')
    axes[0].legend()

    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(time, bias_true[i, :], 'k--', label='Vrai Biais')
        plt.plot(time, bias_est_hist[i, :], 'b', label='Biais Estimé')
        plt.ylabel(f'Bias {["x","y","z"][i]} (rad/s)')
    plt.suptitle('Estimation du Biais du Gyroscope')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_and_plot()
    
