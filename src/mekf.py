import numpy as np
from src.utils import q_mult, q_norm, q_inv, q_to_dgm, skew, quat_to_mrp, mrp_to_quat
from src.dynamics import rk4_step, kinematics

class MEKF:
    def __init__(self, initial_state, P0, Q, R):
        """
        initial_state: [q(4), beta(3)]
        P0: Initial covariance (6x6)
        Q: Process noise covariance (6x6)
        R: Measurement noise covariance (3x3 per sensor)
        """
        self.state = initial_state
        self.P = P0
        self.Q = Q
        self.R = R
    
    def predict(self, omega_meas, dt):
        # Propagate Nominal State
        bias_hat = self.state[4:]
        omega_hat = omega_meas - bias_hat
        
        # Propagate quaternion
        self.state = rk4_step(self.state, omega_hat, dt)
        
        # 2. Propagate Covariance
        # Error Dynamics Matrix F (Continuous)
        # d(delta_theta)/dt = -[omega_hat x] * delta_theta - delta_beta
        # d(delta_beta)/dt = 0
        
        # Discrete transition matrix Phi approx I + F*dt
        F = np.zeros((6, 6))
        F[:3, :3] = -skew(omega_hat)
        F[:3, 3:] = -np.eye(3)
        
        Phi = np.eye(6) + F * dt
        
        self.P = Phi @ self.P @ Phi.T + self.Q
        
    def update(self, z_meas, z_ref):
        # Predicted Measurement
        # h(x) = A(q) * r_inertial
        # A(q) is Inertial to Body DCM
        q_est = self.state[:4]

        dgm = q_to_dgm(q_est)
        A = dgm.T
        z_pred = A @ z_ref
        
        # Measurement Matrix H
        # z = z_pred + H * delta_x + v
        # delta_z = [A * r] x delta_theta 
        #         = - [delta_theta x] A * r
        #         = [A * r x] delta_theta
        # H_theta = [z_pred x] = skew(z_pred)
        # H_beta = 0
        
        H = np.zeros((3, 6))
        H[:3, :3] = skew(z_pred)
        
        # Kalman Gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Correction
        y = z_meas - z_pred
        delta_x = K @ y
        
        delta_theta = delta_x[:3]
        delta_beta = delta_x[3:]
        
        # Update Bias
        self.state[4:] += delta_beta
        
        # Update Quaternion - small angle approx 
        dq = np.concatenate([0.5 * delta_theta, [1.0]])
        dq = q_norm(dq)
        
        self.state[:4] = q_mult(self.state[:4], dq)
        self.state[:4] = q_norm(self.state[:4])
        
        # Update Covariance
        I = np.eye(6)
        # Joseph form for stability
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
        
        # NIS Calculation
        # nis = y.T * S^-1 * y
        # We need S_inv. We computed inv(S) earlier? No, we computed np.linalg.inv(S) inline.
        # Let's recompute or handle efficiency later. For now, recompute.
        try:
            nis = y.T @ np.linalg.inv(S) @ y
        except np.linalg.LinAlgError:
            nis = 0.0
            
        return nis
