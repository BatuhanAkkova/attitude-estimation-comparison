import numpy as np
from src.utils import q_mult, q_norm, q_inv, q_to_dgm, skew
from src.dynamics import rk4_step, kinematics
from src.mekf import MEKF

class AEKF(MEKF):
    def __init__(self, initial_state, P0, Q, R, alpha=0.95):
        """
        Adaptive EKF based on MEKF.
        alpha: Smoothing factor for R adaptation (0 to 1). 
               Higher means slower adaptation (more trust in history).
        """
        super().__init__(initial_state, P0, Q, R)
        self.alpha = alpha
        
    def update(self, z_meas, z_ref):
        # 1. Standard MEKF Update Pre-calculations
        q_est = self.state[:4]
        dgm = q_to_dgm(q_est)
        A = dgm.T
        z_pred = A @ z_ref
        
        H = np.zeros((3, 6))
        H[:3, :3] = skew(z_pred)
        
        # 2. Innovation
        y = z_meas - z_pred
        
        # 3. Adaptation of R
        # R_new = alpha * R_old + (1 - alpha) * (y @ y.T)
        
        # For multi-sensor, adapt the current R block.
        # Innovation-based adaptation
        innovation_cov = np.outer(y, y)
        self.R = self.alpha * self.R + (1.0 - self.alpha) * innovation_cov
        
        # Ensure R remains positive definite and has a minimum floor to avoid collapse
        # (Numerical safeguard)
        self.R += np.eye(3) * 1e-9

        # 4. Proceed with standard K calculation and update
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        delta_x = K @ y
        delta_theta = delta_x[:3]
        delta_beta = delta_x[3:]
        
        self.state[4:] += delta_beta
        dq = np.concatenate([0.5 * delta_theta, [1.0]])
        dq = q_norm(dq)
        self.state[:4] = q_norm(q_mult(self.state[:4], dq))
        
        # Joseph form for stability
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
        
        try:
            nis = y.T @ np.linalg.inv(S) @ y
        except np.linalg.LinAlgError:
            nis = 0.0
            
        return nis
