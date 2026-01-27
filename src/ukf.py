import numpy as np
from src.utils import q_mult, q_norm, q_inv, q_to_dgm, mrp_to_quat, quat_to_mrp, skew
from src.dynamics import rk4_step
from src.measurements import vector_measurement_model

class UKF:
    def __init__(self, initial_state, P0, Q, R):
        """
        initial_state: [q(4), beta(3)]
        P0, Q, R: Covariances (6x6 for P,Q; 3x3 for R)
        """
        self.state = initial_state
        self.P = P0
        self.Q = Q
        self.R = R
        self.n = 6 # Error state dimension
        self.alpha = 1e-3
        self.kappa = 0.0
        self.beta = 2.0
        self.lam = self.alpha**2 * (self.n + self.kappa) - self.n
        
        # Weights
        self.Wm = np.full(2*self.n + 1, 1 / (2*(self.n + self.lam)))
        self.Wc = np.full(2*self.n + 1, 1 / (2*(self.n + self.lam)))
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.lam / (self.n + self.lam) + (1 - self.alpha**2 + self.beta)
        
    def generate_sigma_points(self):
        # P = S * S.T
        try:
            S = np.linalg.cholesky(self.P + self.Q)
            # For simplicity, additive Q: P_pred = P_prop + Q.
            S = np.linalg.cholesky(self.P)
        except np.linalg.LinAlgError:
            print("Cholesky failed, using diagonal approximation")
            S = np.diag(np.sqrt(np.diag(self.P)))

        scale = np.sqrt(self.n + self.lam)
        sigma_errors = np.zeros((2*self.n + 1, self.n))
        
        # Center is 0 error
        # Right sigma points
        for i in range(self.n):
            sigma_errors[i+1] = scale * S[:, i]
        # Left sigma points
        for i in range(self.n):
            sigma_errors[self.n + i + 1] = -scale * S[:, i]
            
        return sigma_errors

    def predict(self, omega_meas, dt):
        # Generate Sigma Points (Error State)
        sigma_errors = self.generate_sigma_points()
        
        # Map to Full State and Propagate
        propagated_states = []
        nom_q = self.state[:4]
        nom_beta = self.state[4:]
        
        for i in range(2*self.n + 1):
            # Error components
            d_mrp = sigma_errors[i, :3]
            d_beta = sigma_errors[i, 3:]
            
            # Reconstruct state
            # q = nom_q * q(mrp)
            dq = mrp_to_quat(d_mrp)

            q_i = q_mult(nom_q, dq)
            beta_i = nom_beta + d_beta
            
            state_i = np.concatenate([q_i, beta_i])
            
            # Propagate
            omega_est_i = omega_meas - beta_i
            prop_state_i = rk4_step(state_i, omega_est_i, dt)
            propagated_states.append(prop_state_i)
            
        propagated_states = np.array(propagated_states)
        
        # Compute Mean State
        # Bias mean: linear
        mean_beta = np.sum(self.Wm[:, None] * propagated_states[:, 4:], axis=0)
        
        # Quaternion mean: Gradient Descent or Iterative
        # Start with center point propagated
        mean_q = propagated_states[0, :4] 
        max_iter = 5
        for _ in range(max_iter):
            e_sum = np.zeros(3)
            for i in range(2*self.n + 1):
                q_i = propagated_states[i, :4]
                # Error q_err = q_mean_inv * q_i
                q_err = q_mult(q_inv(mean_q), q_i)
                # Convert to MRP
                e_i = quat_to_mrp(q_err)
                e_sum += self.Wm[i] * e_i
            
            # Update mean_q
            # q_update = q(e_sum)
            # mean_q = mean_q * q_update
            if np.linalg.norm(e_sum) < 1e-6:
                break
            q_upd = mrp_to_quat(e_sum)
            mean_q = q_mult(mean_q, q_upd)
            mean_q = q_norm(mean_q)
            
        self.state = np.concatenate([mean_q, mean_beta])
        
        # Compute Covariance
        P_pred = np.zeros((self.n, self.n))
        for i in range(2*self.n + 1):
            # Error state relative to new mean
            q_i = propagated_states[i, :4]
            beta_i = propagated_states[i, 4:]
            
            # d_beta
            d_beta = beta_i - mean_beta
            # d_mrp
            q_err = q_mult(q_inv(mean_q), q_i)
            d_mrp = quat_to_mrp(q_err)
            
            delta_x = np.concatenate([d_mrp, d_beta])
            P_pred += self.Wc[i] * np.outer(delta_x, delta_x)
            
        self.P = P_pred + self.Q
        
    def update(self, z_meas, z_ref):
        # New sigma points from P_pred
        sigma_errors = self.generate_sigma_points()
        
        # Map to predicted measurements
        Z_sigmas = []
        nom_q = self.state[:4]
        
        meas_dim = 3
        
        for i in range(2*self.n + 1):
            d_mrp = sigma_errors[i, :3]
            # No bias error needed for measurement if measurement only depends on q
            
            dq = mrp_to_quat(d_mrp)
            q_i = q_mult(nom_q, dq)
            
            # Predict measurement
            # Use shared measurement model to ensure consistency (Passive rotation)
            z_pred_i = vector_measurement_model(np.concatenate([q_i, np.zeros(3)]), z_ref)
            Z_sigmas.append(z_pred_i)
            
        Z_sigmas = np.array(Z_sigmas)
        
        # Mean measurement
        z_mean = np.sum(self.Wm[:, None] * Z_sigmas, axis=0)
        
        # Covariances
        P_zz = np.zeros((meas_dim, meas_dim))
        P_xz = np.zeros((self.n, meas_dim))
        
        for i in range(2*self.n + 1):
            z_diff = Z_sigmas[i] - z_mean
            
            # State error from sigma point generation (sigma_errors[i])
            x_diff = sigma_errors[i] 
            
            P_zz += self.Wc[i] * np.outer(z_diff, z_diff)
            P_xz += self.Wc[i] * np.outer(x_diff, z_diff)
            
        P_zz += self.R
        
        # Kalman Gain
        K = P_xz @ np.linalg.inv(P_zz)
        
        # Update
        y = z_meas - z_mean
        delta_x = K @ y
        
        # Update State
        d_mrp = delta_x[:3]
        d_beta = delta_x[3:]
        
        dq = mrp_to_quat(d_mrp)
        self.state[:4] = q_mult(self.state[:4], dq)
        self.state[:4] = q_norm(self.state[:4])
        self.state[4:] += d_beta
        
        # Update P
        self.P = self.P - K @ P_zz @ K.T
        
        # NIS Calculation
        try:
            nis = y.T @ np.linalg.inv(P_zz) @ y
        except np.linalg.LinAlgError:
            nis = 0.0
            
        return nis
