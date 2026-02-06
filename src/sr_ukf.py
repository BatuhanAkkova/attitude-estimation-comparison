import numpy as np
from scipy.linalg import qr, cholesky, solve_triangular
from src.utils import q_mult, q_norm, q_inv, q_to_dgm, mrp_to_quat, quat_to_mrp, skew
from src.dynamics import rk4_step
from src.measurements import vector_measurement_model

def cholupdate(S, x, weight):
    """
    Rank-1 update of Cholesky factor S.
    S: Cholesky factor (Upper triangular)
    x: vector to update with
    weight: weight of the update
    Returns updated S.
    """
    if weight == 0:
        return S
    
    # Simple wrapper for numerical stability: 
    # Use the QR of [S; sqrt(w)*x.T]
    
    if weight > 0:
        # Update
        M = np.vstack((S, np.sqrt(weight) * x))
        _, R = qr(M, mode='economic')
        return R[:S.shape[0], :]
    else:
        # Fallback to a more stable re-calculation if weight is negative (Wc[0] can be negative).
        P = S.T @ S + weight * np.outer(x, x)
        try:
            return cholesky(P)
        except np.linalg.LinAlgError:
            return S

class SRUKF:
    def __init__(self, initial_state, P0, Q, R):
        """
        initial_state: [q(4), beta(3)]
        P0, Q, R: Covariances
        """
        self.state = initial_state
        self.S = cholesky(P0) # Upper triangular
        self.Q = Q
        self.R = R
        self.n = 6
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
        scale = np.sqrt(self.n + self.lam)
        sigma_errors = np.zeros((2*self.n + 1, self.n))
        
        # Center is 0
        # self.S is upper triangular, so S.T is lower
        for i in range(self.n):
            sigma_errors[i+1] = scale * self.S[i, :] # Using rows of upper triangular S is like columns of S.T
            sigma_errors[self.n + i + 1] = -scale * self.S[i, :]
            
        return sigma_errors

    def predict(self, omega_meas, dt):
        sigma_errors = self.generate_sigma_points()
        
        propagated_states = []
        nom_q = self.state[:4]
        nom_beta = self.state[4:]
        
        for i in range(2*self.n + 1):
            d_mrp = sigma_errors[i, :3]
            d_beta = sigma_errors[i, 3:]
            
            dq = mrp_to_quat(d_mrp)
            q_i = q_mult(nom_q, dq)
            beta_i = nom_beta + d_beta
            
            state_i = np.concatenate([q_i, beta_i])
            omega_est_i = omega_meas - beta_i
            prop_state_i = rk4_step(state_i, omega_est_i, dt)
            propagated_states.append(prop_state_i)
            
        propagated_states = np.array(propagated_states)
        
        # Mean Bias
        mean_beta = np.sum(self.Wm[:, None] * propagated_states[:, 4:], axis=0)
        
        # Mean Quaternion
        mean_q = propagated_states[0, :4] 
        max_iter = 5
        for _ in range(max_iter):
            e_sum = np.zeros(3)
            for i in range(2*self.n + 1):
                q_i = propagated_states[i, :4]
                q_err = q_mult(q_inv(mean_q), q_i)
                e_i = quat_to_mrp(q_err)
                e_sum += self.Wm[i] * e_i
            
            if np.linalg.norm(e_sum) < 1e-6:
                break
            q_upd = mrp_to_quat(e_sum)
            mean_q = q_mult(mean_q, q_upd)
            mean_q = q_norm(mean_q)
            
        self.state = np.concatenate([mean_q, mean_beta])
        
        # Square Root Covariance Update
        # 1. QR of weighted deviations (points 1 to 2n)
        X = np.zeros((2*self.n, self.n))
        for i in range(1, 2*self.n + 1):
            q_i = propagated_states[i, :4]
            beta_i = propagated_states[i, 4:]
            
            d_beta = beta_i - mean_beta
            q_err = q_mult(q_inv(mean_q), q_i)
            d_mrp = quat_to_mrp(q_err)
            
            X[i-1] = np.sqrt(self.Wc[1]) * np.concatenate([d_mrp, d_beta])
            
        # Add process noise square root
        Sq = cholesky(self.Q)
        _, S_pred = qr(np.vstack((X, Sq)), mode='economic')
        S_pred = S_pred[:self.n, :]
        
        # 2. Rank-1 update for first point
        q_0 = propagated_states[0, :4]
        beta_0 = propagated_states[0, 4:]
        d_beta_0 = beta_0 - mean_beta
        q_err_0 = q_mult(q_inv(mean_q), q_0)
        d_mrp_0 = quat_to_mrp(q_err_0)
        x_0 = np.concatenate([d_mrp_0, d_beta_0])
        
        self.S = cholupdate(S_pred, x_0, self.Wc[0])
        
    def update(self, z_meas, z_ref):
        sigma_errors = self.generate_sigma_points()
        
        Z_sigmas = []
        nom_q = self.state[:4]
        meas_dim = 3
        
        for i in range(2*self.n + 1):
            d_mrp = sigma_errors[i, :3]
            dq = mrp_to_quat(d_mrp)
            q_i = q_mult(nom_q, dq)
            z_pred_i = vector_measurement_model(np.concatenate([q_i, np.zeros(3)]), z_ref)
            Z_sigmas.append(z_pred_i)
            
        Z_sigmas = np.array(Z_sigmas)
        z_mean = np.sum(self.Wm[:, None] * Z_sigmas, axis=0)
        
        # Square Root Measurement Covariance
        Z_dev = np.zeros((2*self.n, meas_dim))
        for i in range(1, 2*self.n + 1):
            Z_dev[i-1] = np.sqrt(self.Wc[1]) * (Z_sigmas[i] - z_mean)
            
        Sr = cholesky(self.R)
        _, S_zz = qr(np.vstack((Z_dev, Sr)), mode='economic')
        S_zz = S_zz[:meas_dim, :meas_dim]
        S_zz = cholupdate(S_zz, Z_sigmas[0] - z_mean, self.Wc[0])
        
        # Cross Covariance
        P_xz = np.zeros((self.n, meas_dim))
        for i in range(2*self.n + 1):
            z_diff = Z_sigmas[i] - z_mean
            x_diff = sigma_errors[i]
            P_xz += self.Wc[i] * np.outer(x_diff, z_diff)
            
        # Kalman Gain K = P_xz / P_zz = P_xz / (S_zz.T @ S_zz)
        # Solve (S_zz.T @ S_zz) @ K.T = P_xz.T
        # 1. solve S_zz.T @ Y = P_xz.T
        # 2. solve S_zz @ K.T = Y
        Y = solve_triangular(S_zz, P_xz.T, lower=False, trans='T')
        K = solve_triangular(S_zz, Y, lower=False).T
        
        # Update
        y = z_meas - z_mean
        delta_x = K @ y
        
        self.state[4:] += delta_x[3:]
        dq = mrp_to_quat(delta_x[:3])
        self.state[:4] = q_norm(q_mult(self.state[:4], dq))
        
        # Covariance SR Update
        U = K @ S_zz
        for i in range(meas_dim):
            self.S = cholupdate(self.S, U[:, i], -1.0)
            
        # NIS
        try:
            # y.T @ inv(P_zz) @ y = y.T @ inv(S_zz.T @ S_zz) @ y
            # = (inv(S_zz.T) @ y).T @ (inv(S_zz.T) @ y)
            sol = solve_triangular(S_zz, y, lower=False, trans='T')
            nis = sol.T @ sol
        except:
            nis = 0.0
            
        return nis

    @property
    def P(self):
        return self.S.T @ self.S
