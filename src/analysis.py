import numpy as np
from src.utils import q_mult, q_inv, quat_to_mrp
from scipy.stats import chi2
from scipy.linalg import sqrtm

def compute_angle_error(q_true, q_est):
    """
    Computes representation-invariant angle error:
    theta = 2 * acos(|q_true^T q_est|)
    """
    dot = np.abs(np.dot(q_true, q_est))
    # Clip for numerical stability
    dot = np.clip(dot, 0.0, 1.0)
    return 2.0 * np.arccos(dot)

def compute_nees(est_q, est_bias, true_q, true_bias, P):
    """
    Computes Normalized Estimation Error Squared (NEES).
    epsilon = [4*delta_mrp; delta_bias]
    NEES = epsilon^T * P^-1 * epsilon
    """
    # Attitude Error (MRP)
    q_err = q_mult(q_inv(est_q), true_q)
    delta_mrp = quat_to_mrp(q_err)
    delta_theta = 4.0 * delta_mrp 
    
    # Bias Error
    delta_bias = true_bias - est_bias
    
    # Full Error State
    epsilon = np.concatenate([delta_theta, delta_bias])
    
    # NEES
    try:
        current_nees = epsilon.T @ np.linalg.inv(P) @ epsilon
    except np.linalg.LinAlgError:
        current_nees = np.nan
        
    return current_nees

def compute_consistency_bounds(dof, N_runs, confidence=0.95):
    """
    Computes average consistency (NEES/NIS) bounds for N runs.
    """
    alpha = 1.0 - confidence
    lower = chi2.ppf(alpha / 2, N_runs * dof) / N_runs
    upper = chi2.ppf(1.0 - alpha / 2, N_runs * dof) / N_runs
    return lower, upper

def check_numerical_stability(P):
    """
    Checks for covariance matrix positive definiteness and condition number.
    """
    cond = np.linalg.cond(P)
    try:
        np.linalg.cholesky(P)
        pos_def = True
    except np.linalg.LinAlgError:
        pos_def = False
    return cond, pos_def

def compute_normalized_errors(est_q, est_bias, true_q, true_bias, P):
    """
    Computes normalized error: e = P^{-1/2} * (x_true - x_est)
    Expected: zero mean, unit variance, Gaussian distribution.
    """
    q_err = q_mult(q_inv(est_q), true_q)
    delta_theta = 4.0 * quat_to_mrp(q_err)
    delta_bias = true_bias - est_bias
    epsilon = np.concatenate([delta_theta, delta_bias])
    
    try:
        # P = S * S^T -> e = S^-1 * epsilon
        S = np.linalg.cholesky(P)
        norm_err = np.linalg.solve(S, epsilon)
    except np.linalg.LinAlgError:
        norm_err = np.full(len(epsilon), np.nan)
        
    return norm_err

