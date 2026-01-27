import numpy as np
from src.utils import q_mult, q_inv, quat_to_mrp

def compute_nees(est_q, est_bias, true_q, true_bias, P):
    """
    Computes Normalized Estimation Error Squared (NEES).
    epsilon = [delta_mrp; delta_bias]
    NEES = epsilon^T * P^-1 * epsilon
    """
    # Attitude Error (MRP)
    # dq = q_est_inv * q_true
    q_err = q_mult(q_inv(est_q), true_q)
    
    # MRP is 3 param representation of error quaternion
    # 4 * MRP is roughly Theta.
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

def compute_3sigma(P):
    """
    Returns 3-sigma bounds for each state component.
    """
    variances = np.diag(P)
    std = np.sqrt(np.abs(variances)) # abs just in case
    return 3 * std
