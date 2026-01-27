import numpy as np
from src.utils import q_to_dgm

# Constants for Magnetic Field
MU_0 = 4 * np.pi * 1e-7
M_EARTH = 7.72e22 # A m^2
RE_M = 6378137.0 # meters

def get_true_mag_field(r_eci_km):
    """
    Tilted Dipole Model.
    
    Args:
        r_eci_km (np.array): Position in ECI [km].
        
    Returns:
        np.array: Magnetic field vector in ECI [Tesla].
    """
    r_vec = r_eci_km * 1000.0 # Convert to meters
    r_norm = np.linalg.norm(r_vec)
    r_hat = r_vec / r_norm
    
    # Dipole vector (approximate tilt is 11.5 deg)
    # Assume its aligned with Earth's rotation axis (Z)
    m_vec = np.array([0., 0., -M_EARTH]) 
    
    # B = (mu0 / 4pi) * (3(m.r)r - m) / r^3
    factor = (MU_0 / (4 * np.pi)) / (r_norm**3)
    
    dot = np.dot(m_vec, r_hat)
    B_vec = factor * (3 * dot * r_hat - m_vec)
    
    return B_vec

def get_true_sun_vector(t):
    """
    Returns Sun vector in ECI.
    Assumption: Sun is fixed in ECI along X axis for short duration.
    """
    # Simply point along X
    return np.array([1., 0., 0.])

def vector_measurement_model(state, inertial_vec):
    """
    Generates expected body measurement given state and inertial vector.
    z = R(q)^T * v_inertial  (Passive rotation / Coordinate transform)
    """
    q = state[:4]
    dgm = q_to_dgm(q) 
    return dgm.T @ inertial_vec

def generate_measurement(state, inertial_vec, noise_std):
    """
    Simulates a noisy measurement.
    """
    true_meas = vector_measurement_model(state, inertial_vec)
    noise = np.random.normal(0, noise_std, 3)
    return true_meas + noise
