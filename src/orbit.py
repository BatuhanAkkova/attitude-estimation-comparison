import numpy as np

# Constants
MU = 398600.4418   # Earth gravitational parameter [km^3/s^2]
J2 = 1.0826267e-3  # J2 Zonal Harmonic
RE = 6378.137      # Earth Equatorial Radius [km]

def j2_accel(r_vec):
    """
    Computes acceleration due to Earth's gravity with J2 perturbation.
    
    Args:
        r_vec (np.array): Position vector [km] in ECI.
        
    Returns:
        np.array: Acceleration vector [km/s^2].
    """
    r_norm = np.linalg.norm(r_vec)
    x, y, z = r_vec
    
    # Common terms
    r2 = r_norm**2
    r3 = r2 * r_norm
    j2_factor = 1.5 * J2 * MU * (RE**2) / (r2 * r3)
    z2_r2 = (z / r_norm)**2
    
    # Two-Body acceleration
    a_2body = -MU * r_vec / r3
    
    # J2 Perturbation
    a_j2_x = j2_factor * x * (5 * z2_r2 - 1)
    a_j2_y = j2_factor * y * (5 * z2_r2 - 1)
    a_j2_z = j2_factor * z * (5 * z2_r2 - 3)
    
    a_j2 = np.array([a_j2_x, a_j2_y, a_j2_z])
    
    return a_2body + a_j2

def rk4_orbit_step(state, dt):
    """
    Runge-Kutta 4 integration step for orbit dynamics.
    
    Args:
        state (np.array): [x, y, z, vx, vy, vz] (km, km/s)
        dt (float): Time step (s)
        
    Returns:
        np.array: Next state.
    """
    r = state[:3]
    v = state[3:]
    
    # k1
    k1_v = j2_accel(r)
    k1_r = v
    
    # k2
    k2_r = v + 0.5 * dt * k1_v
    k2_v = j2_accel(r + 0.5 * dt * k1_r)
    
    # k3
    k3_r = v + 0.5 * dt * k2_v
    k3_v = j2_accel(r + 0.5 * dt * k2_r)
    
    # k4
    k4_r = v + dt * k3_v
    k4_v = j2_accel(r + dt * k3_r)
    
    # Update
    next_r = r + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    next_v = v + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
    return np.concatenate([next_r, next_v])
