import numpy as np

def q_mult(q1, q2):
    """
    Quaternion multiplication q_out = q1 * q2
    Format: [x, y, z, w]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def q_conj(q):
    """Quaternion conjugate"""
    return np.array([-q[0], -q[1], -q[2], q[3]])

def q_inv(q):
    """Quaternion inverse"""
    return q_conj(q) / (np.linalg.norm(q)**2)

def q_norm(q):
    """Normalize quaternion"""
    return q / np.linalg.norm(q)

def q_to_dgm(q):
    """
    Quaternion to Direction Cosine Matrix (Body to Inertial)
    Using standard conversion for scalar-last quaternion [x,y,z,w]
    """
    q = q_norm(q)
    x, y, z, w = q
    
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])

def mrp_to_quat(mrp):
    """Modified Rodrigues Parameters to Quaternion [x,y,z,w]"""
    m_sq = np.dot(mrp, mrp)
    scale = 1.0 / (1.0 + m_sq)
    return np.array([
        2*mrp[0]*scale,
        2*mrp[1]*scale,
        2*mrp[2]*scale,
        (1 - m_sq)*scale
    ])

def quat_to_mrp(q):
    """Quaternion [x,y,z,w] to MRP"""
    # simple standard conversion:
    den = 1.0 + q[3]
    return np.array([q[0], q[1], q[2]]) / den

def skew(v):
    """Skew symmetric matrix from vector"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def eci_to_lvlh(r_eci, v_eci):
    """
    Computes Rotation Matrix from ECI to LVLH frame.
    LVLH Definition (Earth Pointing):
    - Z: Points towards Earth Center (Nadir) = -r_norm
    - Y: Opposite to Orbit Normal = - (r x v)_norm
    - X: Completes triad (Roughly Velocity) = Y x Z
    
    Args:
        r_eci (np.array): Position vector [km]
        v_eci (np.array): Velocity vector [km/s]
        
    Returns:
        np.array: 3x3 Rotation Matrix (ECI -> LVLH)
    """
    r_norm = r_eci / np.linalg.norm(r_eci)
    h = np.cross(r_eci, v_eci)
    h_norm = h / np.linalg.norm(h)
    
    z_lvlh = -r_norm
    y_lvlh = -h_norm
    x_lvlh = np.cross(y_lvlh, z_lvlh)
    
    R_lvlh2eci = np.column_stack([x_lvlh, y_lvlh, z_lvlh])
    return R_lvlh2eci.T

def q_from_vectors(u, v):
    """
    Computes quaternion that rotates vector u to vector v.
    q * u * q_inv = v
    """
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    
    dot = np.dot(u, v)
    
    if dot > 0.999999:
        return np.array([0., 0., 0., 1.])
    elif dot < -0.999999:
        # 180 degree rotation around any orthogonal axis
        # Find orthogonal vector
        w = np.cross(u, np.array([1., 0., 0.]))
        if np.linalg.norm(w) < 0.01:
            w = np.cross(u, np.array([0., 1., 0.]))
        w = w / np.linalg.norm(w)
        return np.array([w[0], w[1], w[2], 0.])
    
    axis = np.cross(u, v)
    q_xyz = axis
    q_w = 1.0 + dot
    
    q = np.concatenate([q_xyz, [q_w]])
    return q_norm(q)
