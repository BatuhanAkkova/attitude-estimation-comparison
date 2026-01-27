import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time as pytime

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import q_norm, q_mult, q_inv, q_to_dgm, quat_to_mrp 
from src.dynamics import rk4_step
from src.orbit import rk4_orbit_step
from src.measurements import get_true_mag_field, get_true_sun_vector
from src.sensors.gyro import Gyroscope
from src.sensors.magnetometer import Magnetometer
from src.sensors.sun_sensor import SunSensor

from src.mekf import MEKF
from src.ukf import UKF

def run_simulation():
    # Simulation Parameters
    dt = 0.1
    t_end = 500.0
    steps = int(t_end / dt)
    time = np.linspace(0, t_end, steps)
    
    # Initial Orbit (LEO, 500km altitude, 45 deg inclination)
    RE = 6378.137
    h_orbit = 500.0
    r_mag = RE + h_orbit
    v_mag = np.sqrt(398600.4418 / r_mag)
    
    # Position at equator
    r_init = np.array([r_mag, 0., 0.])
    # Velocity tilted 45 deg
    inc_rad = np.radians(45.0)
    v_init = np.array([0., v_mag * np.cos(inc_rad), v_mag * np.sin(inc_rad)])
    
    true_orbit_state = np.concatenate([r_init, v_init])
    
    # Initial Attitude State
    true_q = np.array([0., 0., 0., 1.])
    # True Bias initialized but will walk
    init_true_bias = np.array([0.01, -0.02, 0.005]) 
    
    # Sensors
    gyro = Gyroscope(initial_bias=init_true_bias, noise_std=0.0001, bias_walk_std=1e-6)
    mag_sensor = Magnetometer(noise_std=0.1e-6) # T
    
    sun_sensor = SunSensor(noise_std=0.005) # approx 0.3 deg
    
    # Filters Initialization
    init_q_est = np.array([0.1, 0., 0., 0.9])
    init_q_est = q_norm(init_q_est)
    init_bias_est = np.array([0., 0., 0.])
    init_state_est = np.concatenate([init_q_est, init_bias_est])
    
    P0 = np.eye(6) * 0.1
    Q = np.eye(6) * 1e-4
    Q[3:, 3:] = np.eye(3) * 1e-8 # Lower Q for bias
    R_mag = np.eye(3) * (mag_sensor.noise_std**2)
    R_sun = np.eye(3) * (sun_sensor.noise_std**2)
    
    # or better, Update MEKF/UKF to accept R in update() (Future Task).
    
    # CRITICAL: We should normalize magnetometer readings for the filter or adjust R dynamically.
    # Let's normalize Mag measurements to unit vectors for consistency with Sun sensor.
    
    R = np.eye(3) * (0.01**2) # Generic R for unit vectors
    mekf = MEKF(init_state_est.copy(), P0.copy(), Q.copy(), R.copy())
    ukf = UKF(init_state_est.copy(), P0.copy(), Q.copy(), R.copy())
    
    # Storage
    history = {
        'time': [],
        'true_q': [], 'true_bias': [],
        'mekf_q': [], 'mekf_bias': [], 'mekf_err': [],
        'ukf_q': [], 'ukf_bias': [], 'ukf_err': []
    }
    
    print(f"Starting simulation for {steps} steps ({t_end} s)...")
    
    # True Angular Velocity Function (Tumbling)
    def get_true_omega(t):
        return np.array([0.06 * np.sin(0.1*t), 0.02 * np.cos(0.05*t), -0.01])

    for k in range(steps):
        t = time[k]
        
        # Dynamics Propagation
        # Orbit
        true_orbit_state = rk4_orbit_step(true_orbit_state, dt)
        r_eci = true_orbit_state[:3]
        
        # Attitude (Kinematics with synthetic omega)
        true_omega_val = get_true_omega(t)
        # Propagate q
        temp_att_state = np.concatenate([true_q, np.zeros(3)]) # Bias handled by sensor
        temp_att_next = rk4_step(temp_att_state, true_omega_val, dt)
        true_q = temp_att_next[:4]
        
        # Environment
        B_eci = get_true_mag_field(r_eci)
        S_eci = get_true_sun_vector(t)
        
        # Measurements
        # Transform to Body
        dgm_true = q_to_dgm(true_q)
        B_body = dgm_true.T @ B_eci
        S_body = dgm_true.T @ S_eci
        
        # Sensor Readings
        omega_meas = gyro.measure(true_omega_val, dt)
        
        # Normalize Mag for filter input (simple approach)
        B_meas_raw = mag_sensor.measure(B_body)
        B_norm = np.linalg.norm(B_meas_raw)
        B_meas_unit = B_meas_raw / B_norm if B_norm > 0 else np.array([1,0,0])
        B_ref_unit = B_eci / np.linalg.norm(B_eci) if np.linalg.norm(B_eci) > 0 else np.array([1,0,0])
        
        S_meas = sun_sensor.measure(S_body) # Returns unit vector or None
        
        # Filter Step
        
        # MEKF
        mekf.predict(omega_meas, dt)
        # Update Mag
        mekf.R = np.eye(3) * (0.01**2) # Approximate noise for unit vector
        mekf.update(B_meas_unit, B_ref_unit)
        # Update Sun
        if S_meas is not None:
            mekf.R = np.eye(3) * (sun_sensor.noise_std**2)
            mekf.update(S_meas, S_eci)
            
        # UKF
        ukf.predict(omega_meas, dt)
        ukf.R = np.eye(3) * (0.01**2)
        ukf.update(B_meas_unit, B_ref_unit)
        if S_meas is not None:
            ukf.R = np.eye(3) * (sun_sensor.noise_std**2)
            ukf.update(S_meas, S_eci)
            
        # Storage
        history['time'].append(t)
        history['true_q'].append(true_q.copy())
        history['true_bias'].append(gyro.get_bias()) # Current bias
        
        history['mekf_q'].append(mekf.state[:4].copy())
        history['mekf_bias'].append(mekf.state[4:].copy())
        
        history['ukf_q'].append(ukf.state[:4].copy())
        history['ukf_bias'].append(ukf.state[4:].copy())
        
        # Errors
        q_err_mekf = q_mult(q_inv(mekf.state[:4]), true_q)
        history['mekf_err'].append(np.linalg.norm(quat_to_mrp(q_err_mekf)) * 4)
        
        q_err_ukf = q_mult(q_inv(ukf.state[:4]), true_q)
        history['ukf_err'].append(np.linalg.norm(quat_to_mrp(q_err_ukf)) * 4)

    # Convert lists to arrays
    for k in history:
        history[k] = np.array(history[k])
        
    print("Optimization Complete. Saving Plots...")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(history['time'], history['mekf_err'], label='MEKF Error (rad)', alpha=0.7)
    plt.plot(history['time'], history['ukf_err'], label='UKF Error (rad)', linestyle='--', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Attitude Error (rad)')
    plt.yscale('log')
    plt.title('Attitude Estimation Error (Dynamic Orbit & Env)')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig('figures/attitude_error.png')
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['time'], history['true_bias'][:, 0], 'k-', label='True Bias X')
    plt.plot(history['time'], history['mekf_bias'][:, 0], 'r--', label='MEKF Est X')
    plt.plot(history['time'], history['ukf_bias'][:, 0], 'b:', label='UKF Est X')
    plt.xlabel('Time (s)')
    plt.ylabel('Bias (rad/s)')
    plt.title('Bias Estimation (Walking Bias)')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/bias_estimation.png')
    
    print("Plots saved to figures/attitude_error.png and figures/bias_estimation.png")

if __name__ == '__main__':
    run_simulation()
