import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import multiprocessing
from tqdm import tqdm

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
from src.analysis import compute_nees, compute_3sigma

def run_worker(args):
    """
    Worker function for a single Monte Carlo run.
    args: (run_idx, config, time_arr, dt, steps, simulation_params)
    """
    run_idx, config, time_arr, dt, steps = args
    
    init_err_deg = config.get('init_error_deg', 10.0)
    bias_scale = config.get('bias_scale', 1.0)
    eclipse_sim = config.get('eclipse_sim', False)
    
    # Initialize Storage for single run
    nees_mekf = np.zeros(steps)
    nees_ukf = np.zeros(steps)
    nis_mekf = np.zeros(steps)
    nis_ukf = np.zeros(steps)
    
    err_mekf = np.zeros((steps, 6))
    err_ukf = np.zeros((steps, 6))
    cov_mekf = np.zeros((steps, 6))
    
    # Random Initial Conditions
    RE = 6378.137
    r_mag = RE + 500.0
    v_mag = np.sqrt(398600.4418 / r_mag)
    inc_rad = np.radians(45.0)
    true_orbit_state = np.array([r_mag, 0., 0., 0., v_mag * np.cos(inc_rad), v_mag * np.sin(inc_rad)])
    
    # Random Initial Attitude
    # Usually np.random is seeded per process, but to be safe:
    np.random.seed(os.getpid() + run_idx) 
    
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(0, 2*np.pi)
    true_q = np.array([axis[0]*np.sin(angle/2), axis[1]*np.sin(angle/2), axis[2]*np.sin(angle/2), np.cos(angle/2)])
    
    # Random Initial Bias
    init_true_bias = np.random.normal(0, 0.01 * bias_scale, 3) 
    
    # Sensors
    gyro = Gyroscope(initial_bias=init_true_bias, noise_std=0.0001, bias_walk_std=1e-6)
    mag_sensor = Magnetometer(noise_std=0.1e-6) 
    sun_sensor = SunSensor(noise_std=0.005) 
    
    # Estimators
    err_axis_est = np.random.randn(3)
    err_axis_est /= np.linalg.norm(err_axis_est)
    err_angle_est = np.radians(init_err_deg) 
    dq_est = np.array([err_axis_est[0]*np.sin(err_angle_est/2), err_axis_est[1]*np.sin(err_angle_est/2), err_axis_est[2]*np.sin(err_angle_est/2), np.cos(err_angle_est/2)])
    init_q_est = q_mult(true_q, dq_est) 
    
    init_bias_est = np.zeros(3)
    init_state_est = np.concatenate([init_q_est, init_bias_est])
    
    P0 = np.eye(6) * (0.1 if init_err_deg < 50 else 1.0) 
    Q = np.eye(6) * 1e-4
    Q[3:, 3:] = np.eye(3) * 1e-8
    R_generic = np.eye(3) * (0.01**2)
    
    mekf = MEKF(init_state_est.copy(), P0.copy(), Q.copy(), R_generic.copy())
    ukf = UKF(init_state_est.copy(), P0.copy(), Q.copy(), R_generic.copy())
    
    def get_true_omega(t):
       return np.array([0.06 * np.sin(0.1*t + run_idx), 0.02 * np.cos(0.05*t), -0.01])

    for k in range(steps):
        t = time_arr[k]
        
        in_eclipse = False
        if eclipse_sim:
            if 50.0 <= t <= 150.0:
                in_eclipse = True
        
        # Dynamics
        true_orbit_state = rk4_orbit_step(true_orbit_state, dt)
        r_eci = true_orbit_state[:3]
        
        true_omega_val = get_true_omega(t)
        temp_state = np.concatenate([true_q, np.zeros(3)])
        temp_next = rk4_step(temp_state, true_omega_val, dt)
        true_q = temp_next[:4]
        current_bias = gyro.get_bias()
        
        # Env
        B_eci = get_true_mag_field(r_eci)
        S_eci = get_true_sun_vector(t)
        
        dgm_true = q_to_dgm(true_q)
        B_body = dgm_true.T @ B_eci
        S_body = dgm_true.T @ S_eci
        
        # Meas
        omega_meas = gyro.measure(true_omega_val, dt)
        
        B_meas_raw = mag_sensor.measure(B_body)
        B_norm = np.linalg.norm(B_meas_raw)
        B_meas_unit = B_meas_raw / B_norm if B_norm > 0 else np.array([1,0,0])
        B_ref_unit = B_eci / np.linalg.norm(B_eci) if np.linalg.norm(B_eci) > 0 else np.array([1,0,0])
        
        S_meas = sun_sensor.measure(S_body, in_eclipse=in_eclipse)
        
        # MEKF
        mekf.predict(omega_meas, dt)
        
        nis_m = 0.0
        count_m = 0
        
        mekf.R = np.eye(3) * (0.01**2)
        nis_val = mekf.update(B_meas_unit, B_ref_unit) 
        nis_m += nis_val
        count_m += 1
        
        if S_meas is not None:
            mekf.R = np.eye(3) * (sun_sensor.noise_std**2)
            nis_val = mekf.update(S_meas, S_eci)
            nis_m += nis_val
            count_m += 1
        
        nis_mekf[k] = nis_m / max(1, count_m)
        
        # UKF
        ukf.predict(omega_meas, dt)
        
        nis_u = 0.0
        count_u = 0
        
        ukf.R = np.eye(3) * (0.01**2)
        nis_val = ukf.update(B_meas_unit, B_ref_unit)
        nis_u += nis_val
        count_u += 1
        
        if S_meas is not None:
            ukf.R = np.eye(3) * (sun_sensor.noise_std**2)
            nis_val = ukf.update(S_meas, S_eci)
            nis_u += nis_val
            count_u += 1
            
        nis_ukf[k] = nis_u / max(1, count_u)

        # Statistics
        nees_mekf[k] = compute_nees(mekf.state[:4], mekf.state[4:], true_q, current_bias, mekf.P)
        nees_ukf[k] = compute_nees(ukf.state[:4], ukf.state[4:], true_q, current_bias, ukf.P)
        
        # MEKF Errors
        q_err_mekf = q_mult(q_inv(mekf.state[:4]), true_q)
        delta_theta_mekf = 4.0 * quat_to_mrp(q_err_mekf)
        delta_bias_mekf = current_bias - mekf.state[4:]
        err_mekf[k, :3] = delta_theta_mekf
        err_mekf[k, 3:] = delta_bias_mekf
        cov_mekf[k, :] = compute_3sigma(mekf.P)
        
        # UKF Errors
        q_err_ukf = q_mult(q_inv(ukf.state[:4]), true_q)
        delta_theta_ukf = 4.0 * quat_to_mrp(q_err_ukf)
        delta_bias_ukf = current_bias - ukf.state[4:]
        err_ukf[k, :3] = delta_theta_ukf
        err_ukf[k, 3:] = delta_bias_ukf
        
    return {
        'nees_mekf': nees_mekf,
        'nees_ukf': nees_ukf,
        'nis_mekf': nis_mekf,
        'nis_ukf': nis_ukf,
        'err_mekf': err_mekf,
        'err_ukf': err_ukf,
        'cov_mekf': cov_mekf
    }

def run_monte_carlo(config=None, show_plots=False, save_prefix="nominal"):
    if config is None:
        config = {}
        
    num_runs = config.get('num_runs', 50)
    init_err_deg = config.get('init_error_deg', 10.0)
    bias_scale = config.get('bias_scale', 1.0)
    eclipse_sim = config.get('eclipse_sim', False)
    
    dt = 0.1
    t_end = 300.0 
    steps = int(t_end / dt)
    time_arr = np.linspace(0, t_end, steps)
    
    print(f"Starting MC (Parallel): {save_prefix} | Runs: {num_runs} | Err: {init_err_deg} deg | Bias: {bias_scale}x")
    
    # Prepare Args
    worker_args = [(i, config, time_arr, dt, steps) for i in range(num_runs)]
    
    # Parallel Execution
    results = []
    # Use max cores
    cpu_count = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=cpu_count) as pool:
        for res in tqdm(pool.imap(run_worker, worker_args), total=num_runs):
            results.append(res)
            
    # Aggregation
    nees_mekf = np.array([r['nees_mekf'] for r in results])
    nees_ukf = np.array([r['nees_ukf'] for r in results])
    nis_mekf = np.array([r['nis_mekf'] for r in results])
    nis_ukf = np.array([r['nis_ukf'] for r in results])
    err_mekf = np.array([r['err_mekf'] for r in results])
    err_ukf = np.array([r['err_ukf'] for r in results])
    cov_mekf = np.array([r['cov_mekf'] for r in results])

    # Post Processing
    avg_nees_mekf = np.mean(nees_mekf, axis=0)
    avg_nees_ukf = np.mean(nees_ukf, axis=0)
    avg_nis_mekf = np.mean(nis_mekf, axis=0)
    avg_nis_ukf = np.mean(nis_ukf, axis=0)
    
    rms_err_mekf = np.sqrt(np.mean(err_mekf**2, axis=0))
    avg_sigma_mekf = np.mean(cov_mekf, axis=0)
    
    # Create Output Directory
    output_dir = os.path.join('figures', save_prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. NEES Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_arr, avg_nees_mekf, label='MEKF NEES')
    plt.plot(time_arr, avg_nees_ukf, label='UKF NEES', linestyle='--')
    plt.axhline(6.0, color='r', linestyle=':', label='Expected (6)')
    from scipy.stats import chi2
    dof = 6
    N = num_runs
    lower = chi2.ppf(0.025, N*dof) / N
    upper = chi2.ppf(0.975, N*dof) / N
    plt.axhline(lower, color='g', linestyle='--', alpha=0.5)
    plt.axhline(upper, color='g', linestyle='--', alpha=0.5)
    plt.title(f'NEES ({save_prefix})')
    plt.ylim(0, 20) 
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'nees.png'))
    plt.close()
    
    # 2. NIS Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_arr, avg_nis_mekf, label='MEKF NIS')
    plt.plot(time_arr, avg_nis_ukf, label='UKF NIS', linestyle='--')
    plt.axhline(3.0, color='r', linestyle=':', label='Expected (3)') 
    plt.title(f'NIS ({save_prefix})')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'nis.png'))
    plt.close()

    # 3. Consistency Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_arr, rms_err_mekf[:, 0], label='RMSE X')
    plt.plot(time_arr, avg_sigma_mekf[:, 0] / 3, 'r--', label='1-Sigma Cov')
    plt.title(f'Consistency X ({save_prefix})')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'consistency.png'))
    plt.close()
    
    # 4. Histograms (Final Error)
    final_err_mekf = np.linalg.norm(err_mekf[:, -1, :3], axis=1) 
    final_err_ukf = np.linalg.norm(err_ukf[:, -1, :3], axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(final_err_mekf, bins=10, alpha=0.5, label='MEKF Error')
    plt.hist(final_err_ukf, bins=10, alpha=0.5, label='UKF Error')
    plt.xlabel('Final Attitude Error (rad)')
    plt.ylabel('Count')
    plt.title(f'Final Attitude Error Histogram ({save_prefix})')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'hist.png'))
    plt.close()

    print(f"Finished {save_prefix}.")

if __name__ == '__main__':
    run_monte_carlo()
