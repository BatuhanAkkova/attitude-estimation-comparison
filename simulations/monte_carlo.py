import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import multiprocessing
import time
from tqdm import tqdm
from scipy.stats import norm

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import q_norm, q_mult, q_inv, q_to_dgm, quat_to_mrp 
from src.dynamics import rk4_step, rk4_dynamics_step
from src.orbit import rk4_orbit_step, gravity_gradient_torque, drag_accel, srp_accel, aerodynamic_torque, srp_torque
from src.measurements import get_true_mag_field, get_true_sun_vector
from src.sensors.gyro import Gyroscope
from src.sensors.magnetometer import Magnetometer
from src.sensors.sun_sensor import SunSensor

from src.mekf import MEKF
from src.ukf import UKF
from src.aekf import AEKF
from src.analysis import (compute_nees, compute_angle_error, 
                          compute_consistency_bounds, 
                          check_numerical_stability, compute_normalized_errors)

def run_worker(args):
    """
    Worker function for a single Monte Carlo run.
    """
    run_idx, config, time_arr, dt, steps = args
    
    init_err_deg = config.get('init_error_deg', 10.0)
    bias_scale = config.get('bias_scale', 1.0)
    eclipse_sim = config.get('eclipse_sim', False)
    theta_max = np.radians(config.get('theta_max_deg', 45.0))
    
    # Spacecraft Properties
    mass = 10.0 # kg
    area = 0.1 # m^2
    Cd = 2.2
    Cr = 1.8
    I = np.diag([0.08, 0.08, 0.02]) # kg m^2
    com_body = np.zeros(3)
    cp_body = np.array([0.05, 0.05, 0.0])
    
    # Initialize Storage for single run
    filters = ['mekf', 'ukf', 'aekf']
    nees = {f: np.zeros(steps) for f in filters}
    nis = {f: np.zeros(steps) for f in filters}
    angle_err = {f: np.zeros(steps) for f in filters}
    err = {f: np.zeros((steps, 6)) for f in filters}
    norm_err = {f: np.zeros((steps, 6)) for f in filters}
    stability = {f: {'cond': np.zeros(steps), 'pos_def': np.ones(steps, dtype=bool)} for f in filters}
    
    divergence_time = {f: np.nan for f in filters}
    diverged = {f: False for f in filters}
    
    total_time = {f: 0.0 for f in filters}
    
    # Random Initial Conditions
    RE = 6378.137
    r_mag = RE + 500.0
    v_mag = np.sqrt(398600.4418 / r_mag)
    inc_rad = np.radians(45.0)
    true_orbit_state = np.array([r_mag, 0., 0., 0., v_mag * np.cos(inc_rad), v_mag * np.sin(inc_rad)])
    
    # Random Initial Quaternion
    np.random.seed(os.getpid() + run_idx) 
    
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(0, 2*np.pi)
    true_q = q_norm(np.array([axis[0]*np.sin(angle/2), axis[1]*np.sin(angle/2), axis[2]*np.sin(angle/2), np.cos(angle/2)]))
    
    # Random Initial Omega
    true_omega = np.array([0.06, 0.02, -0.01]) + np.random.randn(3)*0.001
    true_dyn_state = np.concatenate([true_q, true_omega])
    
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
    
    filter_objs = {
        'mekf': MEKF(init_state_est.copy(), P0.copy(), Q.copy(), R_generic.copy()),
        'ukf': UKF(init_state_est.copy(), P0.copy(), Q.copy(), R_generic.copy()),
        'aekf': AEKF(init_state_est.copy(), P0.copy(), Q.copy(), R_generic.copy())
    }
    
    for k in range(steps):
        t = time_arr[k]
        in_eclipse = False
        if eclipse_sim and 50.0 <= t <= 150.0:
            in_eclipse = True
        
        # Current States
        r_eci = true_orbit_state[:3]
        v_eci = true_orbit_state[3:]
        curr_q = true_dyn_state[:4]
        
        # Calculate Disturbances
        B_eci = get_true_mag_field(r_eci)
        S_eci = get_true_sun_vector(t)
        
        # Forces
        a_drag = drag_accel(r_eci, v_eci, mass, area, Cd)
        a_srp = srp_accel(r_eci, S_eci, mass, area, Cr)
        total_dist_accel = a_drag + a_srp
        
        # Torques
        tau_gg = gravity_gradient_torque(r_eci, I, curr_q)
        tau_aero = aerodynamic_torque(r_eci, v_eci, curr_q, area, Cd, cp_body, com_body)
        tau_srp = srp_torque(r_eci, S_eci, curr_q, area, Cr, cp_body, com_body)
        total_torque = tau_gg + tau_aero + tau_srp
        
        # Dynamics
        true_orbit_state = rk4_orbit_step(true_orbit_state, dt, drag_accel(r_eci, v_eci, mass, area, Cd) + srp_accel(r_eci, S_eci, mass, area, Cr))
        true_dyn_state = rk4_dynamics_step(true_dyn_state, dt, total_torque, I)
        true_q = true_dyn_state[:4]
        true_omega_val = true_dyn_state[4:]
        current_bias = gyro.get_bias()
        
        # Transform for Sensors
        dgm_true = q_to_dgm(true_q)
        B_body = dgm_true.T @ B_eci
        S_body = dgm_true.T @ S_eci
        
        # Measurements
        omega_meas = gyro.measure(true_omega_val, dt)
        B_meas_raw = mag_sensor.measure(B_body)
        B_norm = np.linalg.norm(B_meas_raw)
        B_meas_unit = B_meas_raw / B_norm if B_norm > 0 else np.array([1,0,0])
        B_ref_unit = B_eci / np.linalg.norm(B_eci) if np.linalg.norm(B_eci) > 0 else np.array([1,0,0])
        S_meas = sun_sensor.measure(S_body, in_eclipse=in_eclipse)
        
        # Run all filters
        for f in filters:
            obj = filter_objs[f]
            t_start = time.perf_counter()
            
            # Prediction
            obj.predict(omega_meas, dt)
            
            # Update Mag (Relative noise ~ 0.1uT / 50uT = 0.002)
            obj.R = np.eye(3) * (0.002**2)
            nis_m = obj.update(B_meas_unit, B_ref_unit)
            count = 1
            
            # Update Sun
            if S_meas is not None:
                obj.R = np.eye(3) * (sun_sensor.noise_std**2)
                nis_m += obj.update(S_meas, S_eci)
                count += 1
            
            t_end = time.perf_counter()
            total_time[f] += (t_end - t_start)
            nis[f][k] = nis_m / count

            # Metrics
            est_q = obj.state[:4]
            est_bias = obj.state[4:]
            curr_P = obj.P
            
            angle_err[f][k] = compute_angle_error(true_q, est_q)
            nees[f][k] = compute_nees(est_q, est_bias, true_q, current_bias, curr_P)
            
            q_err = q_mult(q_inv(est_q), true_q)
            err[f][k, :3] = 4.0 * quat_to_mrp(q_err)
            err[f][k, 3:] = current_bias - est_bias
            
            norm_err[f][k] = compute_normalized_errors(est_q, est_bias, true_q, current_bias, curr_P)
            
            stability[f]['cond'][k], stability[f]['pos_def'][k] = check_numerical_stability(curr_P)
            
            if not diverged[f] and angle_err[f][k] > theta_max:
                diverged[f] = True
                divergence_time[f] = t
        
    return {
        'nees': nees,
        'nis': nis,
        'err': err,
        'angle_err': angle_err,
        'norm_err': norm_err,
        'stability': stability,
        'divergence_time': divergence_time,
        'diverged': diverged,
        'total_time': total_time
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
    
    print(f"Starting MC: {save_prefix} | Runs: {num_runs} | Err: {init_err_deg} deg | Bias: {bias_scale}x | Eclipse: {eclipse_sim}")
    
    worker_args = [(i, config, time_arr, dt, steps) for i in range(num_runs)]
    results = []
    cpu_count = min(multiprocessing.cpu_count(), 4)
    with multiprocessing.Pool(processes=cpu_count) as pool:
        for res in tqdm(pool.imap(run_worker, worker_args), total=num_runs):
            results.append(res)
            
    filters = ['mekf', 'ukf', 'aekf']
    
    # Create Output Directory
    output_dir = os.path.join('figures', save_prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. RMSE vs Time (Representation Invariant Angle Error)
    plt.figure(figsize=(10, 6))
    for f in filters:
        all_angle_errs = np.array([r['angle_err'][f] for r in results])
        rmse = np.sqrt(np.mean(all_angle_errs**2, axis=0))
        std = np.std(all_angle_errs, axis=0)
        plt.plot(time_arr, np.degrees(rmse), label=f'{f.upper()}')
        plt.fill_between(time_arr, np.degrees(rmse - 3*std), np.degrees(rmse + 3*std), alpha=0.1)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle Error RMSE (deg)')
    plt.title('Attitude Accuracy (Invariant Error)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'rmse_time.png'))
    plt.close()

    # 2. NEES and NIS with Bounds
    # Note: NIS for unit vector measurements has 2 DOF
    for metric_name, dof in [('nees', 6), ('nis', 1)]:
        plt.figure(figsize=(10, 6))
        lower, upper = compute_consistency_bounds(dof, num_runs)
        for f in filters:
            avg_metric = np.mean([r[metric_name][f] for r in results], axis=0)
            plt.plot(time_arr, avg_metric, label=f'{f.upper()}')
        plt.axhline(dof, color='r', linestyle='--', label='Expected')
        plt.axhline(lower, color='g', linestyle=':', label='95% Lower')
        plt.axhline(upper, color='g', linestyle=':', label='95% Upper')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Mean {metric_name.upper()}')
        plt.title(f'Consistency: {metric_name.upper()}')
        plt.legend()
        plt.ylim(0, dof * 3)
        plt.savefig(os.path.join(output_dir, f'{metric_name}.png'))
        plt.close()

    # 3. Final Error Histogram with Gaussian Fit
    plt.figure(figsize=(10, 6))
    for f in filters:
        final_errs = np.degrees(np.array([r['angle_err'][f][-1] for r in results]))
        mu, std = norm.fit(final_errs)
        plt.hist(final_errs, bins=15, density=True, alpha=0.3, label=f'{f.upper()}')
        x = np.linspace(min(final_errs), max(final_errs), 100)
        plt.plot(x, norm.pdf(x, mu, std), linestyle='--')
    plt.xlabel('Final Angle Error (deg)')
    plt.ylabel('Density')
    plt.title('Final Error Distribution')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'hist_final.png'))
    plt.close()

    # 4. Normalized Error & Q-Q Plot (for MEKF as example)
    f_ref = 'mekf'
    norm_errs_final = np.array([r['norm_err'][f_ref][-1] for r in results])
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(norm_errs_final[:, 0], bins=15, density=True, alpha=0.6)
    x = np.linspace(-4, 4, 100)
    plt.plot(x, norm.pdf(x, 0, 1), 'r--')
    plt.title(f'Normalized Error Hist ({f_ref.upper()}, Comp 1)')
    
    plt.subplot(1, 2, 2)
    from scipy.stats import probplot
    probplot(norm_errs_final[:, 0], dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.savefig(os.path.join(output_dir, 'normalized_qq.png'))
    plt.close()

    # Statistics reporting
    avg_times = {f: np.mean([r['total_time'][f] for r in results]) for f in filters}
    
    summary_data = []
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f_out:
        f_out.write(f"Monte Carlo Results: {save_prefix}\n")
        f_out.write(f"Config: InitErr={init_err_deg} deg, BiasScale={bias_scale}x, Eclipse={eclipse_sim}\n")
        f_out.write("="*40 + "\n")
        for f in filters:
            all_final_errs = np.array([r['angle_err'][f][-1] for r in results])
            final_rmse = np.sqrt(np.mean(all_final_errs**2))
            div_count = sum(r['diverged'][f] for r in results)
            div_pct = (div_count / num_runs) * 100
            
            f_out.write(f"{f.upper():<10}: RMSE={np.degrees(final_rmse):.4f} deg | Diverged={div_pct:.1f}% | Time={avg_times[f]:.4f}s\n")
            
            summary_data.append({
                'filter': f.upper(),
                'rmse_deg': np.degrees(final_rmse),
                'diverged_pct': div_pct,
                'avg_runtime_s': avg_times[f],
                'scenario': save_prefix
            })

if __name__ == '__main__':
    run_monte_carlo(config={'num_runs': 20, 'init_error_deg': 20.0})

