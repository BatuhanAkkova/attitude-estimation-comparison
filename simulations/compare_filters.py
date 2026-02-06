import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import multiprocessing
import time
from tqdm import tqdm

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
from src.ckf import CKF
from src.sr_ukf import SRUKF
from src.aekf import AEKF
from src.analysis import compute_nees, compute_3sigma

def run_worker(args):
    run_idx, config, time_arr, dt, steps = args
    
    init_err_deg = config.get('init_error_deg', 10.0)
    bias_scale = config.get('bias_scale', 1.0)
    eclipse_sim = config.get('eclipse_sim', False)
    
    # Spacecraft Properties
    mass = 10.0; area = 0.1; Cd = 2.2; Cr = 1.8; I = np.diag([0.08, 0.08, 0.02])
    com_body = np.zeros(3); cp_body = np.array([0.05, 0.05, 0.0])
    
    # Initialize Storage
    filters_keys = ['mekf', 'ukf', 'ckf', 'sr_ukf', 'aekf']
    results = {k: {
        'nees': np.zeros(steps),
        'nis': np.zeros(steps),
        'err': np.zeros((steps, 6)),
        'total_time': 0.0
    } for k in filters_keys}
    
    # Random Conditions
    np.random.seed(os.getpid() + run_idx)
    RE = 6378.137; r_mag = RE + 500.0; v_mag = np.sqrt(398600.4418 / r_mag); inc_rad = np.radians(45.0)
    true_orbit_state = np.array([r_mag, 0., 0., 0., v_mag * np.cos(inc_rad), v_mag * np.sin(inc_rad)])
    
    axis = np.random.randn(3); axis /= np.linalg.norm(axis); angle = np.random.uniform(0, 2*np.pi)
    true_q = q_norm(np.array([axis[0]*np.sin(angle/2), axis[1]*np.sin(angle/2), axis[2]*np.sin(angle/2), np.cos(angle/2)]))
    true_omega = np.array([0.06, 0.02, -0.01]) + np.random.randn(3)*0.001
    true_dyn_state = np.concatenate([true_q, true_omega])
    init_true_bias = np.random.normal(0, 0.01 * bias_scale, 3) 
    
    # Sensors
    gyro = Gyroscope(initial_bias=init_true_bias, noise_std=0.0001, bias_walk_std=1e-6)
    mag_sensor = Magnetometer(noise_std=0.1e-6) 
    sun_sensor = SunSensor(noise_std=0.005) 
    
    # Estimators Init
    err_axis_est = np.random.randn(3); err_axis_est /= np.linalg.norm(err_axis_est); err_angle_est = np.radians(init_err_deg)
    dq_est = np.array([err_axis_est[0]*np.sin(err_angle_est/2), err_axis_est[1]*np.sin(err_angle_est/2), err_axis_est[2]*np.sin(err_angle_est/2), np.cos(err_angle_est/2)])
    init_q_est = q_mult(true_q, dq_est); init_bias_est = np.zeros(3); init_state_est = np.concatenate([init_q_est, init_bias_est])
    
    P0 = np.eye(6) * (0.1 if init_err_deg < 50 else 1.0) 
    Q = np.eye(6) * 1e-4; Q[3:, 3:] = np.eye(3) * 1e-8; R_generic = np.eye(3) * (0.01**2)
    
    filters = {
        'mekf': MEKF(init_state_est.copy(), P0.copy(), Q.copy(), R_generic.copy()),
        'ukf': UKF(init_state_est.copy(), P0.copy(), Q.copy(), R_generic.copy()),
        'ckf': CKF(init_state_est.copy(), P0.copy(), Q.copy(), R_generic.copy()),
        'sr_ukf': SRUKF(init_state_est.copy(), P0.copy(), Q.copy(), R_generic.copy()),
        'aekf': AEKF(init_state_est.copy(), P0.copy(), Q.copy(), R_generic.copy())
    }
    
    for k in range(steps):
        t = time_arr[k]; in_eclipse = eclipse_sim and (50.0 <= t <= 150.0)
        r_eci = true_orbit_state[:3]; v_eci = true_orbit_state[3:]; curr_q = true_dyn_state[:4]
        B_eci = get_true_mag_field(r_eci); S_eci = get_true_sun_vector(t)
        
        # Dynamics
        total_torque = gravity_gradient_torque(r_eci, I, curr_q) + aerodynamic_torque(r_eci, v_eci, curr_q, area, Cd, cp_body, com_body) + srp_torque(r_eci, S_eci, curr_q, area, Cr, cp_body, com_body)
        true_orbit_state = rk4_orbit_step(true_orbit_state, dt, disturbance_accel=drag_accel(r_eci, v_eci, mass, area, Cd) + srp_accel(r_eci, S_eci, mass, area, Cr))
        true_dyn_state = rk4_dynamics_step(true_dyn_state, dt, total_torque, I)
        true_q = true_dyn_state[:4]; true_omega_val = true_dyn_state[4:]; current_bias = gyro.get_bias()
        
        # Measurements
        dgm_true = q_to_dgm(true_q); B_body = dgm_true.T @ B_eci; S_body = dgm_true.T @ S_eci
        omega_meas = gyro.measure(true_omega_val, dt)
        B_meas_raw = mag_sensor.measure(B_body); B_norm = np.linalg.norm(B_meas_raw)
        B_meas_unit = B_meas_raw / B_norm if B_norm > 0 else np.array([1,0,0])
        B_ref_unit = B_eci / np.linalg.norm(B_eci) if np.linalg.norm(B_eci) > 0 else np.array([1,0,0])
        S_meas = sun_sensor.measure(S_body, in_eclipse=in_eclipse)
        
        # Filter Loops
        for f_name, f_obj in filters.items():
            t0 = time.perf_counter()
            f_obj.R = np.eye(3) * (0.01**2) # Reset R if needed (except for AEKF which adapts it)
            f_obj.predict(omega_meas, dt)
            nis_sum = f_obj.update(B_meas_unit, B_ref_unit); count = 1
            if S_meas is not None:
                f_obj.R = np.eye(3) * (sun_sensor.noise_std**2)
                nis_sum += f_obj.update(S_meas, S_eci); count += 1
            results[f_name]['total_time'] += (time.perf_counter() - t0)
            
            # Stats
            results[f_name]['nis'][k] = nis_sum / max(1, count)
            results[f_name]['nees'][k] = compute_nees(f_obj.state[:4], f_obj.state[4:], true_q, current_bias, f_obj.P)
            q_err = q_mult(q_inv(f_obj.state[:4]), true_q)
            results[f_name]['err'][k, :3] = 4.0 * quat_to_mrp(q_err) # delta_theta
            results[f_name]['err'][k, 3:] = current_bias - f_obj.state[4:] # delta_bias
            
    return results

def main():
    num_runs = 50; init_err_deg = 30.0; t_end = 200.0; dt = 0.1
    steps = int(t_end / dt); time_arr = np.linspace(0, t_end, steps)
    config = {'num_runs': num_runs, 'init_error_deg': init_err_deg, 'eclipse_sim': True}
    
    print(f"Comparing filters | Runs: {num_runs} | Init Err: {init_err_deg} deg")
    worker_args = [(i, config, time_arr, dt, steps) for i in range(num_runs)]
    
    runs_results = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for res in tqdm(pool.imap(run_worker, worker_args), total=num_runs):
            runs_results.append(res)
            
    # Aggregate
    f_keys = ['mekf', 'ukf', 'ckf', 'sr_ukf', 'aekf']
    agg = {k: {
        'avg_nees': np.zeros(steps), 'avg_nis': np.zeros(steps),
        'rmse_theta': np.zeros(steps), 'avg_time': 0.0
    } for k in f_keys}
    
    for k in f_keys:
        nees_arr = np.array([r[k]['nees'] for r in runs_results])
        nis_arr = np.array([r[k]['nis'] for r in runs_results])
        err_arr = np.array([r[k]['err'] for r in runs_results])
        times = np.array([r[k]['total_time'] for r in runs_results])
        
        agg[k]['avg_nees'] = np.mean(nees_arr, axis=0)
        agg[k]['avg_nis'] = np.mean(nis_arr, axis=0)
        agg[k]['rmse_theta'] = np.sqrt(np.mean(np.sum(err_arr[:, :, :3]**2, axis=2), axis=0))
        agg[k]['avg_time'] = np.mean(times)

    # Plotting
    out_dir = 'figures/comparison'
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. RMSE Attitude
    plt.figure(figsize=(10, 6))
    for k in f_keys:
        plt.plot(time_arr, np.degrees(agg[k]['rmse_theta']), label=k.upper())
    plt.yscale('log'); plt.xlabel('Time (s)'); plt.ylabel('Attitude RMSE (deg)'); plt.title('Attitude Estimation Accuracy Comparison'); plt.legend(); plt.grid(True)
    plt.savefig(f"{out_dir}/rmse_comparison.png"); plt.close()
    
    # 2. NEES
    plt.figure(figsize=(10, 6))
    for k in f_keys:
        plt.plot(time_arr, agg[k]['avg_nees'], label=k.upper())
    plt.axhline(6.0, color='r', linestyle=':'); plt.ylim(0, 15); plt.xlabel('Time (s)'); plt.ylabel('Avg NEES'); plt.title('Filter Consistency (NEES)'); plt.legend()
    plt.savefig(f"{out_dir}/nees_comparison.png"); plt.close()
    
    # 3. Bar Chart - Runtime
    plt.figure(figsize=(10, 6))
    times = [agg[k]['avg_time'] for k in f_keys]
    labels = [k.upper() for k in f_keys]
    plt.bar(labels, times, color=['blue', 'orange', 'green', 'red', 'purple'])
    plt.ylabel('Execution Time per Run (s)'); plt.title('Computational Efficiency Comparison')
    plt.savefig(f"{out_dir}/runtime_comparison.png"); plt.close()

    # Save Stats
    with open(f"{out_dir}/comparison_results.txt", 'w') as f:
        f.write("Filter Comparison Results\n" + "="*25 + "\n")
        for k in f_keys:
            f.write(f"{k.upper()}:\n")
            f.write(f"  Avg Time: {agg[k]['avg_time']:.4f} s\n")
            f.write(f"  Final RMSE (deg): {np.degrees(agg[k]['rmse_theta'][-1]):.6f}\n")
            f.write(f"  Avg NEES: {np.mean(agg[k]['avg_nees']):.4f}\n\n")
    
    print(f"Comparison complete. Results saved in {out_dir}/")

if __name__ == '__main__':
    main()
