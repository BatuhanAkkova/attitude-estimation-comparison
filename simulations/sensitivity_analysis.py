import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from simulations.monte_carlo import run_worker
import multiprocessing
from tqdm import tqdm

def run_sweep(param_name, values, num_runs=4):
    """
    Generalized sweep function.
    """
    dt = 0.1
    t_end = 200.0
    steps = int(t_end / dt)
    time_arr = np.linspace(0, t_end, steps)
    
    filters = ['mekf', 'ukf', 'aekf']
    results_sweep = {f: [] for f in filters}
    
    print(f"\nStarting Sensitivity Analysis: Sweep over {param_name}")
    
    for val in values:
        print(f"Running for {param_name}: {val}")
        config = {'num_runs': num_runs, param_name: val}
        worker_args = [(i, config, time_arr, dt, steps) for i in range(num_runs)]
        
        run_results = []
        cpu_count = min(multiprocessing.cpu_count(), 4)
        with multiprocessing.Pool(processes=cpu_count) as pool:
            for res in tqdm(pool.imap(run_worker, worker_args), total=num_runs):
                run_results.append(res)
        
        for f in filters:
            # Final RMSE for this value
            final_rmse = np.sqrt(np.mean(np.array([r['angle_err'][f][-1]**2 for r in run_results])))
            results_sweep[f].append(np.degrees(final_rmse))
            
    # Plotting
    plt.figure(figsize=(10, 6))
    for f in filters:
        plt.plot(values, results_sweep[f], marker='o', label=f.upper())
    
    plt.xlabel(f'{param_name} value')
    plt.ylabel('Final RMSE (deg)')
    plt.title(f'Sensitivity Analysis: Impact of {param_name}')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    output_dir = 'figures/sensitivity'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{param_name}_sensitivity.png'))
    print(f"Sensitivity plots saved in {output_dir}")

def run_sensitivity_analysis():
    # 1. Initial Error Sweep
    init_errors = [5.0, 10.0, 20.0, 45.0, 90.0]
    run_sweep('init_error_deg', init_errors)
    
    # 2. Bias Scale Sweep
    bias_scales = [1.0, 2.0, 5.0, 10.0, 20.0]
    run_sweep('bias_scale', bias_scales)

if __name__ == '__main__':
    run_sensitivity_analysis()
