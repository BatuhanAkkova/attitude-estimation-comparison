import numpy as np
import sys
import os

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from simulations.monte_carlo import run_monte_carlo

def main():
    print("=== Running Scenario Suite ===")
    
    # 1. Nominal
    print("\n[Scenario A] Nominal")
    run_monte_carlo({
        'num_runs': 50, 
        'init_error_deg': 10.0,
        'bias_scale': 1.0,
        'eclipse_sim': False
    }, save_prefix="scen_A_nominal")
    
    # 2. Large Initial Error
    print("\n[Scenario B] Large Initial Error (120 deg)")
    run_monte_carlo({
        'num_runs': 50,
        'init_error_deg': 120.0,
        'bias_scale': 1.0,
        'eclipse_sim': False
    }, save_prefix="scen_B_large_err")
    
    # 3. High Bias
    print("\n[Scenario C] High Bias (10x)")
    run_monte_carlo({
        'num_runs': 50, 
        'init_error_deg': 10.0,
        'bias_scale': 10.0,
        'eclipse_sim': False
    }, save_prefix="scen_C_high_bias")
    
    # 4. Eclipse
    print("\n[Scenario D] Sensor Dropout (Eclipse)")
    run_monte_carlo({
        'num_runs': 50,
        'init_error_deg': 10.0,
        'bias_scale': 1.0,
        'eclipse_sim': True
    }, save_prefix="scen_D_eclipse")
    
    print("\nAll scenarios complete. Check 'figures/' for plots.")

if __name__ == '__main__':
    main()
