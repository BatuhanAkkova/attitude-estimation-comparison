import numpy as np
import sys
import os

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from simulations.monte_carlo import run_monte_carlo
from simulations.sensitivity_analysis import run_sensitivity_analysis

def main():
    print("=== Running Scenario Suite ===")
    
    # 1. Sensitivity Analysis (Sweeps - find breakpoints)
    print("\n[Scenario 1] Sensitivity Analysis")
    run_sensitivity_analysis()
    
    # 2. Anchor Point: Nominal (Consistency Check)
    print("\n[Scenario 2] Anchor: Nominal Performance")
    run_monte_carlo({
        'num_runs': 20,
        'init_error_deg': 10.0,
        'bias_scale': 1.0,
        'eclipse_sim': False
    }, save_prefix="anchor_nominal")

    # 3. Anchor Point: High Tumble (Non-linearity Check)
    print("\n[Scenario 3] Anchor: High Tumble (Large Init Error)")
    run_monte_carlo({
        'num_runs': 20,
        'init_error_deg': 90.0,
        'bias_scale': 1.0,
        'eclipse_sim': False
    }, save_prefix="anchor_high_tumble")

    # 4. Anchor Point: Stress Test (High Bias)
    print("\n[Scenario 4] Anchor: Stress Test (Degraded Gyros)")
    run_monte_carlo({
        'num_runs': 20,
        'init_error_deg': 10.0,
        'bias_scale': 15.0,
        'eclipse_sim': False
    }, save_prefix="anchor_high_bias")
    
    # 5. Eclipse (Time-dependent behavior)
    print("\n[Scenario 5] Sensor Dropout (Eclipse)")
    run_monte_carlo({
        'num_runs': 20,
        'init_error_deg': 10.0,
        'bias_scale': 1.0,
        'eclipse_sim': True
    }, save_prefix="scen_eclipse")
    
    print("\nAll scenarios complete.")

if __name__ == '__main__':
    main()
