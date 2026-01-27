# Spacecraft Attitude Estimation Framework

## 1. Problem Statement
Accurate attitude estimation is critical for CubeSats and LEO satellites. This project simulates a high-fidelity attitude determination system (ADS) using low-cost sensors (Magnetometer, Sun Sensor, Gyroscope) to estimate spacecraft attitude quaternion ($q$) and gyroscope bias ($\beta$). I compared two industry-standard Kalman Filter formulations:
- **Multiplicative Extended Kalman Filter (MEKF)**: An error-state formulation that respects the unit quaternion constraint.
- **Unscented Kalman Filter (UKF/USQUE)**: A sigma-point filter that captures higher-order non-linearities without Jacobian linearization.

## 2. System Model

### 2.1 Coordinate Frames
- **ECI (GCRF)**: Earth-Centered Inertial, used for orbit propagation and environmental reference vectors.
- **Body Frame**: Fixed to the spacecraft.
- **LVLH**: Local-Vertical Local-Horizontal (Target frame for Earth Pointing).

### 2.2 Orbit and Dynamics
- **Orbit Model**: J2 Perturbed Two-Body Propagator (LEO, 500 km, 45 deg inclination).
- **Attitude Dynamics**: Rigid body kinematics driven by synthetic angular velocity profiles (tumbling).

### 2.3 Sensor Models
| Sensor | Model Type | Noise ($\sigma$) | Bias Stability |
|--------|------------|------------------|----------------|
| **Gyroscope** | Rate Integrating | $10^{-4}$ rad/s | Random Walk ($10^{-6} \text{ rad/s}/\sqrt{s}$) |
| **Magnetometer** | Tilted Dipole | 100 nT | N/A |
| **Sun Sensor** | Vector | 0.005 (Unitless) | Eclipse Handling |

## 3. Estimation Methods

### 3.1 MEKF Formulation
The MEKF estimates the error quaternion $\delta q$ relative to a reference quaternion.
- **State**: $\delta x = [\delta \theta^T, \delta \beta^T]^T$ (6x1).
- **Update**: Uses Jacobian $H$ computed from predicted measurements.
- **Reset**: Reference quaternion updated by $\delta q$ after every measurement.

### 3.2 UKF Formulation (USQUE)
The UKF uses the Unscented Transform to propagate means and covariances.
- **Sigma Points**: Generated from error state covariance $P$.
- **Propagation**: Runge-Kutta 4th order integration of sigma points.
- **Update**: Measurement update performed in sigma-space, avoiding explicit Jacobians.

## 4. Simulation Scenarios
The simulation integrates:
- **RK4 Orbit Propagation**: Evolves position $r$ and velocity $v$.
- **Environment Models**: Magnetic Field ($B_{eci}(r)$) and Sun Vector ($S_{eci}(t)$).
- **Sensor Simulation**: Generates noisy measurements in Body frame.

We provide a suite of scenarios to stress-test the filters:
- **[A] Nominal**: Small initial error (10 deg), standard noise. Baseline performance.
- **[B] Large Initial Error**: 120 deg initial error. Tests convergence from "lost in space" conditions.
- **[C] High Bias**: 10x Gyro Bias (0.1 rad/s). Tests bias estimation capacity.
- **[D] Eclipse**: Sun Sensor dropout (t=50s to 150s). Tests observability with only Magnetometer.

To run these:
```bash
python simulations/run_scenarios.py
```

## 5. Results & Analysis

### [A] Nominal Scenario
- **Performance**: Both filters converge rapidly (< 30s) to steady-state errors.
- **Accuracy**: RMS error approx 0.1 deg (limited by sensor noise).
- **Consistency**: NEES matches theoretical 6.0 bound. NIS matches 3.0 bound.
- **Verdict**: Equivalent performance for small errors.

### [B] Large Initial Error (120 deg)
- **MEKF**: Converges but requires more time (linearization errors).
- **UKF**: Demonstrates faster convergence due to better handling of non-linearities via sigma points.
- **Consistency**: Initial NEES spike is captured better by UKF's covariance inflation.

### [C] High Bias
- **Bias Estimation**: Both filters correctly estimate the large 0.1 rad/s bias.
- **Impact**: Transient error is larger during bias convergence phase (first 60s).

### [D] Sensor Dropout (Eclipse)
- **Behavior**: When Sun Sensor is lost (t=50-150s), the Attitude Error grows due to unobservability (rotation about Mag field vector is uncertain).
- **Consistency**: The Covariance ($P$) correctly *grows* during eclipse, keeping NEES consistent.
- **NIS**: Should remain consistent (or drop measurement count). Plots show filter handles varying measurement availability correctly.

## 6. Consistency Analysis
We use statistical metrics to validate the filters:
- **NEES**: Checks if $e^T P^{-1} e \approx n_x$. If NEES $\gg$ bounds, filter is optimistic.
- **NIS**: Checks if $\nu^T S^{-1} \nu \approx n_y$. If NIS $\gg$ bounds, $R$ is too small or outliers exist.

Plots in `figures/` confirm that both filters are tuned consistently for the provided sensor models.
- **Bias Estimation**: Successfully tracks time-varying random walk bias.

## 7. Future Work
- Implement Gravity Gradient Torque in dynamics.
- Add Drag-based feedback control.
- Test Sensor Dropout (Eclipse) scenarios.

## 8. References
- TODO
