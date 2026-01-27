import numpy as np

class Gyroscope:
    def __init__(self, initial_bias=None, noise_std=0.001, bias_walk_std=1e-5):
        """
        Rate Integrating Gyroscope Model.
        
        Args:
            initial_bias (np.array): Initial bias vector [rad/s]. Defaults to zeros.
            noise_std (float): Standard deviation of White Noise (Angle Random Walk variant) [rad/s].
            bias_walk_std (float): Standard deviation of Bias Random Walk [rad/s/sqrt(s)].
        """
        self.bias = np.array(initial_bias) if initial_bias is not None else np.zeros(3)
        self.noise_std = noise_std
        self.bias_walk_std = bias_walk_std
        
    def measure(self, true_omega, dt):
        """
        Generates a noisy gyroscope measurement.
        
        Args:
            true_omega (np.array): True angular velocity [rad/s].
            dt (float): Time step since last update [s].
            
        Returns:
            np.array: Measured angular velocity [rad/s].
        """
        # Update Bias (Random Walk)
        # Random walk variance grows with time: sigma_d = sigma_c * sqrt(dt)
        if dt > 0:
            bias_noise = np.random.normal(0, self.bias_walk_std, 3) * np.sqrt(dt)
            self.bias += bias_noise
            
        # Measurement Noise (White Noise)
        meas_noise = np.random.normal(0, self.noise_std, 3)
        
        # Total Measurement
        return true_omega + self.bias + meas_noise
    
    def get_bias(self):
        return self.bias.copy()
