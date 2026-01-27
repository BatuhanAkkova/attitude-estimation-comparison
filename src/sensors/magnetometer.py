import numpy as np

class Magnetometer:
    def __init__(self, noise_std=0.01):
        """
        Magnetometer Model.
        
        Args:
            noise_std (float): Standard deviation of measurement noise [Tesla] or normalized units.
        """
        self.noise_std = noise_std
        
    def measure(self, true_mag_field):
        """
        Generates a noisy magnetometer measurement.
        
        Args:
            true_mag_field (np.array): True magnetic field vector in body frame.
            
        Returns:
            np.array: Measured magnetic field vector.
        """
        noise = np.random.normal(0, self.noise_std, 3)
        return true_mag_field + noise
