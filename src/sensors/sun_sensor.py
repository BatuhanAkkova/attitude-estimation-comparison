import numpy as np

class SunSensor:
    def __init__(self, noise_std=0.005, fov_deg=90.0):
        """
        Sun Sensor Model.
        
        Args:
            noise_std (float): Standard deviation of measurement noise (direction vector).
            fov_deg (float): Field of View (not used for now)
        """
        self.noise_std = noise_std
        
    def measure(self, true_sun_vec, in_eclipse=False):
        """
        Generates a noisy sun vector measurement.
        
        Args:
            true_sun_vec (np.array): True unit vector to Sun in BODY frame.
            in_eclipse (bool): If True, sensor returns None.
            
        Returns:
            np.array or None: Measured sun vector or None if in eclipse.
        """
        if in_eclipse:
            return None

        noise = np.random.normal(0, self.noise_std, 3)
        measured_vec = true_sun_vec + noise
        
        # Renormalize
        norm = np.linalg.norm(measured_vec)
        if norm < 1e-6:
            return np.array([0., 0., 0.])
            
        return measured_vec / norm
