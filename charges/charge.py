from abc import ABC, abstractmethod
import numpy as np
from helpers import Utils
from scipy.constants import c


class Charge(ABC):
    def __init__(self, mass=1.0, magnitude=1.0):
        """
        Initialize a general charge.
        """
        self.mass = mass
        self.magnitude = magnitude

    @property
    @abstractmethod
    def positions(self):
        pass

    @property
    @abstractmethod
    def velocities(self):
        pass

    @property
    @abstractmethod
    def accelerations(self):
        pass

    @property
    @abstractmethod
    def times(self):
        pass

    def beta(self, time=None) -> np.ndarray:
        time = time if time else self.times[-1]
        return self.velocity(time) / c

    def gamma(self, time=None) -> float:
        time = time if time else self.times[-1]
        beta_magnitude = np.linalg.norm(self.beta(time))
        if beta_magnitude > 1:
            raise ValueError("Velocity cannot greater than the speed of light.")
        return 1 / np.sqrt(1 - beta_magnitude ** 2)  # Calculate gamma

    def position(self, time=None) -> np.ndarray:
        if time is not None:
            return Utils.linear_interpolate(self.times, self.positions, time).T
        return self.positions[-1]

    def velocity(self, time=None) -> np.ndarray:
        if time is not None:
            return Utils.linear_interpolate(self.times, self.velocities, time).T
        return self.velocities[-1]

    def acceleration(self, time=None) -> np.ndarray:
        if time is not None:
            return Utils.linear_interpolate(self.times, self.accelerations, time).T
        return self.accelerations[-1]

    def density(self, x, y, z, time: float, sigma=0.01) -> np.ndarray:
        density_array = self.magnitude * Utils.gaussian3D(x, y, z, sigma=sigma, mu=self.position(time))
        return density_array

    def current_density(self, x, y, z, time) -> np.ndarray:
        current_density_array = self.density(x, y, z, time) * self.velocity(time)
        return current_density_array





