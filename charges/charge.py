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

    def beta(self, time=None):
        time = time if time else self.times[-1]
        return self.velocity(time) / c

    def gamma(self, time=None):
        time = time if time else self.times[-1]
        beta_magnitude = np.linalg.norm(self.beta(time))
        if beta_magnitude > 1:
            raise ValueError("Velocity cannot greater than the speed of light.")
        return 1 / np.sqrt(1 - beta_magnitude ** 2)  # Calculate gamma

    def position(self, time=None):
        """
        :param time: If provided, interpolate to get the position at the specific time.
        :return: The position as a (1, 3) array [x, y, z].
        """
        if time is not None:
            return Utils.linear_interpolate(self.times, self.positions, time).flatten()
        return self.positions[-1]

    def velocity(self, time=None):
        """
        :param time: If provided, interpolate to get the velocity at the specific time.
        :return: The velocity as a (1, 3) array [vx, vy, vz].
        """
        if time is not None:
            return Utils.linear_interpolate(self.times, self.velocities, time).flatten()
        return self.velocities[-1]

    def acceleration(self, time=None):
        """
        :param time: If provided, interpolate to get the acceleration at the specific time.
        :return: The acceleration as a (1, 3) array [ax, ay, az].
        """
        if time is not None:
            return Utils.linear_interpolate(self.times, self.accelerations, time).flatten()
        return self.accelerations[-1]

    def density(self, x, y, z, time: float, sigma=0.01):
        """
        :param time: Time at which the density will be evaluated.
        :param sigma: The standard deviation of the Gaussian.
        :return: A 4D grid representing the charge density field in space-time.
        """
        density_array = self.magnitude * Utils.gaussian3D(x, y, z, sigma=sigma, mu=self.position(time))
        return density_array

    def current_density(self, x, y, z, time):
        """
        Compute the current density at a specific time and location.
        :param time: The time at which the current density is evaluated.
        :return: A 4D grid representing the current density field in space-time.
        """
        current_density_array = self.density(x, y, z, time) * self.velocity(time)
        return current_density_array





