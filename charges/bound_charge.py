from charges.charge import Charge
import numpy as np
from charges.free_charge import FreeCharge


class BoundCharge(Charge):
    def __init__(self, x_domain, y_domain, z_domain, time_domain, magnitude: float, mass: float= 1):
        """
        Initialize a bound point charge.
        """
        super().__init__(mass=mass, magnitude=magnitude)
        self._positions = np.stack([x_domain, y_domain, z_domain], axis=1)
        self._time_domain = time_domain


    @property
    def positions(self):
        return self._positions

    @property
    def times(self):
        return self._time_domain

    @property
    def velocities(self):
        """
        :return: 3xN array of velocity values of the charge across time.
        """
        return np.gradient(self._positions, self._time_domain, axis=0)

    @property
    def accelerations(self):
        """
        :return: 3xN array of acceleration values of the charge across time.
        """
        return np.gradient(self.velocities, self._time_domain, axis=0)

    def set_free(self, time=None):
        """
        Switch this charge to a free-moving charge at a specific time.
        :param time: The time at which to switch.
        :return: A FreeCharge instance with initial conditions set from the current state of the BoundCharge.
        """
        time = time if time else self.times[-1]
        return FreeCharge(self.position(time), self.velocity(time), 0, self.mass, self.magnitude)
