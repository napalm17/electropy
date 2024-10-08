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
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def times(self) -> np.ndarray:
        return self._time_domain

    @property
    def velocities(self) -> np.ndarray:
        return np.gradient(self._positions, self._time_domain, axis=0)

    @property
    def accelerations(self) -> np.ndarray:
        return np.gradient(self.velocities, self._time_domain, axis=0)

    def set_free(self) -> FreeCharge:
        """
        Set this charge free at the end of its trajectory .
        :return: A FreeCharge instance with initial conditions set from the current state of the BoundCharge.
        """
        old_values = self.times, self.positions, self.velocities, self.accelerations
        return FreeCharge(mass=self.mass, magnitude=self.magnitude, old_values=old_values)
