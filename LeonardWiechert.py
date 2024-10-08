from charges import Charge
from scipy.constants import c
import numpy as np


class LW:
    def __init__(self, charge: Charge, position, time):
        self.charge = charge
        self.position = position
        self.time = time
        self.retarded_time = self.get_retarded_time()

    @property
    def beta(self):
        return self.charge.velocity(self.retarded_time) / c

    @property
    def gamma(self):
        return 1 / np.sqrt(1 - np.vdot(self.beta, self.beta))

    @property
    def direction(self):
        return self.position - self.charge.position(self.retarded_time)

    @property
    def unit_direction(self):
        if np.linalg.norm(self.direction) == 0:
            return self.direction
        return self.direction / np.linalg.norm(self.direction)

    def get_retarded_time(self, max_iter=100, tolerance=1e-9):
        t_retarded = self.time  # Start with the current observation time
        for i in range(max_iter):
            distance = np.linalg.norm(self.position - self.charge.position(t_retarded))
            new_t_retarded = self.time - distance / c
            if abs(new_t_retarded - t_retarded) < tolerance:
                break
            t_retarded = new_t_retarded
        return t_retarded

    def scalar_potential(self):
        delta_r = self.position - self.charge.position(self.retarded_time)
        relativistic_correction = 1 / (1 - np.vdot(self.charge.velocity(self.retarded_time), delta_r)/c**2)
        return self.charge.magnitude / np.linalg.norm(delta_r) * relativistic_correction

    def vector_potential(self):
        return self.scalar_potential() * self.charge.velocity(self.retarded_time)

    def electric_field(self):
        return self.e_field_velocity_component() + self.e_field_acceleration_component()

    def e_field_velocity_component(self):
        num = self.charge.magnitude * (self.unit_direction - self.beta)
        denom = self.gamma ** 2 * (1 - np.vdot(self.unit_direction, self.beta)) * np.linalg.norm(self.direction) ** 2
        return num / denom

    def e_field_acceleration_component(self):
        accel = np.cross(self.unit_direction - self.beta, self.charge.acceleration(self.retarded_time))
        num = self.charge.magnitude * np.cross(self.unit_direction, accel)
        denom = c * (1 - np.vdot(self.unit_direction, self.beta))**3 * np.linalg.norm(self.direction)
        return num / denom

    def magnetic_field(self):
        return np.cross(self.unit_direction / c, self.electric_field())