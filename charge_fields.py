from charges import Charge
from scipy.constants import c
import numpy as np


class ChargeFields:
    def __init__(self, charge: Charge, position: np.ndarray, time: float):
        self.charge = charge
        self.position = position  # Positions can now be a grid (n, 3) where n is the number of positions.
        self.time = time
        self.retarded_time = self.get_retarded_time()

    @property
    def beta(self) -> np.ndarray:
        return self.charge.velocity(self.retarded_time) / c

    @property
    def gamma(self) -> np.ndarray:
        return 1 / np.sqrt(1 - np.linalg.norm(self.beta, axis=-1) ** 2)

    @property
    def direction(self) -> np.ndarray:
        return self.position - self.charge.position(self.retarded_time)  # Broadcasting to handle array of positions.

    @property
    def unit_direction(self) -> np.ndarray:
        norm = np.linalg.norm(self.direction, axis=-1, keepdims=True)  # Compute norm along the last axis for each position.
        norm[norm == 0] = 1  # Prevent division by zero.
        return self.direction / norm  # Element-wise division to get unit vectors for each position.

    def get_retarded_time(self, max_iter=100, tolerance=1e-9) -> np.ndarray:
        t_retarded = np.full(self.position.shape[:-1], self.time)  # Start with the current observation time
        for i in range(max_iter):
            # Compute distances for each position from charge's retarded position using broadcasting
            distance = np.linalg.norm(self.position - self.charge.position(t_retarded), axis=-1)
            new_t_retarded = self.time - distance / c
            if np.all(np.abs(new_t_retarded - t_retarded) < tolerance):  # Check if all differences converge
                break
            t_retarded = new_t_retarded
        return t_retarded

    def scalar_potential(self) -> np.ndarray:
        relativistic_correction = 1 / (1 - np.sum(self.charge.velocity(self.retarded_time) * self.direction, axis=-1)/c**2)
        return self.charge.magnitude / np.linalg.norm(self.direction, axis=-1) * relativistic_correction

    def vector_potential(self) -> np.ndarray:
        return np.atleast_1d(self.scalar_potential())[:, np.newaxis] * self.charge.velocity(self.retarded_time)  # Broadcasting to match dimensions.

    def electric_field(self) -> np.ndarray:
        return self.e_field_velocity_component() + self.e_field_acceleration_component()

    def e_field_velocity_component(self) -> np.ndarray:
        num = self.charge.magnitude * (self.unit_direction - self.beta)
        denom = self.gamma ** 2 * (1 - np.sum(self.unit_direction * self.beta, axis=-1)) * np.linalg.norm(self.direction, axis=-1) ** 2
        print(denom, num.shape, 'test')
        return num / np.atleast_1d(denom)[:, np.newaxis]

    def e_field_acceleration_component(self) -> np.ndarray:
        accel = np.cross(self.unit_direction - self.beta, self.charge.acceleration(self.retarded_time))
        num = self.charge.magnitude * np.cross(self.unit_direction, accel)
        denom = c * (1 - np.sum(self.unit_direction * self.beta, axis=-1)) ** 3 * np.linalg.norm(self.direction, axis=-1)
        return num / np.atleast_1d(denom)[:, np.newaxis]

    def magnetic_field(self) -> np.ndarray:
        return np.cross(self.unit_direction / c, self.electric_field(), axis=-1)  # Cross product for arrays
