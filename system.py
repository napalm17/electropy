import numpy as np
from charges import FreeCharge, BoundCharge
from LeonardWiechert import LW
from scipy.constants import epsilon_0, mu_0, c
from numpy.linalg import norm
from helpers import Utils


class System:

    def __init__(self, ranges=None, charges=None, e_field=Utils.zero_field(3), b_field=Utils.zero_field(3), init_time=0):
        self.charges: list[BoundCharge or FreeCharge] = [] if charges is None else charges
        self.external_fields = {'electric_field': e_field, 'magnetic_field': b_field}
        self.time: float = init_time
        self.ranges = ranges
        print(self.time)

    @property
    def free_charges(self):
        return [charge for charge in self.charges if isinstance(charge, FreeCharge)]

    @property
    def grid(self):
        xx, yy, zz = np.meshgrid(self.ranges[0], self.ranges[1], self.ranges[2], indexing='ij')
        return np.stack([xx, yy, zz], axis=-1)

    @property
    def charge_positions(self):
        return np.array([charge.position(self.time) for charge in self.charges])

    @property
    def charge_velocities(self):
        return np.array([charge.velocity(self.time) for charge in self.charges])

    @staticmethod
    def is_repelling(charge1, charge2):
        return charge1.magnitude * charge2.magnitude > 0

    @staticmethod
    def are_close(charge1, charge2, tolerance=1e-1):
        return np.all(np.isclose(charge1.position(), charge2.position(), atol=tolerance))

    def add_charge(self, charge):
        """Adds a particle to the system."""
        self.charges.append(charge)

    def add_field(self, func, field_type):
        old_field = self.external_fields[field_type]
        self.external_fields[field_type] = lambda pos, t: old_field(pos, t) + func(pos, t)

    def get_charge_density(self, position, time):
        """Apply the superposition principle to obtain the total charge density."""
        x, y, z = position
        return np.sum([charge.density(x, y, z, time) for charge in self.charges])

    def get_current_density(self, position, time):
        x, y, z = position
        return np.sum([charge.current_density(x, y, z, time) for charge in self.charges], axis=0)

    def _sum_fields(self, position, time, field_type):

        charge_field = np.sum([getattr(LW(charge, position, time), field_type)() for charge in self.charges], axis=0)
        external_field = np.zeros(3)
        if field_type in ["electric_field", "magnetic_field"]:
            external_field += self.external_fields[field_type](position, time)
        return charge_field + external_field

    def scalar_potential(self, position, time):
        return self._sum_fields(position, time, "scalar_potential")

    def vector_potential(self, position, time):
        return self._sum_fields(position, time, "vector_potential")

    def E_field(self, position, time):
        return self._sum_fields(position, time, "electric_field")

    def B_field(self, position, time):
        return self._sum_fields(position, time, "magnetic_field")

    def energy_density(self, position, time):
        return epsilon_0 / 2 * norm(self.E_field(position, time)) + 0.5 / mu_0 * norm(self.B_field(position, time))

    def poynting_vector(self, position, time):
        return 1 / mu_0 * np.cross(self.E_field(position, time), self.B_field(position, time))

    def lorentz_force(self, charge_index: int, time):
        charge = self.charges.pop(charge_index)
        e_field = self.E_field(charge.position(time), time)
        b_field = self.B_field(charge.position(time), time)
        self.charges.insert(charge_index, charge)
        return Utils.get_lorentz_force(e_field, b_field, charge.velocity(time), charge.magnitude)

    def set_charge_free(self, index: int):
        if isinstance(self.charges[index], BoundCharge):
            old = self.charges.pop(index)
            new = old.set_free()
            self.charges.insert(index, new)

    def _evolve_infinitesimal(self, dt=0.1):
        for i, charge in enumerate(self.charges):
            if isinstance(charge, FreeCharge):
                self.charges[i].evolve_infinitesimal(dt, self.lorentz_force(i, self.time))

    def evolve_by(self, time, dt=0.01):
        """Evolves the system forward by a time step dt."""
        iterations = int(time / dt)
        for _ in range(iterations):
            self._evolve_infinitesimal(dt)
            self._handle_charge_collision()
            self.time += dt


    def _handle_charge_collision(self):
        for charge1 in self.free_charges:
            for charge2 in self.charges:
                if System.are_close(charge1, charge2) and not System.is_repelling(charge1, charge2):
                    self.charges.remove(charge1)
                    self.charges.remove(charge2)

    def reset(self):
        """Resets the system back to its vacuum state."""
        self.charges = []
        self.time = 0
        print("System reset to vacuum state.")








#for t in range(100):

#   print(sys.E_field((11, 0, 0), t))
