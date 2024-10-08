from charges.charge import Charge
import numpy as np
from scipy.constants import c
from scipy.integrate import solve_ivp
from helpers import Utils


class FreeCharge(Charge):
    def __init__(self, initial_position=None, initial_velocity=None, initial_time=None, mass=1.0, magnitude=1.0, old_values=None):
        """
        Initialize a free-moving charge.
        """
        super().__init__(mass=mass, magnitude=magnitude)
        if old_values is not None:
            old_t, old_pos, old_vel, old_acc = old_values
            self._positions = old_pos
            self._velocities = old_vel
            self._accelerations = old_acc
            self._times = old_t
        else:
            self._positions = np.array([initial_position])
            self._velocities = np.array([initial_velocity])
            self._accelerations = np.array([[0, 0, 0]])
            self._times = np.array([initial_time])

    @property
    def positions(self):
        return self._positions

    @property
    def velocities(self):
        return self._velocities

    @property
    def accelerations(self):
        return self._accelerations

    @property
    def times(self):
        return self._times


    def _update_time(self, new_time):
        """
        Update the time array by adding the new time.
        """
        self._times = np.concatenate((self.times, [new_time]))

    def _update_position(self, new_position):
        """
        Update the position array by adding the new position.
        """
        self._positions = np.vstack((self._positions, new_position))

    def _update_velocity(self, new_velocity):
        """
        Update the velocity array by adding the new velocity.
        If the velocity exceeds the speed of light, renormalize it accordingly.
        """
        velocity_magnitude = np.linalg.norm(new_velocity)
        if velocity_magnitude > c:
            new_velocity = (new_velocity / velocity_magnitude) * (c-1e-5)  # Scale down to c
        self._velocities = np.vstack((self._velocities, new_velocity))

    def _update_acceleration(self, new_acceleration):
        """
        Update the acceleration array by adding the new acceleration.
        """
        self._accelerations = np.vstack((self._accelerations, new_acceleration))

    def equations_of_motion(self, t, y, acceleration):
        position = y[0:3]  # Assuming y contains position in the first 3 elements
        velocity = y[3:6]  # Assuming y contains velocity in the last 3 elements
        return np.concatenate((velocity, acceleration))

    def evolve_infinitesimal(self, dt, force):
        """
        Evolve the charge's position and velocity using the solve_ivp method.
        :param dt: Time step.
        :param force: Force to apply.
        """
        initial_conditions = np.concatenate((self.position(), self.velocity()))
        t_span = (self.times[-1], self.times[-1] + dt)
        t_eval = [self.times[-1] + dt]  # We want to evaluate only at the new time point
        a = Utils.get_acceleration_from_force(force, self.mass, self.gamma(), self.beta())
        self._update_acceleration(a)
        solution = solve_ivp(self.equations_of_motion, t_span, initial_conditions, args=(a,), t_eval=t_eval)
        self._update_position(solution.y[0:3, -1])  # New position
        self._update_velocity(solution.y[3:6, -1])  # New velocity
        self._update_time(solution.t[-1])  # New time


