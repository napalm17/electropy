import numpy as np
from scipy.constants import c
from helpers import Utils

class Transformations:
    @staticmethod
    def transform_fields(E, B, v):
        """
        Perform Lorentz transformation on the electric and magnetic fields.

        Parameters:
        E : function
            Function that returns the electric field E(pos, t) as a numpy array.
        B : function
            Function that returns the magnetic field B(pos, t) as a numpy array.
        v : array-like
            3D velocity vector for the Lorentz boost.

        Returns:
        E_prime : function
            Transformed electric field E'(pos, t).
        B_prime : function
            Transformed magnetic field B'(pos, t).
        """

        v = np.asarray(v)
        v_mag = np.linalg.norm(v)
        gamma = 1 / np.sqrt(1 - (v_mag / c) ** 2)
        v_hat = v / v_mag if v_mag != 0 else np.zeros_like(v)  # Unit vector along v

        def E_prime(pos, t):
            return gamma * (E(pos, t) + np.cross(v, B(pos, t))) - (gamma - 1) * np.dot(E(pos, t), v_hat) * v_hat

        def B_prime(pos, t):
            return gamma * (B(pos, t) - np.cross(v, E(pos, t)) / c ** 2) - (gamma - 1) * np.dot(B(pos, t),
                                                                                                v_hat) * v_hat

        return E_prime, B_prime

    @staticmethod
    def poincare_transform(event: np.ndarray, velocity=None, distance=None, orientation=None):
        """
        Apply a series of transformations to a system based on the parameters.
        """
        t, x, y, z = event
        if velocity is not None:
            t, x, y, z = Transformations.boost(t, x, y, z, velocity)
        if distance is not None:
            x, y, z = Transformations.translate(x, y, z, distance)
        if orientation is not None:
            axis, angle = orientation
            x, y, z = Transformations.rotate(x, y, z, axis, angle)
        return t, x, y, z

    @staticmethod
    def translate(x, y, z, distance):
        """
        Apply an affine (translation) transformation to the system.
        """
        dx, dy, dz = distance
        return np.array([x + dx, y + dy, z + dz])

    @staticmethod
    def boost_matrix(velocity):
        """
        Compute the Lorentz boost matrix for a given 3D velocity vector.

        Parameters:
            velocity: A 3D array-like representing velocity (v_x, v_y, v_z).

        Returns:
            A 4x4 Lorentz boost matrix.
        """
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude == 0:
            return np.eye(4)  # Identity matrix for zero velocity

        beta = velocity_magnitude / c
        gamma = 1 / np.sqrt(1 - beta ** 2)
        u_x, u_y, u_z = velocity / velocity_magnitude

        boost_matrix = np.array([
            [gamma, -gamma * beta * u_x, -gamma * beta * u_y, -gamma * beta * u_z],
            [-gamma * beta * u_x, 1 + (gamma - 1) * u_x ** 2, (gamma - 1) * u_x * u_y, (gamma - 1) * u_x * u_z],
            [-gamma * beta * u_y, (gamma - 1) * u_x * u_y, 1 + (gamma - 1) * u_y ** 2, (gamma - 1) * u_y * u_z],
            [-gamma * beta * u_z, (gamma - 1) * u_x * u_z, (gamma - 1) * u_y * u_z, 1 + (gamma - 1) * u_z ** 2]
        ])

        return boost_matrix

    @staticmethod
    def boost(t, x, y, z, velocity):
        """
        Perform a Lorentz transformation on a 4-vector (x, y, z, t) with a given 3D velocity.
        Parameters:
            x, y, z, t: Arrays representing the 4-vector components.
            velocity: A 3D array-like representing velocity (v_x, v_y, v_z).
        Returns:
            Transformed 4-vector (x', y', z', t') as arrays.
        """
        vector4d = np.array([c*t, x, y, z])
        ct_prime, x_prime, y_prime, z_prime = np.dot(Transformations.boost_matrix(velocity), vector4d)
        return np.array([ct_prime / c, x_prime, y_prime, z_prime])

    @staticmethod
    def rotation_matrix(axis, angle):
        """
        Compute the 3D rotation matrix for a given axis and angle.
        Parameters:
        axis : array-like
            A 3D array-like representing the axis of rotation (x, y, z).
        angle : float
            The rotation angle in radians.

        Returns:
        numpy.ndarray
            A 3x3 rotation matrix.
        """
        axis = axis / np.linalg.norm(axis)  # Normalize the axis
        axis_skew = Utils.skew(axis)  # Get skew-symmetric matrix of the axis
        return np.eye(3) + np.sin(angle) * axis_skew + (1 - np.cos(angle)) * np.dot(axis_skew, axis_skew)



    @staticmethod
    def rotate(x, y, z, axis, angle):
        """
        Parameters:
            x, y, z: The spatial components of the 4-vector to be rotated.
            axis: A 3D array-like representing the axis of rotation (x, y, z).
            angle: The rotation angle in radians.
        Returns:
            A tuple (x', y', z'),  the rotated spatial components.
        """
        vector3d = np.array([x, y, z])
        x_prime, y_prime, z_prime = np.dot(Transformations.rotation_matrix(axis, angle), vector3d)
        return np.array([x_prime, y_prime, z_prime])

