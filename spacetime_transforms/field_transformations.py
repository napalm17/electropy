import numpy as np
from spacetime_transforms import Transformations
from scipy.constants import c

class FieldTransforms:

    @staticmethod
    def rotate_fields(E: callable, B: callable, axis, angle: float):
        rotation_matrix = Transformations.rotation_matrix(axis, angle)

        def E_prime(pos, t):
            return np.dot(rotation_matrix, E(pos, t))

        def B_prime(pos, t):
            return np.dot(rotation_matrix, B(pos, t))

        return E_prime, B_prime

    @staticmethod
    def translate_fields(E: callable, B: callable, d):
        """
        Performs constant translation on the electric and magnetic fields.
        Parameters:
        E : callable
            Function that returns the electric field E(pos, t) as a numpy array.
        B : callable
            Function that returns the magnetic field B(pos, t) as a numpy array.
        d : array-like
            3D displacement vector for the translation.

        Returns:
        E_prime : function
            Transformed electric field E'(pos, t).
        B_prime : function
            Transformed magnetic field B'(pos, t).
        """
        def E_prime(pos, t):
            return E(pos - d, t)

        def B_prime(pos, t):
            return B(pos - d, t)

        return E_prime, B_prime

    @staticmethod
    def boost_fields(E: callable, B: callable, v):
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
            return gamma * (B(pos, t) - np.cross(v, E(pos, t)) / c**2) - (gamma - 1) * np.dot(B(pos, t), v_hat) * v_hat

        return E_prime, B_prime