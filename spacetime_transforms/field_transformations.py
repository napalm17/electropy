import numpy as np
from spacetime_transforms import Transformations
from scipy.constants import c


class FieldTransforms:

    @staticmethod
    def rotated_fields(E: callable, B: callable, axis, angle: float) -> (callable, callable):
        rotation_matrix = Transformations.rotation_matrix(axis, angle)
        def E_prime(pos, t):
            return np.dot(rotation_matrix, E(pos, t))
        def B_prime(pos, t):
            return np.dot(rotation_matrix, B(pos, t))

        return E_prime, B_prime

    @staticmethod
    def translated_fields(E: callable, B: callable, d: np.ndarray) -> (callable, callable):
        def E_prime(pos, t):
            return E(pos - d, t)
        def B_prime(pos, t):
            return B(pos - d, t)

        return E_prime, B_prime

    @staticmethod
    def boosted_fields(E: callable, B: callable, v: np.ndarray) -> (callable, callable):
        v = np.asarray(v)
        v_mag = np.linalg.norm(v)
        gamma = 1 / np.sqrt(1 - (v_mag / c) ** 2)
        v_hat = v / v_mag if v_mag != 0 else np.zeros_like(v)  # Unit vector along v
        def E_prime(pos, t):
            return gamma * (E(pos, t) + np.cross(v, B(pos, t), axis=-1)) - (gamma - 1) * np.dot(E(pos, t), v_hat) * v_hat
        def B_prime(pos, t):
            return gamma * (B(pos, t) - np.cross(v, E(pos, t), axis=-1) / c**2) - (gamma - 1) * np.dot(B(pos, t), v_hat) * v_hat

        return E_prime, B_prime