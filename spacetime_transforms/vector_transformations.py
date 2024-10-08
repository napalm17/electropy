import numpy as np
from scipy.constants import c
from helpers import Utils


class Transformations:

    @staticmethod
    def poincare_transform(event: np.ndarray, velocity=None, distance=None, orientation=None) -> np.ndarray:
        t, x, y, z = event
        if velocity is not None:
            t, x, y, z = Transformations.boost(t, x, y, z, velocity)
        if distance is not None:
            x, y, z = Transformations.translate(x, y, z, distance)
        if orientation is not None:
            axis, angle = orientation
            x, y, z = Transformations.rotate(x, y, z, axis, angle)
        return np.array([t, x, y, z])

    @staticmethod
    def translate(x, y, z, distance) -> np.ndarray:
        dx, dy, dz = distance
        return np.array([x + dx, y + dy, z + dz])

    @staticmethod
    def boost_matrix(velocity: np.ndarray) -> np.ndarray:
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
    def boost(t, x, y, z, velocity: np.ndarray) -> np.ndarray:
        vector4d = np.array([c*t, x, y, z])
        ct_prime, x_prime, y_prime, z_prime = np.dot(Transformations.boost_matrix(velocity), vector4d)
        if isinstance(t, float):
            print(ct_prime, 'vec4')
        return np.array([ct_prime / c, x_prime, y_prime, z_prime])

    @staticmethod
    def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
        axis = axis / np.linalg.norm(axis)  # Normalize the axis
        axis_skew = Utils.skew(axis)  # Get skew-symmetric matrix of the axis
        return np.eye(3) + np.sin(angle) * axis_skew + (1 - np.cos(angle)) * np.dot(axis_skew, axis_skew)


    @staticmethod
    def rotate(x, y, z, axis: np.ndarray, angle: float) -> np.ndarray:
        vector3d = np.array([x, y, z])
        x_prime, y_prime, z_prime = np.dot(Transformations.rotation_matrix(axis, angle), vector3d)
        return np.array([x_prime, y_prime, z_prime])

