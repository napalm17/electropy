import numpy as np


class Utils:
    @staticmethod
    def skew(vector: np.ndarray) -> np.ndarray:
        """
        Parameters:
        vector : np.ndarray
        Returns:
        np.ndarray
            A 3x3 skew-symmetric matrix.
        """
        x, y, z = vector
        return np.array([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0]
        ])

    @staticmethod
    def zero_field(dimension):
        return lambda position, time: np.zeros(dimension)

    @staticmethod
    def get_lorentz_force(e_field, b_field, v, q):
        return q * (e_field + np.cross(v, b_field))

    @staticmethod
    def get_acceleration_from_force(force, mass, gamma, beta):
        """
        Calculate acceleration based on the applied force.
        :param force: Force to apply.
        :return: Acceleration vector.
        """
        return force / (mass * gamma * (1 - np.vdot(beta, beta)/gamma**2))


    @staticmethod
    def gaussian1D(x, sigma=1, mu=0):
        """
        Parameters:
        x (float or np.ndarray): The input value(s) for which to compute the Gaussian.
        sigma (float): The standard deviation of the Gaussian.
        mu (float): The mean (center) of the Gaussian.

        Returns:
        float or np.ndarray: The computed Gaussian value(s).
        """
        x = np.asarray(x)
        coeff = 1 / (sigma * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((x - mu) / sigma) ** 2
        return coeff * np.exp(exponent)

    @staticmethod
    def gaussian3D(x_range, y_range, z_range, mu, sigma):
        """Returns the joint Gaussian distribution over 3D space."""
        # Create a 3D grid
        X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        # Compute Gaussian distribution along each axis using broadcasting
        gaussian_distributions = np.array([
            Utils.gaussian1D((X, Y, Z)[i], mu=mu[i], sigma=sigma) for i in range(3)])

        return np.prod(gaussian_distributions, axis=0)

    @staticmethod
    def linear_interpolate(x_data, y_data, x):
        """
        Compute the interpolated value of y at a given x using linear interpolation.

        Parameters:
        x_data (np.ndarray): The array of x values.
        y_data (np.ndarray): The array of y values corresponding to x_data.
        x (float): The x value for which to compute the interpolated y value.

        Returns:
        float: The interpolated y value.
        """
        # Check if x is outside the range of x_data
        #if not (x_data[0] <= x <= x_data[-1]):
        #    print(x_data, x)
        #    raise ValueError(f'Value {x} not in function domain.')
        return np.array([np.interp(x, x_data, y_data[:, i]) for i in range(y_data.shape[1])])

        #for i in range(len(x_data) - 1):

         #   if x_data[i] <= x <= x_data[i + 1]:
          #      return y_data[i] + (y_data[i + 1] - y_data[i]) * (x - x_data[i]) / (x_data[i + 1] - x_data[i])

