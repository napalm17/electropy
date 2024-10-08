import numpy as np


class Utils:

    @staticmethod
    def scalars_to_grid(scalars: np.ndarray, grid_shape: tuple) -> np.ndarray:
        if scalars.ndim != 1:
            raise ValueError("Input must be a 1D array of scalars.")
        if scalars.shape[0] != np.prod(grid_shape):
            raise ValueError("The number of scalars must match the number of points in the grid.")
        return scalars.reshape(grid_shape)

    @staticmethod
    def grid_to_vectors(grid: np.ndarray) -> np.ndarray:
        xx, yy, zz = grid
        return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    @staticmethod
    def vectors_to_grid(vectors: np.ndarray, shape: tuple) -> np.ndarray:
        if vectors.shape[0] != np.prod(shape):
            raise ValueError("The number of vectors must match the number of points in the grid.")
        # Reshape the vectors to get the original grid shape
        xx = vectors[:, 0].reshape(shape)
        yy = vectors[:, 1].reshape(shape)
        zz = vectors[:, 2].reshape(shape)
        return xx, yy, zz

    @staticmethod
    def skew(vector: np.ndarray) -> np.ndarray:
        x, y, z = vector
        return np.array([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0]])

    @staticmethod
    def zero_field(dimension: int) -> callable:
        return lambda position, time: np.zeros(dimension)

    @staticmethod
    def equations_of_motion(t, y: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
        position = y[0:3]  # Assuming y contains position in the first 3 elements
        velocity = y[3:6]  # Assuming y contains velocity in the last 3 elements
        return np.concatenate((velocity, acceleration))

    @staticmethod
    def get_lorentz_force(e_field: np.ndarray, b_field: np.ndarray, v: np.ndarray, q: float) -> np.ndarray:
        return q * (e_field + np.cross(v, b_field))

    @staticmethod
    def get_acceleration_from_force(force: np.ndarray, mass: float, gamma: float, beta: np.ndarray) -> np.ndarray:
        return force / (mass * gamma * (1 - np.vdot(beta, beta)/gamma**2))

    @staticmethod
    def gaussian1D(x, sigma=1, mu=0):
        x = np.asarray(x)
        coeff = 1 / (sigma * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((x - mu) / sigma) ** 2
        return coeff * np.exp(exponent)

    @staticmethod
    def gaussian3D(x_range, y_range, z_range, mu, sigma):
        # Create a 3D grid
        X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        # Compute Gaussian distribution along each axis using broadcasting
        gaussian_distributions = np.array([
            Utils.gaussian1D((X, Y, Z)[i], mu=mu[i], sigma=sigma) for i in range(3)])

        return np.prod(gaussian_distributions, axis=0)

    @staticmethod
    def linear_interpolate(x_data: np.ndarray, y_data: np.ndarray, x) -> np.ndarray:
        return np.array([np.interp(x, x_data, y_data[:, i]) for i in range(y_data.shape[1])])


#print(freed.positions)
x_range = np.linspace(-5, 5, 10)  # 10 points from -5 to 5 in x
y_range = np.linspace(-5, 5, 10)  # 10 points from -5 to 5 in y
z_range = np.linspace(-5, 5, 10)  # 10 points from -5 to 5 in z
grid = np.meshgrid(x_range, y_range, z_range, indexing='ij')
# Initialize the Test Charge
positions = Utils.grid_to_vectors(grid)
grid2 = Utils.vectors_to_grid(positions, grid[0].shape)

print(np.allclose(grid, grid2))