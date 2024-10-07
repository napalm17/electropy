from inertial_frame_tranforms.vector_transformations import Transformations
import numpy as np
from scipy.constants import c

t, x, y, z = 1, 1, 3, 0
angle = np.pi
axis = (1, 0, 0)
dist = (1, 2, 3)
velocity = (0.5*c, 0, 0)

vec1 = Transformations.rotate(x, y, z, axis=axis, angle=angle)
vec2 = Transformations.translate(x, y, z, dist)
vec3 = Transformations.boost(t, x, y, z, velocity=velocity)
print(vec1)