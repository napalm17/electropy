from spacetime_transforms import SystemTransformer
from charges import BoundCharge, FreeCharge
from system import System
import numpy as np
from scipy.constants import c
from helpers import Utils
from visualizer import Visualizer

t = np.linspace(0, 10, 1000)
w = 10
T = 0.01
x = 5 * np.cos(t * (2 * np.pi) * w)
y = -5 * np.cos(t * (2 * np.pi) * w)
z = np.zeros_like(t)
charge1 = BoundCharge(x, y, z, t, 100, 1)


charge2 = BoundCharge(-x, y, z, t, 100, 1)

freed = FreeCharge((0, 0, 4), (0,0, 0), initial_time=0, magnitude=-25, mass=1)
freed2 = FreeCharge((5,0,2), (0,0,-10), initial_time=0, magnitude=-25)
freed3 = FreeCharge((0, 0, 5), (0,0, 0), initial_time=0, magnitude=-25, mass=100)


sys = System()
sys.add_charge(charge1)
sys.add_charge(charge2)
sys.add_charge(freed)
sys.add_charge(freed2)
#sys.add_field(lambda pos, time: np.array([0, 100, 0]), 'electric_field')
sys.add_field(lambda pos, time: np.array([0,0,1]), 'magnetic_field')
x_range = np.linspace(-5, 5, 5)  # 10 points from -5 to 5 in x
y_range = np.linspace(-5, 5, 5)  # 10 points from -5 to 5 in y
z_range = np.linspace(-5, 5, 5)  # 10 points from -5 to 5 in z
grid = np.meshgrid(x_range, y_range, z_range, indexing='ij')
# Initialize the Test Charge
positions = Utils.grid_to_vectors(grid)
# Initialize the System with the defined grid and the test charge
sys.evolve_by(2)
# Evaluate electric and magnetic fields on the grid


vis = Visualizer(sys)

vis.animate('electric_field', time_interval=(0, 90), pos=positions)






# sys.evolve_by(1)
#
# vel = np.array([0.1*c, 0, 0])
# dist = np.array([100, 200, 300])
# axis = np.array([0, 0, 1])
# angle = np.pi / 2
#
# print(sys.charge_velocities)
#
# transformer = SystemTransformer(sys, velocity=vel, distance=dist, orientation=(axis, angle))
#
# new_sys = transformer.poincare_transform()
# print('new system!!!!!!!!!!!!!!!!!!!!!!!')
# new_sys.evolve_by(3)
# print(new_sys.charge_velocities)




