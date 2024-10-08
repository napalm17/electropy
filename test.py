from spacetime_transforms import SystemTransformer
from charges import BoundCharge, FreeCharge
from system import System
import numpy as np
from scipy.constants import c


t = np.linspace(-5, 100, 100)
w = 1
x = 10 * np.cos(w * t / (2 * np.pi))
y = np.zeros_like(t)
z = np.zeros_like(t)
charge1 = BoundCharge(x, y, z, t, 1, 1)
charge2 = BoundCharge(-y + 20, y, z, t, 1, 1)

freed = FreeCharge((0, 0, 1), (0, 0.2*c, 0), initial_time=0, magnitude=100)
freed2 = FreeCharge((10,0,0), (0,0,0), initial_time=0, magnitude=-12)


sys = System()
#sys.add_charge(charge1)
#sys.add_charge(freed)
sys.add_charge(freed2)
#sys.add_field(lambda position, time: np.array([0, 100, 0]), 'electric_field')
#sys.add_field(lambda r, t: np.array([0,0,1]), 'magnetic_field')

#print(freed.positions)

sys.evolve_by(1)

vel = np.array([0.1*c, 0, 0])
dist = np.array([100, 200, 300])
axis = np.array([0, 0, 1])
angle = np.pi / 2

print(sys.charge_velocities)

transformer = SystemTransformer(sys, velocity=vel, distance=dist, orientation=(axis, angle))

new_sys = transformer.poincare_transform()
print('new system!!!!!!!!!!!!!!!!!!!!!!!')
#new_sys.evolve_by(3)
print(new_sys.charge_velocities)




