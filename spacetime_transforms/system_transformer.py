import numpy as np

from system import System
from charges import FreeCharge, BoundCharge, Charge
from spacetime_transforms import Transformations, FieldTransforms


class SystemTransformer:
    def __init__(self, system: System, velocity=None, distance=None, orientation=None):
        """
        :param system: A System object to be transformed
        :param velocity: A tuple representing the velocity for Lorentz boost (e.g., (vx, vy, vz))
        :param distance: A tuple representing the translation distance (e.g., (dx, dy, dz))
        :param orientation: A tuple representing rotation (axis, angle) for orientation
        """
        self.old_system = system
        self.new_system = system
        self.velocity = np.array(velocity)
        self.distance = np.array(distance)
        self.orientation = orientation
        self.axis, self.angle = np.array(self.orientation) if self.orientation is not None else (None, None)

    def __repr__(self):
        return (f"Transformer(system={self.new_system}, velocity={self.velocity}, "
                f"distance={self.distance}, orientation={self.orientation})")

    def reset_system(self, system) -> None:
        self.old_system = system
        self.new_system = system

    def transformed_system(self) -> System:
        return self.new_system

    def poincare_transform(self) -> None:
        """
        Placeholder method for applying the transformations.
        """
        if self.velocity is not None:
            self.boost_system()
        if self.orientation is not None:
            self.rotate_system()
        if self.distance is not None:
            self.translate_system()

    def apply_transformation(self, transform_fn_charges: callable, transform_fn_fields: callable, new_time: float) -> None:
        new_charges = transform_fn_charges()
        new_e_field, new_b_field = transform_fn_fields()
        self.new_system = System(charges=new_charges, e_field=new_e_field, b_field=new_b_field, init_time=new_time)

    def boost_system(self) -> None:
        print(self.new_system.time, 'transformer')
        new_time = Transformations.boost(self.new_system.time, 0, 0, 0, self.velocity)[0]
        print(new_time)
        self.apply_transformation(self.boosted_charges, self.boosted_fields, new_time)

    def rotate_system(self) -> None:
        self.apply_transformation(self.rotated_charges, self.rotated_fields, self.new_system.time)

    def translate_system(self) -> None:
        self.apply_transformation(self.translated_charges, self.translated_fields, self.new_system.time)

    def boosted_fields(self) -> (callable, callable):
        e_field, b_field = self.new_system.external_fields.values()
        return FieldTransforms.boosted_fields(e_field, b_field, self.velocity)

    def rotated_fields(self) -> (callable, callable):
        e_field, b_field = self.new_system.external_fields.values()
        return FieldTransforms.rotated_fields(e_field, b_field, self.axis, self.angle)

    def translated_fields(self) -> (callable, callable):
        e_field, b_field = self.new_system.external_fields.values()
        return FieldTransforms.translated_fields(e_field, b_field, self.distance)

    def rotated_charges(self) -> list[Charge]:
        return self.get_transformed_charges(
            self.new_system.charges, lambda x, y, z: Transformations.rotate(x, y, z, self.axis, self.angle))

    def translated_charges(self) -> list[Charge]:
        return self.get_transformed_charges(
            self.new_system.charges,lambda x, y, z: Transformations.translate(x, y, z, self.distance))

    def boosted_charges(self) -> list[Charge]:
        return self.get_transformed_charges(
            self.new_system.charges, lambda t, x, y, z: Transformations.boost(t, x, y, z, self.velocity), is_boost=True)

    @staticmethod
    def get_transformed_charges(charges: list[Charge], transformation_fn: callable, is_boost=False) -> list[Charge]:
        charges_to_add = []
        for charge in charges:
            t_old = charge.times
            x_old, y_old, z_old = charge.position.T
            if is_boost:
                t, x, y, z = transformation_fn(t_old, x_old, y_old, z_old)
            else:
                t = t_old
                x, y, z = transformation_fn(x_old, y_old, z_old)
            new_charge = BoundCharge(x, y, z, t, charge.magnitude, charge.mass)
            if isinstance(charge, FreeCharge):
                new_charge = new_charge.set_free()
            charges_to_add.append(new_charge)
        return charges_to_add