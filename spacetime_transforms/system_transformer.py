from system import System
from charges import FreeCharge, BoundCharge
from spacetime_transforms import Transformations, FieldTransforms

class SystemTransformer:
    def __init__(self, system: System, velocity=None, distance=None, orientation=None):
        """
        Initialize the Observer with a system and optional transformation parameters.

        :param system: A System object to be transformed
        :param velocity: A tuple representing the velocity for Lorentz boost (e.g., (vx, vy, vz))
        :param distance: A tuple representing the translation distance (e.g., (dx, dy, dz))
        :param orientation: A tuple representing rotation (axis, angle) for orientation
        """
        self.old_system = system
        self.new_system = system
        self.velocity = velocity
        self.distance = distance
        self.orientation = orientation
        self.axis, self.angle = self.orientation if self.orientation is not None else (None, None)

    def __repr__(self):
        return (f"Transformer(system={self.new_system}, velocity={self.velocity}, "
                f"distance={self.distance}, orientation={self.orientation})")

    def reset_system(self, system):
        self.old_system = system
        self.new_system = system

    def poincare_transform(self):
        """
        Placeholder method for applying the transformations.
        """
        if self.velocity is not None:
            self.boost_system()
        if self.orientation is not None:
            self.rotate_system()
        if self.distance is not None:
            self.translate_system()

        return self.new_system

    def apply_transformation(self, transformation_fn_charges, transformation_fn_fields, new_time):
        """
        General method to apply a transformation to the system.
        Parameters:
            transformation_fn_charges: A callable that applies a transformation to the charges.
            transformation_fn_fields: A callable that applies a transformation to the E and B fields.
            new_time: The transformed time after applying the transformation.
        """
        new_charges = transformation_fn_charges()
        new_e_field, new_b_field = transformation_fn_fields()
        self.new_system = System(charges=new_charges, e_field=new_e_field, b_field=new_b_field, init_time=new_time)

    def boost_system(self):
        print(self.new_system.time, 'transformer')
        new_time = Transformations.boost(self.new_system.time, 0, 0, 0, self.velocity)[0]
        print(new_time)
        self.apply_transformation(self.get_boosted_charges, self.get_boosted_fields, new_time)

    def rotate_system(self):
        self.apply_transformation(self.get_rotated_charges, self.get_rotated_fields, self.new_system.time)

    def translate_system(self):
        self.apply_transformation(self.get_translated_charges, self.get_translated_fields,self.new_system.time)

    def get_boosted_fields(self):
        e_field, b_field = self.new_system.external_fields.values()
        return FieldTransforms.boost_fields(e_field, b_field, self.velocity)

    def get_rotated_fields(self):
        e_field, b_field = self.new_system.external_fields.values()
        return FieldTransforms.rotate_fields(e_field, b_field, self.axis, self.angle)

    def get_translated_fields(self):
        e_field, b_field = self.new_system.external_fields.values()
        return FieldTransforms.translate_fields(e_field, b_field, self.distance)

    def get_rotated_charges(self):
        """
        Apply rotation transformation to the charges in the system.
        """
        return self.get_transformed_charges(
            self.new_system.charges, lambda x, y, z: Transformations.rotate(x, y, z, self.axis, self.angle))

    def get_translated_charges(self):
        """
        Apply translation transformation to the charges in the system.
        """
        return self.get_transformed_charges(
            self.new_system.charges,lambda x, y, z: Transformations.translate(x, y, z, self.distance))

    def get_boosted_charges(self):
        """
        :return:
        """
        return self.get_transformed_charges(
            self.new_system.charges, lambda t, x, y, z: Transformations.boost(t, x, y, z, self.velocity), is_boost=True)

    @staticmethod
    def get_transformed_charges(charges, transformation_fn, is_boost=False):
        """
        General method to apply a transformation to the charges in the system.
        Parameters:
            transformation_fn: A callable that takes x, y, z positions and applies the desired transformation.
                               Example: Transformations.rotate or Transformations.translate
        Returns:
            List of transformed charges.
        """
        charges_to_add = []
        for charge in charges:
            t_old = charge.times
            x_old, y_old, z_old = charge.positions.T
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