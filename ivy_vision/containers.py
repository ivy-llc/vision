# global
import ivy as _ivy
try:
    import tensorflow as _tf
except (ModuleNotFoundError, ImportError):
    _tf = None
from ivy.core.container import Container as _Container

# local
import ivy_vision.sdf as ivy_sdf


# noinspection PyMissingConstructor
class PrimitiveScene(_Container):

    def __init__(self,
                 sphere_positions=None,
                 sphere_radii=None,
                 cuboid_ext_mats=None,
                 cuboid_dims=None):
        """
        Initialize scene description as a composition of primitive shapes.
        # ToDo: extend this to include cylinders and cones once supported in ivy_vision.sdf module

        :param sphere_positions: Sphere positions *[batch_shape,num_spheres,3]*
        :type sphere_positions: array, optional
        :param sphere_radii: Sphere radii *[batch_shape,num_spheres,1]*
        :type sphere_radii: array, optional
        :param cuboid_ext_mats: Cuboid inverse extrinsic matrices *[batch_shape,num_cuboids,3,4]*
        :type cuboid_ext_mats: array, optional
        :param cuboid_dims: Cuboid dimensions, in order of x, y, z *[batch_shape,num_cuboids,3]*
        :type cuboid_dims: array, optional
        """
        self['sphere_positions'] = sphere_positions
        self['sphere_radii'] = sphere_radii
        self['cuboid_ext_mats'] = cuboid_ext_mats
        self['cuboid_dims'] = cuboid_dims

    # Class Methods #
    # --------------#

    @staticmethod
    def as_keras_inputs(batch_shape):
        """
        Return primitive scene object with array attributes as keras.Input objects. Only applicable for tensorflow.

        :param batch_shape: Batch shape for each geometric array attribute
        :type batch_shape: sequence of ints
        :return: New primitive scene object, with each entry as keras.Input objects.
        """
        if _tf is None:
            raise Exception('Tensorflow is not installed, as_keras_inputs() cannot be called.')
        batch_shape = list(batch_shape)
        batch_size = batch_shape[0]
        batch_shape = batch_shape[1:] if len(batch_shape) > 1 else []
        sphere_positions = _tf.keras.Input((batch_shape + [3, 4]), batch_size=batch_size)
        sphere_radii = _tf.keras.Input((batch_shape + [1]), batch_size=batch_size)
        cuboid_ext_mats = _tf.keras.Input((batch_shape + [3, 4]), batch_size=batch_size)
        cuboid_dims = _tf.keras.Input((batch_shape + [3]), batch_size=batch_size)
        return __class__(sphere_positions, sphere_radii, cuboid_ext_mats, cuboid_dims)

    @staticmethod
    def as_tensor_spec(prefix):
        """
        Return primitive scene object with array attributes as tf.TensorSpec objects. Only applicable for tensorflow.

        :param prefix: prefix string for TensorSpec names.
        :type prefix: str
        :return: New primitive scene object, with each entry as TensorSpec objects.
        """
        if _tf is None:
            raise Exception('Tensorflow is not installed, as_keras_inputs() cannot be called.')
        sphere_positions = _tf.TensorSpec([1], _tf.float32, prefix + '_sphere_positions')
        sphere_radii = _tf.TensorSpec([3, 4], _tf.float32, prefix + '_sphere_radii')
        cuboid_ext_mats = _tf.TensorSpec([3], _tf.float32, prefix + '_cuboid_ext_mats')
        cuboid_dims = _tf.TensorSpec([3], _tf.float32, prefix + '_cuboid_dims')
        return __class__(sphere_positions, sphere_radii, cuboid_ext_mats, cuboid_dims)

    @staticmethod
    def as_identity(batch_shape):
        """
        Return primitive scene object with array attributes as either zeros or identity matrices.

        :param batch_shape: Batch shape for each geometric array attribute
        :type batch_shape: sequence of ints
        :return: New primitive scene object, with each entry as either zeros or identity matrices.
        """
        batch_shape = list(batch_shape)
        sphere_positions = _ivy.identity(4, batch_shape=batch_shape)[..., 0:3, :]
        sphere_radii = _ivy.ones(batch_shape + [1])
        cuboid_ext_mats = _ivy.identity(4, batch_shape=batch_shape)[..., 0:3, :]
        cuboid_dims = _ivy.ones(batch_shape + [3])
        return __class__(sphere_positions, sphere_radii, cuboid_ext_mats, cuboid_dims)

    # Public Methods #
    # ---------------#

    def set_slice(self, slice_obj, primitive_scene):
        """
        Set slice of primitive scene object.

        :param slice_obj: slice object to set slice for all container elements.
        :type slice_obj: slice of sequence of slices
        :param primitive_scene: Intrinsics object to set the slice equal to.
        :type primitive_scene: Intrinsics
        :return: PrimitiveScene object, after setting desired slice.
        """
        self.sphere_positions[slice_obj] = primitive_scene.sphere_positions
        self.sphere_radii[slice_obj] = primitive_scene.sphere_radii
        self.cuboid_ext_mats[slice_obj] = primitive_scene.cuboid_ext_mats
        self.cuboid_dims[slice_obj] = primitive_scene.cuboid_dims

    def sdf(self, query_positions):
        """
        Return signed distance function for the scene

        :param query_positions: Point for which to query the signed distance *[batch_shape,num_points,3]*
        :type query_positions: array
        :return: The signed distance values for each of the query points in the scene *[batch_shape,num_points,1]*
        """

        # BS x NP x 1
        all_sdfs_list = list()
        if self.sphere_positions is not None:
            sphere_sdfs = ivy_sdf.sphere_signed_distances(
                self.sphere_positions[..., 0:3, -1], self.sphere_radii, query_positions)
            all_sdfs_list.append(sphere_sdfs)
        if self.cuboid_ext_mats is not None:
            cuboid_sdfs = ivy_sdf.cuboid_signed_distances(
                self.cuboid_ext_mats, self.cuboid_dims, query_positions)
            all_sdfs_list.append(cuboid_sdfs)
        sdfs_concatted = _ivy.concatenate(all_sdfs_list, -1) if len(all_sdfs_list) > 1 else all_sdfs_list[0]
        return _ivy.reduce_min(sdfs_concatted, -1, keepdims=True)

    # Getters #
    # --------#

    @property
    def batch_shape(self):
        """
        Batch shape of each element in container
        """
        return self.shape_types.batch_shape[:-1]


# noinspection PyMissingConstructor
class Intrinsics(_Container):

    def __init__(self,
                 focal_lengths,
                 persp_angles,
                 pp_offsets,
                 calib_mats,
                 inv_calib_mats):
        """
        Initialize camera intrinsics container.

        :param focal_lengths: Focal lengths *[batch_shape,2]*
        :type focal_lengths: array
        :param persp_angles: perspective angles *[batch_shape,2]*
        :type persp_angles: array
        :param pp_offsets: Principal-point offsets *[batch_shape,2]*
        :type pp_offsets: array
        :param calib_mats: Calibration matrices *[batch_shape,3,3]*
        :type calib_mats: array
        :param inv_calib_mats: Inverse calibration matrices *[batch_shape,3,3]*
        :type inv_calib_mats: array
        """
        self['focal_lengths'] = focal_lengths
        self['persp_angles'] = persp_angles
        self['pp_offsets'] = pp_offsets
        self['calib_mats'] = calib_mats
        self['inv_calib_mats'] = inv_calib_mats

    # Class Methods #
    # --------------#

    @staticmethod
    def as_keras_inputs(batch_shape):
        """
        Return camera intrinsics object with array attributes as keras.Input objects. Only applicable for tensorflow.

        :param batch_shape: Batch shape for each geometric array attribute
        :type batch_shape: sequence of ints
        :return: New camera intrinsics object, with each entry as keras.Input objects.
        """
        if _tf is None:
            raise Exception('Tensorflow is not installed, as_keras_inputs() cannot be called.')
        batch_shape = list(batch_shape)
        batch_size = batch_shape[0]
        batch_shape = batch_shape[1:] if len(batch_shape) > 1 else []
        focal_lengths = _tf.keras.Input((batch_shape + [2]), batch_size=batch_size)
        persp_angles = _tf.keras.Input((batch_shape + [2]), batch_size=batch_size)
        pp_offsets = _tf.keras.Input((batch_shape + [2]), batch_size=batch_size)
        calib_mats = _tf.keras.Input((batch_shape + [3, 3]), batch_size=batch_size)
        inv_calib_mats = _tf.keras.Input((batch_shape + [3, 3]), batch_size=batch_size)
        return __class__(focal_lengths, persp_angles, pp_offsets, calib_mats, inv_calib_mats)

    @staticmethod
    def as_tensor_spec(prefix):
        """
        Return camera intrinsics object with array attributes as tf.TensorSpec objects. Only applicable for tensorflow.

        :param prefix: prefix string for TensorSpec names.
        :type prefix: str
        :return: New camera intrinsics object, with each entry as TensorSpec objects.
        """
        if _tf is None:
            raise Exception('Tensorflow is not installed, as_keras_inputs() cannot be called.')
        focal_lengths = _tf.TensorSpec([2], _tf.float32, prefix + '_focal_lengths')
        persp_angles = _tf.TensorSpec([2], _tf.float32, prefix + '_persp_angles')
        pp_offsets = _tf.TensorSpec([2], _tf.float32, prefix + '_pp_offsets')
        calib_mats = _tf.TensorSpec([3, 3], _tf.float32, prefix + '_calib_mats')
        inv_calib_mats = _tf.TensorSpec([3, 3], _tf.float32, prefix + '_inv_calib_mats')
        return __class__(focal_lengths, persp_angles, pp_offsets, calib_mats, inv_calib_mats)

    @staticmethod
    def as_identity(batch_shape):
        """
        Return camera intrinsics object with array attributes as either zeros or identity matrices.

        :param batch_shape: Batch shape for each geometric array attribute
        :type batch_shape: sequence of ints
        :return: New camera intrinsics object, with each entry as either zeros or identity matrices.
        """
        batch_shape = list(batch_shape)
        focal_lengths = _ivy.ones(batch_shape + [2])
        persp_angles = _ivy.ones(batch_shape + [2])
        pp_offsets = _ivy.zeros(batch_shape + [2])
        calib_mats = _ivy.identity(3, batch_shape=batch_shape)
        inv_calib_mats = _ivy.identity(3, batch_shape=batch_shape)
        return __class__(focal_lengths, persp_angles, pp_offsets, calib_mats, inv_calib_mats)

    # Public Methods #
    # ---------------#

    def set_slice(self, slice_obj, intrinsics):
        """
        Set slice of intrinsics object.

        :param slice_obj: slice object to set slice for all container elements.
        :type slice_obj: slice of sequence of slices
        :param intrinsics: Intrinsics object to set the slice equal to.
        :type intrinsics: Intrinsics
        :return: Intrinsics object, after setting desired slice.
        """
        self.focal_lengths[slice_obj] = intrinsics.focal_lengths
        self.persp_angles[slice_obj] = intrinsics.persp_angles
        self.pp_offsets[slice_obj] = intrinsics.pp_offsets
        self.calib_mats[slice_obj] = intrinsics.calib_mats
        self.inv_calib_mats[slice_obj] = intrinsics.inv_calib_mats

    # Getters #
    # --------#

    @property
    def batch_shape(self):
        """
        Batch shape of each element in container
        """
        return self.focal_lengths.batch_shape[:-1]


# noinspection PyMissingConstructor
class Extrinsics(_Container):

    def __init__(self,
                 cam_centers,
                 Rs,
                 inv_Rs,
                 ext_mats_homo,
                 inv_ext_mats_homo):
        """
        Initialize camera extrinsics container.

        :param cam_centers: Camera centers *[batch_shape,3,1]*
        :type cam_centers: array
        :param Rs: Rotation matrices *[batch_shape,3,3]*
        :type Rs: array
        :param inv_Rs: Inverse rotation matrices *[batch_shape,3,3]*
        :type inv_Rs: array
        :param ext_mats_homo: Homogeneous extrinsic matrices *[batch_shape,4,4]*
        :type ext_mats_homo: array
        :param inv_ext_mats_homo: Inverse homogeneous extrinsic matrices *[batch_shape,4,4]*
        :type inv_ext_mats_homo: array
        """
        self['cam_centers'] = cam_centers
        self['Rs'] = Rs
        self['inv_Rs'] = inv_Rs
        self['ext_mats_homo'] = ext_mats_homo
        self['inv_ext_mats_homo'] = inv_ext_mats_homo

    # Class Methods #
    # --------------#

    @staticmethod
    def as_keras_inputs(batch_shape):
        """
        Return camera extrinsics object with array attributes as keras.Input objects. Only applicable for tensorflow.

        :param batch_shape: Batch shape for each geometric array attribute.
        :type batch_shape: sequence of ints
        :return: New camera extrinsics object, with each entry as keras.Input objects.
        """
        if _tf is None:
            raise Exception('Tensorflow is not installed, as_keras_inputs() cannot be called.')
        batch_shape = list(batch_shape)
        batch_size = batch_shape[0]
        batch_shape = batch_shape[1:] if len(batch_shape) > 1 else []
        cam_centers = _tf.keras.Input(tuple(batch_shape + [3, 1]), batch_size=batch_size)
        Rs = _tf.keras.Input((batch_shape + [3, 3]), batch_size=batch_size)
        inv_Rs = _tf.keras.Input((batch_shape + [3, 3]), batch_size=batch_size)
        ext_mats_homo = _tf.keras.Input((batch_shape + [4, 4]), batch_size=batch_size)
        inv_ext_mats_homo = _tf.keras.Input((batch_shape + [4, 4]), batch_size=batch_size)
        return __class__(cam_centers, Rs, inv_Rs, ext_mats_homo, inv_ext_mats_homo)

    @staticmethod
    def as_tensor_spec(prefix):
        """
        Return camera extrinsics object with array attributes as tf.TensorSpec objects. Only applicable for tensorflow.

        :param prefix: prefix string for TensorSpec names.
        :type prefix: str
        :return: New camera extrinsics object, with each entry as TensorSpec objects.
        """
        if _tf is None:
            raise Exception('Tensorflow is not installed, as_keras_inputs() cannot be called.')
        cam_centers = _tf.TensorSpec([3, 1], _tf.float32, prefix + '_cam_centers')
        Rs = _tf.TensorSpec([3, 3], _tf.float32, prefix + '_Rs')
        inv_Rs = _tf.TensorSpec([3, 3], _tf.float32, prefix + '_inv_Rs')
        ext_mats_homo = _tf.TensorSpec([4, 4], _tf.float32, prefix + '_ext_mats_homo')
        inv_ext_mats_homo = _tf.TensorSpec([4, 4], _tf.float32, prefix + '_inv_ext_mats_homo')
        return __class__(cam_centers, Rs, inv_Rs, ext_mats_homo, inv_ext_mats_homo)

    @staticmethod
    def as_identity(batch_shape):
        """
        Return camera extrinsics object with array attributes as either zeros or identity matrices.

        :param batch_shape: Batch shape for each geometric array attribute.
        :type batch_shape: sequence of ints
        :return: New camera extrinsics object, with each entry as either zeros or identity matrices.
        """
        batch_shape = list(batch_shape)
        cam_centers = _ivy.zeros(batch_shape + [3, 1])
        Rs = _ivy.identity(3, batch_shape=batch_shape)
        inv_Rs = _ivy.identity(3, batch_shape=batch_shape)
        ext_mats_homo = _ivy.identity(4, batch_shape=batch_shape)
        inv_ext_mats_homo = _ivy.identity(4, batch_shape=batch_shape)
        return __class__(cam_centers, Rs, inv_Rs, ext_mats_homo, inv_ext_mats_homo)

    # Public Methods #
    # ---------------#

    def set_slice(self, slice_obj, extrinsics):
        """
        Set slice of extrinsics object.

        :param slice_obj: slice object to set slice for all container elements.
        :type slice_obj: slice of sequence of slices
        :param extrinsics: Extrinsics object to set the slice equal to.
        :type extrinsics: Extrinsics
        :return: Extrinsics object, after setting desired slice.
        """
        self.cam_centers[slice_obj] = extrinsics.cam_centers
        self.Rs[slice_obj] = extrinsics.Rs
        self.inv_Rs[slice_obj] = extrinsics.inv_Rs
        self.ext_mats_homo[slice_obj] = extrinsics.ext_mats_homo
        self.inv_ext_mats_homo[slice_obj] = extrinsics.inv_ext_mats_homo

    # Getters #
    # --------#

    @property
    def batch_shape(self):
        """
        Batch shape of each element in container
        """
        return self.cam_centers.batch_shape[:-2]


# noinspection PyMissingConstructor
class CameraGeometry(_Container):

    def __init__(self,
                 intrinsics,
                 extrinsics,
                 full_mats_homo,
                 inv_full_mats_homo):
        """
        Initialize camera geometry container.

        :param intrinsics: Camera intrinsics object.
        :type intrinsics: Intrinsics
        :param extrinsics: Camera extrinsics object.
        :type extrinsics: Extrinsics
        :param full_mats_homo: Full homogeneous projection matrices *[batch_shape,4,4]*
        :type full_mats_homo: array
        :param inv_full_mats_homo: Inverse full homogeneous projection matrices *[batch_shape,4,4]*
        :type inv_full_mats_homo: array
        """
        self['intrinsics'] = intrinsics
        self['extrinsics'] = extrinsics
        self['full_mats_homo'] = full_mats_homo
        self['inv_full_mats_homo'] = inv_full_mats_homo

    # Class Methods #
    # --------------#

    @staticmethod
    def as_keras_inputs(batch_shape):
        """
        Return camera geometry object with array attributes as keras.Input objects. Only applicable for tensorflow.

        :param batch_shape: Batch shape for each geometric array attribute
        :type batch_shape: sequence of ints
        :return: New camera geometry object, with each entry as keras.Input objects.
        """
        if _tf is None:
            raise Exception('Tensorflow is not installed, as_keras_inputs() cannot be called.')
        intrinsics = Intrinsics.as_keras_inputs(batch_shape)
        extrinsics = Extrinsics.as_keras_inputs(batch_shape)
        batch_shape = list(batch_shape)
        batch_size = batch_shape[0]
        batch_shape = batch_shape[1:] if len(batch_shape) > 1 else []
        full_mats_homo = _tf.keras.Input(batch_shape + [4, 4], batch_size=batch_size)
        inv_full_mats_homo = _tf.keras.Input((batch_shape + [4, 4]), batch_size=batch_size)
        return __class__(intrinsics, extrinsics, full_mats_homo, inv_full_mats_homo)

    @staticmethod
    def as_tensor_spec(prefix):
        """
        Return camera geometry object with array attributes as tf.TensorSpec objects. Only applicable for tensorflow.

        :param prefix: prefix string for TensorSpec names.
        :type prefix: str
        :return: New camera geometry object, with each entry as TensorSpec objects.
        """
        if _tf is None:
            raise Exception('Tensorflow is not installed, as_tensor_spec() cannot be called.')
        intrinsics = Intrinsics.as_tensor_spec(prefix)
        extrinsics = Extrinsics.as_tensor_spec(prefix)
        full_mats_homo = _tf.TensorSpec([4, 4], _tf.float32, prefix + '_full_mats_homo')
        inv_full_mats_homo = _tf.TensorSpec([4, 4], _tf.float32, prefix + '_inv_full_mats_homo')
        return __class__(intrinsics, extrinsics, full_mats_homo, inv_full_mats_homo)

    @staticmethod
    def as_identity(batch_shape):
        """
        Return camera geometry object with array attributes as either zeros or identity matrices.

        :param batch_shape: Batch shape for each geometric array attribute
        :type batch_shape: sequence of ints
        :return: New camera geometry object, with each entry as either zeros or identity matrices.
        """
        intrinsics = Intrinsics.as_identity(batch_shape)
        extrinsics = Extrinsics.as_identity(batch_shape)
        full_mats_homo = _ivy.identity(4, batch_shape=batch_shape)
        inv_full_mats_homo = _ivy.identity(4, batch_shape=batch_shape)
        return __class__(intrinsics, extrinsics, full_mats_homo, inv_full_mats_homo)

    # Public Methods #
    # ---------------#

    def set_slice(self, slice_obj, cam_geom):
        """
        Set slice of camera geometry object.

        :param slice_obj: slice object to set slice for all container elements.
        :type slice_obj: slice of sequence of slices
        :param cam_geom: Camera geometry object to set the slice equal to.
        :type cam_geom: CameraGeometry
        :return: CameraGeometry object, after setting desired slice.
        """
        self.intrinsics.set_slice(slice_obj, cam_geom.intrinsics)
        self.extrinsics.set_slice(slice_obj, cam_geom.extrinsics)
        self.full_mats_homo[slice_obj] = cam_geom.full_mats_homo
        self.inv_full_mats_homo[slice_obj] = cam_geom.inv_full_mats_homo

    # Getters #
    # --------#

    @property
    def batch_shape(self):
        """
        Batch shape of each element in container
        """
        return self.full_mats_homo.batch_shape[:-2]
