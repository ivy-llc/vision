# global
import ivy as _ivy
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
        super(PrimitiveScene, self).__init__(
            sphere_positions=sphere_positions,
            sphere_radii=sphere_radii,
            cuboid_ext_mats=cuboid_ext_mats,
            cuboid_dims=cuboid_dims)

    # Class Methods #
    # --------------#

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
        super(Intrinsics, self).__init__(
            focal_lengths=focal_lengths,
            persp_angles=persp_angles,
            pp_offsets=pp_offsets,
            calib_mats=calib_mats,
            inv_calib_mats=inv_calib_mats)

    # Class Methods #
    # --------------#

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
        super(Extrinsics, self).__init__(
            cam_centers=cam_centers,
            Rs=Rs,
            inv_Rs=inv_Rs,
            ext_mats_homo=ext_mats_homo,
            inv_ext_mats_homo=inv_ext_mats_homo)

    # Class Methods #
    # --------------#

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
        super(CameraGeometry, self).__init__(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            full_mats_homo=full_mats_homo,
            inv_full_mats_homo=inv_full_mats_homo)

    # Class Methods #
    # --------------#

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
