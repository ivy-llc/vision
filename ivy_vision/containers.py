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
        """Initialize scene description as a composition of primitive shapes.
        # ToDo: extend this to include cylinders and cones once supported in ivy_vision.sdf module

        Parameters
        ----------
        sphere_positions
            Sphere positions *[batch_shape,num_spheres,3]* (Default value = None)
        sphere_radii
            Sphere radii *[batch_shape,num_spheres,1]* (Default value = None)
        cuboid_ext_mats
            Cuboid inverse extrinsic matrices *[batch_shape,num_cuboids,3,4]* (Default value = None)
        cuboid_dims
            Cuboid dimensions, in order of x, y, z *[batch_shape,num_cuboids,3]* (Default value = None)

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
        """Return primitive scene object with array attributes as either zeros or identity matrices.

        Parameters
        ----------
        batch_shape
            Batch shape for each geometric array attribute

        Returns
        -------
        ret
            New primitive scene object, with each entry as either zeros or identity matrices.

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
        """Return signed distance function for the scene

        Parameters
        ----------
        query_positions
            Point for which to query the signed distance *[batch_shape,num_points,3]*

        Returns
        -------
        ret
            The signed distance values for each of the query points in the scene *[batch_shape,num_points,1]*

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
        """Initialize camera intrinsics container.

        Parameters
        ----------
        focal_lengths
            Focal lengths *[batch_shape,2]*
        persp_angles
            perspective angles *[batch_shape,2]*
        pp_offsets
            Principal-point offsets *[batch_shape,2]*
        calib_mats
            Calibration matrices *[batch_shape,3,3]*
        inv_calib_mats
            Inverse calibration matrices *[batch_shape,3,3]*

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
        """Return camera intrinsics object with array attributes as either zeros or identity matrices.

        Parameters
        ----------
        batch_shape
            Batch shape for each geometric array attribute

        Returns
        -------
        ret
            New camera intrinsics object, with each entry as either zeros or identity matrices.

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
        """Initialize camera extrinsics container.

        Parameters
        ----------
        cam_centers
            Camera centers *[batch_shape,3,1]*
        Rs
            Rotation matrices *[batch_shape,3,3]*
        inv_Rs
            Inverse rotation matrices *[batch_shape,3,3]*
        ext_mats_homo
            Homogeneous extrinsic matrices *[batch_shape,4,4]*
        inv_ext_mats_homo
            Inverse homogeneous extrinsic matrices *[batch_shape,4,4]*

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
        """Return camera extrinsics object with array attributes as either zeros or identity matrices.

        Parameters
        ----------
        batch_shape
            Batch shape for each geometric array attribute.

        Returns
        -------
        ret
            New camera extrinsics object, with each entry as either zeros or identity matrices.

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
        """Initialize camera geometry container.

        Parameters
        ----------
        intrinsics
            Camera intrinsics object.
        extrinsics
            Camera extrinsics object.
        full_mats_homo
            Full homogeneous projection matrices *[batch_shape,4,4]*
        inv_full_mats_homo
            Inverse full homogeneous projection matrices *[batch_shape,4,4]*

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
        """Return camera geometry object with array attributes as either zeros or identity matrices.

        Parameters
        ----------
        batch_shape
            Batch shape for each geometric array attribute

        Returns
        -------
        ret
            New camera geometry object, with each entry as either zeros or identity matrices.

        """
        intrinsics = Intrinsics.as_identity(batch_shape)
        extrinsics = Extrinsics.as_identity(batch_shape)
        full_mats_homo = _ivy.identity(4, batch_shape=batch_shape)
        inv_full_mats_homo = _ivy.identity(4, batch_shape=batch_shape)
        return __class__(intrinsics, extrinsics, full_mats_homo, inv_full_mats_homo)
