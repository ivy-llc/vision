"""
Collection of Single-View-Geometry Functions
"""

# global
from functools import reduce as _reduce
from operator import mul as _mul
import numpy as np
import ivy_mech as _ivy_mec
from ivy.framework_handler import get_framework as _get_framework

# local
from ivy_vision import projective_geometry as _ivy_pg
from ivy_vision.containers import Intrinsics as _Intrinsics
from ivy_vision.containers import Extrinsics as _Extrinsics
from ivy_vision.containers import CameraGeometry as _CameraGeometry


MIN_DENOMINATOR = 1e-12


def create_uniform_pixel_coords_image(image_dims, batch_shape=None, normalized=False, dev='cpu', f=None):
    """
    Create image of homogeneous integer :math:`xy` pixel co-ordinates :math:`\mathbf{X}\in\mathbb{Z}^{h×w×3}`, stored
    as floating point values. The origin is at the top-left corner of the image, with :math:`+x` rightwards, and
    :math:`+y` downwards. The final homogeneous dimension are all ones. In subsequent use of this image, the depth of
    each pixel can be represented using this same homogeneous representation, by simply scaling each 3-vector by the
    depth value. The final dimension therefore always holds the depth value, while the former two dimensions hold depth
    scaled pixel co-ordinates.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=172>`_
    deduction from top of page 154, section 6.1, equation 6.1

    :param image_dims: Image dimensions.
    :type image_dims: sequence of ints.
    :param batch_shape: Shape of batch. Assumed no batch dimensions if None.
    :type batch_shape: sequence of ints, optional
    :param normalized: Whether to normalize x-y pixel co-ordinates to the range 0-1.
    :type normalized: bool
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    :type dev: str
    :param f: Machine learning framework. Global framework used if None.
    :type f: ml_framework, optional
    :return: Image of homogeneous pixel co-ordinates *[batch_shape,height,width,3]*
    """

    f = _get_framework(f=f)

    # shapes as lists
    batch_shape = [] if batch_shape is None else batch_shape
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # other shape specs
    num_batch_dims = len(batch_shape)
    flat_shape = [1] * num_batch_dims + image_dims + [3]
    tile_shape = batch_shape + [1] * 3

    # H x W x 1
    pixel_x_coords = f.cast(f.reshape(f.tile(f.arange(image_dims[1], dev=dev), [image_dims[0]]),
                                            (image_dims[0], image_dims[1], 1)), 'float32')
    if normalized:
        pixel_x_coords = pixel_x_coords / (float(image_dims[1]) + MIN_DENOMINATOR)

    # W x H x 1
    pixel_y_coords_ = f.cast(f.reshape(f.tile(f.arange(image_dims[0], dev=dev), [image_dims[1]]),
                                             (image_dims[1], image_dims[0], 1)), 'float32')

    # H x W x 1
    pixel_y_coords = f.transpose(pixel_y_coords_, (1, 0, 2))
    if normalized:
        pixel_y_coords = pixel_y_coords / (float(image_dims[0]) + MIN_DENOMINATOR)

    # H x W x 1
    ones = f.ones_like(pixel_x_coords, dev=dev)

    # BS x H x W x 3
    return f.tile(f.reshape(f.concatenate((pixel_x_coords, pixel_y_coords, ones), -1),
                                  flat_shape), tile_shape)


def persp_angles_to_focal_lengths(persp_angles, image_dims, dev=None, f=None):
    """
    Compute focal lengths :math:`f_x, f_y` from perspective angles :math:`θ_x, θ_y`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=172>`_
    deduction from page 154, section 6.1, figure 6.1

    :param persp_angles: Perspective angles *[batch_shape,2]*
    :type persp_angles: array
    :param image_dims: Image dimensions.
    :type image_dims: sequence of ints
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Focal lengths *[batch_shape,2]*
    """

    f = _get_framework(persp_angles, f=f)

    if dev is None:
        dev = f.get_device(persp_angles)

    # shapes as list
    image_dims = list(image_dims)

    # BS x 2
    return -f.flip(f.cast(f.array(image_dims, dev=dev), 'float32'), -1) / \
           (2 * f.tan(persp_angles / 2) + MIN_DENOMINATOR)


def focal_lengths_to_persp_angles(focal_lengths, image_dims, dev=None, f=None):
    """
    Compute perspective angles :math:`θ_x, θ_y` from focal lengths :math:`f_x, f_y`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=172>`_
    deduction from page 154, section 6.1, figure 6.1

    :param focal_lengths: *[batch_shape,2]*
    :type focal_lengths: array
    :param image_dims: Image dimensions.
    :type image_dims: sequence of ints
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Perspective angles *[batch_shape,2]*
    """

    f = _get_framework(focal_lengths, f=f)

    if dev is None:
        dev = f.get_device(focal_lengths)

    # shapes as list
    image_dims = list(image_dims)

    # BS x 2
    return -2 * f.atan(f.flip(f.cast(f.array(image_dims, dev=dev),
                                              'float32'), -1) / (2 * focal_lengths + MIN_DENOMINATOR))


def focal_lengths_and_pp_offsets_to_calib_mat(focal_lengths, pp_offsets, batch_shape=None, dev=None, f=None):
    """
    Compute calibration matrix :math:`\mathbf{K}\in\mathbb{R}^{3×3}` from focal lengths :math:`f_x, f_y` and
    principal-point offsets :math:`p_x, p_y`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=173>`_
    page 155, section 6.1, equation 6.4

    :param focal_lengths: Focal lengths *[batch_shape,2]*
    :type focal_lengths: array
    :param pp_offsets: Principal-point offsets *[batch_shape,2]*
    :type pp_offsets: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Calibration matrix *[batch_shape,3,3]*
    """

    f = _get_framework(focal_lengths, f=f)

    if batch_shape is None:
        batch_shape = focal_lengths.shape[:-1]

    if dev is None:
        dev = f.get_device(focal_lengths)

    # shapes as list
    batch_shape = list(batch_shape)

    # BS x 1 x 1
    zeros = f.zeros(batch_shape + [1, 1], dev=dev)
    ones = f.ones(batch_shape + [1, 1], dev=dev)

    # BS x 2 x 1
    focal_lengths_reshaped = f.expand_dims(focal_lengths, -1)
    pp_offsets_reshaped = f.expand_dims(pp_offsets, -1)

    # BS x 1 x 3
    row1 = f.concatenate((focal_lengths_reshaped[..., 0:1, :], zeros, pp_offsets_reshaped[..., 0:1, :]), -1)
    row2 = f.concatenate((zeros, focal_lengths_reshaped[..., 1:2, :], pp_offsets_reshaped[..., 1:2, :]), -1)
    row3 = f.concatenate((zeros, zeros, ones), -1)

    # BS x 3 x 3
    return f.concatenate((row1, row2, row3), -2)


def rot_mat_and_cam_center_to_ext_mat(rotation_mat, camera_center, batch_shape=None, f=None):
    """
    Get extrinsic matrix :math:`\mathbf{E}\in\mathbb{R}^{3×4}` from rotation matrix
    :math:`\mathbf{R}\in\mathbb{R}^{3×3}` and camera centers :math:`\overset{\sim}{\mathbf{C}}\in\mathbb{R}^{3×1}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=175>`_
    page 157, section 6.1, equation 6.11

    :param rotation_mat: Rotation matrix *[batch_shape,3,3]*
    :type rotation_mat: array
    :param camera_center: Camera center *[batch_shape,3,1]*
    :type camera_center: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Extrinsic matrix *[batch_shape,3,4]*
    """

    f = _get_framework(rotation_mat, f=f)

    if batch_shape is None:
        batch_shape = rotation_mat.shape[:-2]

    # shapes as list
    batch_shape = list(batch_shape)

    # num batch dims
    num_batch_dims = len(batch_shape)

    # BS x 3 x 3
    identity = f.tile(f.reshape(f.identity(3), [1] * num_batch_dims + [3, 3]),
                         batch_shape + [1, 1])

    # BS x 3 x 4
    identity_w_cam_center = f.concatenate((identity, -camera_center), -1)

    # BS x 3 x 4
    return f.matmul(rotation_mat, identity_w_cam_center)


def cam_to_pixel_coords(coords_wrt_cam, calib_mat, batch_shape=None, image_dims=None, f=None):
    """
    Get depth scaled homogeneous pixel co-ordinates image :math:`\mathbf{X}_p\in\mathbb{R}^{h×w×3}` from camera-centric
    homogeneous co-ordinates image :math:`\mathbf{X}_c\in\mathbb{R}^{h×w×4}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=173>`_
    page 155, equation 6.3

    :param coords_wrt_cam: Camera-centric homogeneous co-ordinates image *[batch_shape,h,w,4]*
    :type coords_wrt_cam: array
    :param calib_mat: Calibration matrix *[batch_shape,3,3]*
    :type calib_mat: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Depth scaled homogeneous pixel co-ordinates image *[batch_shape,h,w,3]*
    """

    f = _get_framework(coords_wrt_cam, f=f)

    if batch_shape is None:
        batch_shape = coords_wrt_cam.shape[:-3]

    if image_dims is None:
        image_dims = coords_wrt_cam.shape[-3:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x H x W x 3
    coords_wrt_cam = coords_wrt_cam[..., 0:3]

    # BS x H x W x 3
    return _ivy_pg.transform(coords_wrt_cam, calib_mat, batch_shape, image_dims, f=f)


def pixel_to_cam_coords(pixel_coords, inv_calib_mat, batch_shape=None, image_dims=None, dev=None, f=None):
    """
    Get camera-centric homogeneous co-ordinates image :math:`\mathbf{X}_c\in\mathbb{R}^{h×w×4}` from
    depth scaled homogeneous pixel co-ordinates image :math:`\mathbf{X}_p\in\mathbb{R}^{h×w×3}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=173>`_
    page 155, matrix inverse of equation 6.3

    :param pixel_coords: Depth scaled homogeneous pixel co-ordinates image *[batch_shape,h,w,3]*
    :type pixel_coords: array
    :param inv_calib_mat: Inverse calibration matrix *[batch_shape,3,3]*
    :type inv_calib_mat: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Camera-centric homogeneous co-ordinates image *[batch_shape,h,w,4]*
    """

    f = _get_framework(pixel_coords, f=f)

    if batch_shape is None:
        batch_shape = pixel_coords.shape[:-3]

    if image_dims is None:
        image_dims = pixel_coords.shape[-3:-1]

    if dev is None:
        dev = f.get_device(pixel_coords)

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x H x W x 3
    cam_coords = _ivy_pg.transform(pixel_coords, inv_calib_mat, batch_shape, image_dims, f=f)

    # BS x H x W x 4
    return f.concatenate((cam_coords, f.ones(batch_shape + image_dims + [1], dev=dev)), -1)


def world_to_cam_coords(coords_wrt_world, ext_mat, batch_shape=None, image_dims=None, dev=None, f=None):
    """
    Get camera-centric homogeneous co-ordinates image :math:`\mathbf{X}_c\in\mathbb{R}^{h×w×4}` from world-centric
    homogeneous co-ordinates image :math:`\mathbf{X}_w\in\mathbb{R}^{h×w×4}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=174>`_
    page 156, equation 6.6

    :param coords_wrt_world: World-centric homogeneous co-ordinates image *[batch_shape,h,w,4]*
    :type coords_wrt_world: array
    :param ext_mat: Extrinsic matrix *[batch_shape,3,4]*
    :type ext_mat: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Camera-centric homogeneous co-ordinates image *[batch_shape,h,w,4]*
    """

    f = _get_framework(coords_wrt_world, f=f)

    if batch_shape is None:
        batch_shape = coords_wrt_world.shape[:-3]

    if image_dims is None:
        image_dims = coords_wrt_world.shape[-3:-1]

    if dev is None:
        dev = f.get_device(coords_wrt_world)

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x H x W x 3
    cam_coords = _ivy_pg.transform(coords_wrt_world, ext_mat, batch_shape, image_dims, f=f)

    # BS x H x W x 4
    return f.concatenate((cam_coords, f.ones(batch_shape + image_dims + [1], dev=dev)), -1)


def cam_to_world_coords(coords_wrt_cam, inv_ext_mat, batch_shape=None, image_dims=None, dev=None, f=None):
    """
    Get world-centric homogeneous co-ordinates image :math:`\mathbf{X}_w\in\mathbb{R}^{h×w×4}` from camera-centric
    homogeneous co-ordinates image :math:`\mathbf{X}_c\in\mathbb{R}^{h×w×4}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=174>`_
    matrix inverse of page 156, equation 6.6

    :param coords_wrt_cam: Camera-centric homogeneous co-ordinates image *[batch_shape,h,w,4]*
    :type coords_wrt_cam: array
    :param inv_ext_mat: Inverse extrinsic matrix *[batch_shape,3,4]*
    :type inv_ext_mat: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: World-centric homogeneous co-ordinates image *[batch_shape,h,w,4]*
    """

    f = _get_framework(coords_wrt_cam, f=f)

    if batch_shape is None:
        batch_shape = coords_wrt_cam.shape[:-3]

    if image_dims is None:
        image_dims = coords_wrt_cam.shape[-3:-1]

    if dev is None:
        dev = f.get_device(coords_wrt_cam)

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x H x W x 3
    world_coords = _ivy_pg.transform(coords_wrt_cam, inv_ext_mat, batch_shape, image_dims, f=f)

    # BS x H x W x 4
    return f.concatenate((world_coords, f.ones(batch_shape + image_dims + [1], dev=dev)), -1)


def world_to_pixel_coords(coords_wrt_world, full_mat, batch_shape=None, image_dims=None, f=None):
    """
    Get depth scaled homogeneous pixel co-ordinates image :math:`\mathbf{X}_p\in\mathbb{R}^{h×w×3}` from world-centric
    homogeneous co-ordinates image :math:`\mathbf{X}_w\in\mathbb{R}^{h×w×4}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=173>`_
    combination of page 156, equation 6.6, and page 155, equation 6.3

    :param coords_wrt_world: World-centric homogeneous co-ordinates image *[batch_shape,h,w,4]*
    :type coords_wrt_world: array
    :param full_mat: Full projection matrix *[batch_shape,3,4]*
    :type full_mat: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Depth scaled homogeneous pixel co-ordinates image *[batch_shape,h,w,3]*
    """

    f = _get_framework(coords_wrt_world, f=f)

    if batch_shape is None:
        batch_shape = coords_wrt_world.shape[:-3]

    if image_dims is None:
        image_dims = coords_wrt_world.shape[-3:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x H x W x 3
    return _ivy_pg.transform(coords_wrt_world, full_mat, batch_shape, image_dims, f=f)


def pixel_to_world_coords(pixel_coords, inv_full_mat, batch_shape=None, image_dims=None, dev=None, f=None):
    """
    Get world-centric homogeneous co-ordinates image :math:`\mathbf{X}_w\in\mathbb{R}^{h×w×4}` from depth scaled
    homogeneous pixel co-ordinates image :math:`\mathbf{X}_p\in\mathbb{R}^{h×w×3}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=173>`_
    combination of page 155, matrix inverse of equation 6.3, and matrix inverse of page 156, equation 6.6

    :param pixel_coords: Depth scaled homogeneous pixel co-ordinates image: *[batch_shape,h,w,3]*
    :type pixel_coords: array
    :param inv_full_mat: Inverse full projection matrix *[batch_shape,3,4]*
    :type inv_full_mat: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: World-centric homogeneous co-ordinates image *[batch_shape,h,w,4]*
    """

    f = _get_framework(pixel_coords, f=f)

    if batch_shape is None:
        batch_shape = pixel_coords.shape[:-3]

    if image_dims is None:
        image_dims = pixel_coords.shape[-3:-1]

    if dev is None:
        dev = f.get_device(pixel_coords)

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x H x W x 4
    pixel_coords = f.concatenate((pixel_coords, f.ones(batch_shape + image_dims + [1], dev=dev)), -1)

    # BS x H x W x 3
    world_coords = _ivy_pg.transform(pixel_coords, inv_full_mat, batch_shape, image_dims, f=f)

    # BS x H x W x 4
    return f.concatenate((world_coords, f.ones(batch_shape + image_dims + [1], dev=dev)), -1)


def pixel_coords_to_world_ray_vectors(pixel_coords, inv_full_mat, camera_center=None, batch_shape=None, image_dims=None,
                                      f=None):
    """
    Calculate world-centric ray vector image :math:`\mathbf{RV}\in\mathbb{R}^{h×w×3}` from homogeneous pixel co-ordinate
    image :math:`\mathbf{X}_p\in\mathbb{R}^{h×w×3}`. Each ray vector :math:`\mathbf{rv}_{i,j}\in\mathbb{R}^{3}` is
    represented as a unit vector from the camera center :math:`\overset{\sim}{\mathbf{C}}\in\mathbb{R}^{3×1}`, in the
    world frame. Co-ordinates :math:`\mathbf{x}_{i,j}\in\mathbb{R}^{3}` along the world ray can then be parameterized as
    :math:`\mathbf{x}_{i,j}=\overset{\sim}{\mathbf{C}} + λ\mathbf{rv}_{i,j}`, where :math:`λ` is a scalar who's
    magnitude dictates the position of the world co-ordinate along the world ray.

    :param pixel_coords: Homogeneous pixel co-ordinates image *[batch_shape,h,w,3]*
    :type pixel_coords: array
    :param inv_full_mat: Inverse full projection matrix *[batch_shape,3,4]*
    :type inv_full_mat: array
    :param camera_center: Camera centers, inferred from inv_full_mat if None *[batch_shape,3,1]*
    :type camera_center: array, optional
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: World ray vectors *[batch_shape,h,w,4]*
    """

    f = _get_framework(pixel_coords, f=f)

    if batch_shape is None:
        batch_shape = pixel_coords.shape[:-3]

    if image_dims is None:
        image_dims = pixel_coords.shape[-3:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    if camera_center is None:
        camera_center = inv_ext_mat_to_camera_center(inv_full_mat, f=f)

    # BS x 1 x 1 x 3
    camera_centers_reshaped = f.reshape(camera_center, batch_shape + [1, 1, 3])

    # BS x H x W x 3
    vectors = pixel_to_world_coords(pixel_coords, inv_full_mat, batch_shape, image_dims)[..., 0:3] \
              - camera_centers_reshaped

    # BS x H x W x 3
    return vectors / (f.reduce_sum(vectors ** 2, -1, keepdims=True) ** 0.5 + MIN_DENOMINATOR)


def bilinearly_interpolate_image(image, sampling_pixel_coords, batch_shape=None, image_dims=None,
                                 f=None):
    """
    Bilinearly interpolate image :math:`\mathbf{X}\in\mathbb{R}^{h×w×d}` at sampling pixel locations
    :math:`\mathbf{S}\in\mathbb{R}^{h×w×2}`, to return interpolated image :math:`\mathbf{X}_I\in\mathbb{R}^{h×w×d}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Bilinear_interpolation>`_

    :param image: Image to be interpolated *[batch_shape,h,w,d]*
    :type image: array
    :param sampling_pixel_coords: Pixel co-ordinates to sample the image at *[batch_shape,h,w,2]*
    :type sampling_pixel_coords: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Interpolated image *[batch_shape,h,w,d]*
    """

    f = _get_framework(image, f=f)

    if batch_shape is None:
        batch_shape = image.shape[:-3]

    if image_dims is None:
        image_dims = image.shape[-3:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # batch shape product
    batch_shape_product = _reduce(_mul, batch_shape, 1)

    # prod(BS) x H x W x D
    uniform_values_flat = f.reshape(image, [batch_shape_product] + image_dims + [-1])

    # prod(BS) x H x W x 2
    sampling_pixel_coords_flat = f.reshape(sampling_pixel_coords, [batch_shape_product] + image_dims + [2])

    # prod(BS) x H x W x D
    interpolation_flat = f.bilinear_resample(uniform_values_flat, sampling_pixel_coords_flat, [batch_shape_product],
                                                image_dims)

    # BS x H x W x D
    return f.reshape(interpolation_flat, batch_shape + image_dims + [-1])


# noinspection PyUnusedLocal
def inv_ext_mat_to_camera_center(inv_ext_mat, f=None):
    """
    Compute camera center :math:`\overset{\sim}{\mathbf{C}}\in\mathbb{R}^{3×1}` from camera extrinsic matrix
    :math:`\mathbf{E}\in\mathbb{R}^{3×4}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=174>`_
    matrix inverse of page 156, equation 6.6

    :param inv_ext_mat: Inverse extrinsic matrix *[batch_shape,3,4]*
    :type inv_ext_mat: array
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Camera center *[batch_shape,3,1]*
    """

    # BS x 3 x 1
    return inv_ext_mat[..., -1:]


def calib_and_ext_to_full_mat(calib_mat, ext_mat, f=None):
    """
    Compute full projection matrix :math:`\mathbf{P}\in\mathbb{R}^{3×4}` from calibration
    :math:`\mathbf{K}\in\mathbb{R}^{3×3}` and extrinsic matrix :math:`\mathbf{E}\in\mathbb{R}^{3×4}`.\n

    :param calib_mat: Calibration matrix *[batch_shape,3,3]*
    :type calib_mat: array
    :param ext_mat: Extrinsic matrix *[batch_shape,3,4]*
    :type ext_mat: array
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Full projection matrix *[batch_shape,3,4]*
    """

    f = _get_framework(calib_mat, f=f)

    # BS x 3 x 4
    return f.matmul(calib_mat, ext_mat)


def cam_to_sphere_coords(cam_coords, batch_shape=None, image_dims=None, f=None):
    """
    Convert camera-centric homogeneous cartesian co-ordinates image :math:`\mathbf{X}_c\in\mathbb{R}^{h×w×4}` to
    camera-centric ego-sphere polar co-ordinates image :math:`\mathbf{S}_c\in\mathbb{R}^{h×w×3}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_

    :param cam_coords: Camera-centric homogeneous cartesian co-ordinates image *[batch_shape,h,w,4]*
    :type cam_coords: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Camera-centric ego-sphere polar co-ordinates image *[batch_shape,h,w,3]*
    """

    f = _get_framework(cam_coords, f=f)

    if batch_shape is None:
        batch_shape = cam_coords.shape[:-3]

    if image_dims is None:
        image_dims = cam_coords.shape[-3:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x H x W x 3
    return f.reshape(
        _ivy_mec.cartesian_to_polar_coords(f.reshape(cam_coords[..., 0:3], (-1, 3)), f=f),
        batch_shape + image_dims + [3])


def pixel_to_sphere_coords(pixel_coords, inv_calib_mat, batch_shape=None, image_dims=None, f=None):
    """
    Convert depth scaled homogeneous pixel co-ordinates image :math:`\mathbf{X}_p\in\mathbb{R}^{h×w×3}` to
    camera-centric ego-sphere polar co-ordinates image :math:`\mathbf{S}_c\in\mathbb{R}^{h×w×3}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_

    :param pixel_coords: Depth scaled homogeneous pixel co-ordinates image *[batch_shape,h,w,3]*
    :type pixel_coords: array
    :param inv_calib_mat: Inverse calibration matrix *[batch_shape,3,3]*
    :type inv_calib_mat: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Camera-centric ego-sphere polar co-ordinates image *[batch_shape,h,w,3]*
    """

    f = _get_framework(pixel_coords, f=f)

    if batch_shape is None:
        batch_shape = pixel_coords.shape[:-3]

    if image_dims is None:
        image_dims = pixel_coords.shape[-3:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x H x W x 4
    cam_coords = pixel_to_cam_coords(pixel_coords, inv_calib_mat, batch_shape, image_dims, f=f)

    # BS x H x W x 3
    return cam_to_sphere_coords(cam_coords, batch_shape, image_dims)


def angular_pixel_to_sphere_coords(angular_pixel_coords, pixels_per_degree, f=None):
    """
    Convert angular pixel co-ordinates image :math:`\mathbf{A}_p\in\mathbb{R}^{h×w×3}` to camera-centric ego-sphere
    polar co-ordinates image :math:`\mathbf{S}_c\in\mathbb{R}^{h×w×3}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_

    :param angular_pixel_coords: Angular pixel co-ordinates image *[batch_shape,h,w,2]*
    :type angular_pixel_coords: array
    :param pixels_per_degree: Number of pixels per angular degree
    :type pixels_per_degree: float
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Camera-centric ego-sphere polar co-ordinates image *[batch_shape,h,w,3]*
    """

    f = _get_framework(angular_pixel_coords, f=f)

    # BS x H x W x 1
    sphere_x_coords = angular_pixel_coords[..., 0:1]
    sphere_y_coords = angular_pixel_coords[..., 1:2]
    radius_values = angular_pixel_coords[..., 2:3]

    sphere_x_angle_coords_in_degs = sphere_x_coords/(pixels_per_degree + MIN_DENOMINATOR) - 180
    sphere_y_angle_coords_in_degs = sphere_y_coords/(pixels_per_degree + MIN_DENOMINATOR)

    # BS x H x W x 2
    sphere_angle_coords_in_degs = f.concatenate((sphere_x_angle_coords_in_degs, sphere_y_angle_coords_in_degs),
                                                   -1)
    sphere_angle_coords = sphere_angle_coords_in_degs * np.pi / 180

    # BS x H x W x 3
    return f.concatenate((sphere_angle_coords, radius_values), -1)


def sphere_to_cam_coords(sphere_coords, batch_shape=None, image_dims=None, dev=None, f=None):
    """
    Convert camera-centric ego-sphere polar co-ordinates image :math:`\mathbf{S}_c\in\mathbb{R}^{h×w×3}` to
    camera-centric homogeneous cartesian co-ordinates image :math:`\mathbf{X}_c\in\mathbb{R}^{h×w×4}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_

    :param sphere_coords: Camera-centric ego-sphere polar co-ordinates image *[batch_shape,h,w,3]*
    :type sphere_coords: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: *Camera-centric homogeneous cartesian co-ordinates image *[batch_shape,h,w,4]*
    """

    f = _get_framework(sphere_coords, f=f)

    if batch_shape is None:
        batch_shape = sphere_coords.shape[:-3]

    if image_dims is None:
        image_dims = sphere_coords.shape[-3:-1]

    if dev is None:
        dev = f.get_device(sphere_coords)

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x H x W x 3
    cam_coords_not_homo = _ivy_mec.polar_to_cartesian_coords(sphere_coords, f=f)

    # BS x H x W x 4
    return f.concatenate((cam_coords_not_homo, f.ones(batch_shape + image_dims + [1], dev=dev)), -1)


def sphere_to_pixel_coords(sphere_coords, calib_mat, batch_shape=None, image_dims=None, f=None):
    """
    Convert camera-centric ego-sphere polar co-ordinates image :math:`\mathbf{S}_c\in\mathbb{R}^{h×w×3}` to depth scaled
    homogeneous pixel co-ordinates image :math:`\mathbf{X}_p\in\mathbb{R}^{h×w×3}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_

    :param sphere_coords: Camera-centric ego-sphere polar co-ordinates image *[batch_shape,h,w,3]*
    :type sphere_coords: array
    :param calib_mat: Calibration matrix *[batch_shape,3,3]*
    :type calib_mat: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Depth scaled homogeneous pixel co-ordinates image *[batch_shape,h,w,3]*
    """

    f = _get_framework(sphere_coords, f=f)

    if batch_shape is None:
        batch_shape = sphere_coords.shape[:-3]

    if image_dims is None:
        image_dims = sphere_coords.shape[-3:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x H x W x 4
    cam_coords = sphere_to_cam_coords(sphere_coords, batch_shape, image_dims, f=f)

    # BS x H x W x 3
    return cam_to_pixel_coords(cam_coords, calib_mat, batch_shape, image_dims, f=f)


def sphere_to_angular_pixel_coords(sphere_coords, pixels_per_degree, f=None):
    """
    Convert camera-centric ego-sphere polar co-ordinates image :math:`\mathbf{S}_c\in\mathbb{R}^{h×w×3}` to angular
    pixel co-ordinates image :math:`\mathbf{A}_p\in\mathbb{R}^{h×w×3}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_

    :param sphere_coords: Camera-centric ego-sphere polar co-ordinates image *[batch_shape,h,w,3]*
    :type sphere_coords: array
    :param pixels_per_degree: Number of pixels per angular degree
    :type pixels_per_degree: float
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Angular pixel co-ordinates image *[batch_shape,h,w,3]*
    """

    f = _get_framework(sphere_coords, f=f)

    # BS x H x W x 1
    sphere_radius_vals = sphere_coords[..., -1:]

    # BS x H x W x 2
    sphere_angle_coords = sphere_coords[..., 0:2]

    # BS x H x W x 2
    sphere_angle_coords_in_degs = sphere_angle_coords * 180 / np.pi

    # BS x H x W x 1
    sphere_x_coords = (sphere_angle_coords_in_degs[..., 0:1] + 180) % 360 * pixels_per_degree
    sphere_y_coords = sphere_angle_coords_in_degs[..., 1:2] % 180 * pixels_per_degree

    # BS x H x W x 3
    return f.concatenate((sphere_x_coords, sphere_y_coords, sphere_radius_vals), -1)


# Camera Geometry Object Functions #
# ---------------------------------#


def persp_angles_and_pp_offsets_to_intrinsics_object(persp_angles, pp_offsets, image_dims, batch_shape=None,
                                                     f=None):
    """
    Create camera intrinsics object from perspective angles :math:`θ_x, θ_y`, principal-point offsets :math:`p_x, p_y`
    and image dimensions [height, width].

    :param persp_angles: Perspective angles *[batch_shape,2]*
    :type persp_angles: array
    :param pp_offsets: Principal-point offsets *[batch_shape,2]*
    :type pp_offsets: array
    :param image_dims: Image dimensions.
    :type image_dims: sequence of ints
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Camera intrinsics object.
    """

    f = _get_framework(persp_angles, f=f)

    if batch_shape is None:
        batch_shape = persp_angles.shape[:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x 2
    focal_lengths = persp_angles_to_focal_lengths(persp_angles, image_dims)

    # BS x 3 x 3
    calib_mat = focal_lengths_and_pp_offsets_to_calib_mat(focal_lengths, pp_offsets, batch_shape)

    # BS x 3 x 3
    inv_calib_mat = f.inv(calib_mat)

    # intrinsics object
    intrinsics = _Intrinsics(focal_lengths, persp_angles, pp_offsets, calib_mat, inv_calib_mat)
    return intrinsics


def focal_lengths_and_pp_offsets_to_intrinsics_object(focal_lengths, pp_offsets, image_dims, batch_shape=None, f=None):
    """
    Create camera intrinsics object from focal lengths :math:`f_x, f_y`, principal-point offsets :math:`p_x, p_y`, and
    image dimensions [height, width].

    :param focal_lengths: Focal lengths *[batch_shape,2]*
    :type focal_lengths: array
    :param pp_offsets: Principal-point offsets *[batch_shape,2]*
    :type pp_offsets: array
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Camera intrinsics object
    """

    f = _get_framework(focal_lengths, f=f)

    if batch_shape is None:
        batch_shape = focal_lengths.shape[:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x 2
    persp_angles = focal_lengths_to_persp_angles(focal_lengths, image_dims)

    # BS x 3 x 3
    calib_mat = focal_lengths_and_pp_offsets_to_calib_mat(focal_lengths, pp_offsets, batch_shape)

    # BS x 3 x 3
    inv_calib_mat = f.inv(calib_mat)

    # intrinsics object
    intrinsics = _Intrinsics(focal_lengths, persp_angles, pp_offsets, calib_mat, inv_calib_mat)
    return intrinsics


def ext_mat_and_intrinsics_to_cam_geometry_object(ext_mat, intrinsics, batch_shape=None, dev=None, f=None):
    """
    Create camera geometry object from extrinsic matrix :math:`\mathbf{E}\in\mathbb{R}^{3×4}`, and camera intrinsics
    object.

    :param ext_mat: Extrinsic matrix *[batch_shape,3,4]*
    :type ext_mat: array
    :param intrinsics: camera intrinsics object
    :type intrinsics: camera_intrinsics
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Camera geometry object
    """

    f = _get_framework(ext_mat, f=f)

    if batch_shape is None:
        batch_shape = ext_mat.shape[:-2]

    if dev is None:
        dev = f.get_device(ext_mat)

    # shapes as list
    batch_shape = list(batch_shape)

    # num batch dims
    num_batch_dims = len(batch_shape)

    # BS x 4 x 4
    ext_mat_homo = \
        f.concatenate(
            (ext_mat, f.tile(f.reshape(f.array([0., 0., 0., 1.], dev=dev),
                                             [1] * (num_batch_dims + 1) + [4]),
                                batch_shape + [1, 1])), -2)

    # BS x 4 x 4
    inv_ext_mat_homo = f.inv(ext_mat_homo)

    # BS x 3 x 4
    inv_ext_mat = inv_ext_mat_homo[..., 0:3, :]

    # BS x 3 x 1
    cam_center = inv_ext_mat_to_camera_center(inv_ext_mat)

    # BS x 3 x 3
    Rs = ext_mat[..., 0:3]

    # BS x 3 x 3
    inv_Rs = inv_ext_mat[..., 0:3]

    # extrinsics object
    extrinsics = _Extrinsics(cam_center, Rs, inv_Rs, ext_mat_homo, inv_ext_mat_homo)

    # BS x 3 x 4
    full_mat = calib_and_ext_to_full_mat(intrinsics.calib_mats, ext_mat)

    # BS x 4 x 4
    full_mat_homo = \
        f.concatenate((
            full_mat, f.tile(f.reshape(f.array([0., 0., 0., 1.], dev=dev),
                                             [1] * (num_batch_dims + 1) + [4]),
                                batch_shape + [1, 1])), -2)

    # BS x 4 x 4
    inv_full_mat_homo = f.inv(full_mat_homo)

    # camera geometry object
    cam_geometry = _CameraGeometry(intrinsics, extrinsics, full_mat_homo, inv_full_mat_homo, f=f)
    return cam_geometry


def inv_ext_mat_and_intrinsics_to_cam_geometry_object(inv_ext_mat, intrinsics, batch_shape=None, dev=None, f=None):
    """
    Create camera geometry object from inverse extrinsic matrix :math:`\mathbf{E}^{-1}\in\mathbb{R}^{3×4}`, and camera
    intrinsics object.

    :param inv_ext_mat: Inverse extrinsic matrix *[batch_shape,3,4]*
    :type inv_ext_mat: array
    :param intrinsics: camera intrinsics object
    :type intrinsics: camera_intrinsics
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Camera geometry object
    """

    f = _get_framework(inv_ext_mat, f=f)

    if batch_shape is None:
        batch_shape = inv_ext_mat.shape[:-2]

    if dev is None:
        dev = f.get_device(inv_ext_mat)

    # shapes as list
    batch_shape = list(batch_shape)

    # num batch dims
    num_batch_dims = len(batch_shape)

    # BS x 4 x 4
    inv_ext_mat_homo = \
        f.concatenate((inv_ext_mat, f.tile(
            f.reshape(f.array([0., 0., 0., 1.], dev=dev), [1] * (num_batch_dims + 1) + [4]),
            batch_shape + [1, 1])), -2)

    # BS x 4 x 4
    ext_mat_homo = f.inv(inv_ext_mat_homo)

    # BS x 3 x 4
    ext_mat = ext_mat_homo[..., 0:3, :]

    # BS x 3 x 1
    cam_center = inv_ext_mat_to_camera_center(inv_ext_mat)

    # BS x 3 x 3
    Rs = ext_mat[..., 0:3]

    # BS x 3 x 3
    inv_Rs = inv_ext_mat[..., 0:3]

    # extrinsics object
    extrinsics = _Extrinsics(cam_center, Rs, inv_Rs, ext_mat_homo, inv_ext_mat_homo)

    # BS x 3 x 4
    full_mat = calib_and_ext_to_full_mat(intrinsics.calib_mats, ext_mat)

    # BS x 4 x 4
    full_mat_homo = \
        f.concatenate((
            full_mat, f.tile(f.reshape(f.array([0., 0., 0., 1.], dev=dev),
                                             [1] * (num_batch_dims + 1) + [4]),
                                batch_shape + [1, 1])), -2)

    # BS x 4 x 4
    inv_full_mat_homo = f.inv(full_mat_homo)

    # camera geometry object
    camera_geometry = _CameraGeometry(intrinsics, extrinsics, full_mat_homo, inv_full_mat_homo)
    return camera_geometry
