"""Collection of Single-View-Geometry Functions"""

# global
import ivy as _ivy
import numpy as np
import ivy_mech as _ivy_mec
from operator import mul as _mul
from functools import reduce as _reduce

# local
from ivy_vision import image as _ivy_img
from ivy_vision import projective_geometry as _ivy_pg
from ivy_vision.containers import Intrinsics as _Intrinsics
from ivy_vision.containers import Extrinsics as _Extrinsics
from ivy_vision.containers import CameraGeometry as _CameraGeometry


MIN_DENOMINATOR = 1e-12


def create_uniform_pixel_coords_image(image_dims, batch_shape=None, normalized=False, homogeneous=True, device=None):
    """Create image of homogeneous integer :math:`xy` pixel co-ordinates :math:`\mathbf{X}\in\mathbb{Z}^{h×w×3}`, stored
    as floating point values. The origin is at the top-left corner of the image, with :math:`+x` rightwards, and
    :math:`+y` downwards. The final homogeneous dimension are all ones. In subsequent use of this image, the depth of
    each pixel can be represented using this same homogeneous representation, by simply scaling each 3-vector by the
    depth value. The final dimension therefore always holds the depth value, while the former two dimensions hold depth
    scaled pixel co-ordinates.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=172>`_
    deduction from top of page 154, section 6.1, equation 6.1

    Parameters
    ----------
    image_dims
        Image dimensions.
    batch_shape
        Shape of batch. Assumed no batch dimensions if None. (Default value = None)
    normalized
        Whether to normalize x-y pixel co-ordinates to the range 0-1. Default is False.
    homogeneous
        Whether the pixel co-ordinates should be 3D homogeneous or just 2D. Default is True.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. (Default value = None)

    Returns
    -------
    ret
        Image of homogeneous pixel co-ordinates *[batch_shape,height,width,3]*

    """

    # shapes as lists
    batch_shape = [] if batch_shape is None else batch_shape
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # other shape specs
    num_batch_dims = len(batch_shape)
    flat_shape = [1] * num_batch_dims + image_dims + [2]
    tile_shape = batch_shape + [1] * 3

    # H x W x 1
    pixel_x_coords = _ivy.cast(_ivy.reshape(_ivy.tile(_ivy.arange(image_dims[1], device=device), [image_dims[0]]),
                                            (image_dims[0], image_dims[1], 1)), 'float32')
    if normalized:
        pixel_x_coords = pixel_x_coords / (float(image_dims[1]) + MIN_DENOMINATOR)

    # W x H x 1
    pixel_y_coords_ = _ivy.cast(_ivy.reshape(_ivy.tile(_ivy.arange(image_dims[0], device=device), [image_dims[1]]),
                                             (image_dims[1], image_dims[0], 1)), 'float32')

    # H x W x 1
    pixel_y_coords = _ivy.transpose(pixel_y_coords_, (1, 0, 2))
    if normalized:
        pixel_y_coords = pixel_y_coords / (float(image_dims[0]) + MIN_DENOMINATOR)

    # BS x H x W x 2
    pix_coords = _ivy.tile(_ivy.reshape(_ivy.concat([pixel_x_coords, pixel_y_coords], -1), flat_shape), tile_shape)

    if homogeneous:
        # BS x H x W x 3
        pix_coords = _ivy_mec.make_coordinates_homogeneous(pix_coords)

    # BS x H x W x 2or3
    return pix_coords


def persp_angles_to_focal_lengths(persp_angles, image_dims, device=None):
    """Compute focal lengths :math:`f_x, f_y` from perspective angles :math:`θ_x, θ_y`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=172>`_
    deduction from page 154, section 6.1, figure 6.1

    Parameters
    ----------
    persp_angles
        Perspective angles *[batch_shape,2]*
    image_dims
        Image dimensions.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Focal lengths *[batch_shape,2]*

    """

    if device is None:
        device = _ivy.device(persp_angles)

    # shapes as list
    image_dims = list(image_dims)

    # BS x 2
    return -_ivy.flip(_ivy.cast(_ivy.array(image_dims, device=device), 'float32'), -1) /\
           (2 * _ivy.tan(persp_angles / 2) + MIN_DENOMINATOR)


def focal_lengths_to_persp_angles(focal_lengths, image_dims, device=None):
    """Compute perspective angles :math:`θ_x, θ_y` from focal lengths :math:`f_x, f_y`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=172>`_
    deduction from page 154, section 6.1, figure 6.1

    Parameters
    ----------
    focal_lengths
        batch_shape,2]*
    image_dims
        Image dimensions.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Perspective angles *[batch_shape,2]*

    """

    if device is None:
        device = _ivy.device(focal_lengths)

    # shapes as list
    image_dims = list(image_dims)

    # BS x 2
    return -2 * _ivy.atan(_ivy.flip(_ivy.cast(_ivy.array(image_dims, device=device),
                                              'float32'), -1) / (2 * focal_lengths + MIN_DENOMINATOR))


def focal_lengths_and_pp_offsets_to_calib_mat(focal_lengths, pp_offsets, batch_shape=None, device=None):
    """Compute calibration matrix :math:`\mathbf{K}\in\mathbb{R}^{3×3}` from focal lengths :math:`f_x, f_y` and
    principal-point offsets :math:`p_x, p_y`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=173>`_
    page 155, section 6.1, equation 6.4

    Parameters
    ----------
    focal_lengths
        Focal lengths *[batch_shape,2]*
    pp_offsets
        Principal-point offsets *[batch_shape,2]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Calibration matrix *[batch_shape,3,3]*

    """

    if batch_shape is None:
        batch_shape = focal_lengths.shape[:-1]

    if device is None:
        device = _ivy.device(focal_lengths)

    # shapes as list
    batch_shape = list(batch_shape)

    # BS x 1 x 1
    zeros = _ivy.zeros(batch_shape + [1, 1], device=device)
    ones = _ivy.ones(batch_shape + [1, 1], device=device)

    # BS x 2 x 1
    focal_lengths_reshaped = _ivy.expand_dims(focal_lengths, -1)
    pp_offsets_reshaped = _ivy.expand_dims(pp_offsets, -1)

    # BS x 1 x 3
    row1 = _ivy.concat([focal_lengths_reshaped[..., 0:1, :], zeros, pp_offsets_reshaped[..., 0:1, :]], -1)
    row2 = _ivy.concat([zeros, focal_lengths_reshaped[..., 1:2, :], pp_offsets_reshaped[..., 1:2, :]], -1)
    row3 = _ivy.concat([zeros, zeros, ones], -1)

    # BS x 3 x 3
    return _ivy.concat([row1, row2, row3], -2)


# noinspection PyUnresolvedReferences
def rot_mat_and_cam_center_to_ext_mat(rotation_mat, camera_center, batch_shape=None):
    """Get extrinsic matrix :math:`\mathbf{E}\in\mathbb{R}^{3×4}` from rotation matrix
    :math:`\mathbf{R}\in\mathbb{R}^{3×3}` and camera centers :math:`\overset{\sim}{\mathbf{C}}\in\mathbb{R}^{3×1}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=175>`_
    page 157, section 6.1, equation 6.11

    Parameters
    ----------
    rotation_mat
        Rotation matrix *[batch_shape,3,3]*
    camera_center
        Camera center *[batch_shape,3,1]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)

    Returns
    -------
    ret
        Extrinsic matrix *[batch_shape,3,4]*

    """

    if batch_shape is None:
        batch_shape = rotation_mat.shape[:-2]

    # shapes as list
    batch_shape = list(batch_shape)

    # num batch dims
    num_batch_dims = len(batch_shape)

    # BS x 3 x 3
    identity = _ivy.tile(_ivy.reshape(_ivy.eye(3), [1] * num_batch_dims + [3, 3]),
                         batch_shape + [1, 1])

    # BS x 3 x 4
    identity_w_cam_center = _ivy.concat([identity, -camera_center], -1)

    # BS x 3 x 4
    return _ivy.matmul(rotation_mat, identity_w_cam_center)


def depth_to_ds_pixel_coords(depth, uniform_pixel_coords=None, batch_shape=None, image_dims=None):
    """Get depth scaled homogeneous pixel co-ordinates image :math:`\mathbf{X}_p\in\mathbb{R}^{h×w×3}` from depth image
    :math:`\mathbf{X}_d\in\mathbb{R}^{h×w×1}`.\n

    Parameters
    ----------
    depth
        Depth image *[batch_shape,h,w,1]*
    uniform_pixel_coords
        Image of homogeneous pixel co-ordinates. Created if None. *[batch_shape,h,w,3]* (Default value = None)
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_dims
        Image dimensions. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        Depth scaled homogeneous pixel co-ordinates image *[batch_shape,h,w,3]*

    """

    if batch_shape is None:
        batch_shape = depth.shape[:-3]

    if image_dims is None:
        image_dims = depth.shape[-3:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x H x W x 3
    if uniform_pixel_coords is None:
        uniform_pixel_coords = create_uniform_pixel_coords_image(image_dims, batch_shape, device=_ivy.device(depth))

    # BS x H x W x 3
    return uniform_pixel_coords * depth


def depth_to_radial_depth(depth, inv_calib_mat, uniform_pix_coords=None, batch_shape=None, image_dims=None):
    """Get radial depth image :math:`\mathbf{X}_{rd}\in\mathbb{R}^{hxw×1}` from depth image
    :math:`\mathbf{X}_d\in\mathbb{R}^{hxw×1}`.\n

    Parameters
    ----------
    depth
        Depth scaled homogeneous pixel co-ordinates image *[batch_shape,image_dims,1]*
    inv_calib_mat
        Inverse calibration matrix *[batch_shape,3,3]*
    uniform_pix_coords
        Uniform homogeneous pixel co-ordinates, constructed if None. *[batch_shape,image_dims,3]* (Default value = None)
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_dims
        Image shape. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        Radial depth image *[batch_shape,image_dims,1]*

    """

    if batch_shape is None:
        batch_shape = inv_calib_mat.shape[:-2]
    num_batch_dims = len(batch_shape)

    if image_dims is None:
        image_dims = depth.shape[num_batch_dims:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x H x W x 3
    if uniform_pix_coords is None:
        uniform_pix_coords = create_uniform_pixel_coords_image(image_dims, batch_shape, device=_ivy.device(depth))
    ds_pixel_coords = uniform_pix_coords * depth

    # BS x H x W x 1
    return ds_pixel_coords_to_radial_depth(ds_pixel_coords, inv_calib_mat, batch_shape, image_dims)


def ds_pixel_coords_to_radial_depth(ds_pixel_coords, inv_calib_mat, batch_shape=None, image_shape=None):
    """Get radial depth image :math:`\mathbf{X}_{rd}\in\mathbb{R}^{is×1}` from depth scaled homogeneous pixel
    co-ordinates image :math:`\mathbf{X}_p\in\mathbb{R}^{is×3}`.\n

    Parameters
    ----------
    ds_pixel_coords
        Depth scaled homogeneous pixel co-ordinates image *[batch_shape,image_shape,1]*
    inv_calib_mat
        Inverse calibration matrix *[batch_shape,3,3]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image shape. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        Radial depth image *[batch_shape,image_shape,1]*

    """

    if batch_shape is None:
        batch_shape = inv_calib_mat.shape[:-2]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        image_shape = ds_pixel_coords.shape[num_batch_dims:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    # BS x IS x 3
    cam_coords = ds_pixel_to_cam_coords(ds_pixel_coords, inv_calib_mat, batch_shape, image_shape)[..., 0:3]

    # BS x IS x 1
    return _ivy.sum(cam_coords**2, -1, keepdims=True)**0.5


def cam_to_ds_pixel_coords(coords_wrt_cam, calib_mat, batch_shape=None, image_shape=None):
    """Get depth scaled homogeneous pixel co-ordinates image :math:`\mathbf{X}_p\in\mathbb{R}^{is×3}` from camera-centric
    homogeneous co-ordinates image :math:`\mathbf{X}_c\in\mathbb{R}^{is×4}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=173>`_
    page 155, equation 6.3

    Parameters
    ----------
    coords_wrt_cam
        Camera-centric homogeneous co-ordinates image *[batch_shape,image_shape,4]*
    calib_mat
        Calibration matrix *[batch_shape,3,3]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image shape. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        Depth scaled homogeneous pixel co-ordinates image *[batch_shape,image_shape,3]*

    """

    if batch_shape is None:
        batch_shape = calib_mat.shape[:-2]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        image_shape = coords_wrt_cam.shape[num_batch_dims:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    # BS x IS x 3
    coords_wrt_cam = coords_wrt_cam[..., 0:3]

    # BS x IS x 3
    return _ivy_pg.transform(coords_wrt_cam, calib_mat, batch_shape, image_shape)


def cam_coords_to_depth(coords_wrt_cam, calib_mat, batch_shape=None, image_shape=None):
    """Get depth image :math:`\mathbf{X}_p\in\mathbb{R}^{is×1}` from camera-centric
    homogeneous co-ordinates image :math:`\mathbf{X}_c\in\mathbb{R}^{is×4}`.\n

    Parameters
    ----------
    coords_wrt_cam
        Camera-centric homogeneous co-ordinates image *[batch_shape,image_shape,4]*
    calib_mat
        Calibration matrix *[batch_shape,3,3]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image shape. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        Depth image *[batch_shape,image_shape,1]*

    """
    # BS x IS x 1
    return cam_to_ds_pixel_coords(coords_wrt_cam, calib_mat, batch_shape, image_shape)[..., -1:]


def ds_pixel_to_cam_coords(ds_pixel_coords, inv_calib_mat, batch_shape=None, image_shape=None, device=None):
    """Get camera-centric homogeneous co-ordinates image :math:`\mathbf{X}_c\in\mathbb{R}^{is×4}` from
    depth scaled homogeneous pixel co-ordinates image :math:`\mathbf{X}_p\in\mathbb{R}^{is×3}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=173>`_
    page 155, matrix inverse of equation 6.3

    Parameters
    ----------
    ds_pixel_coords
        Depth scaled homogeneous pixel co-ordinates image *[batch_shape,image_shap,3]*
    inv_calib_mat
        Inverse calibration matrix *[batch_shape,3,3]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image dimensions. Inferred from inputs in None. (Default value = None)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Camera-centric homogeneous co-ordinates image *[batch_shape,image_shape,4]*

    """

    if batch_shape is None:
        batch_shape = inv_calib_mat.shape[:-2]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        image_shape = ds_pixel_coords.shape[num_batch_dims:-1]

    if device is None:
        device = _ivy.device(ds_pixel_coords)

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    # BS x IS x 3
    cam_coords = _ivy_pg.transform(ds_pixel_coords, inv_calib_mat, batch_shape, image_shape)

    # BS x IS x 4
    return _ivy_mec.make_coordinates_homogeneous(cam_coords, batch_shape + image_shape)


def depth_to_cam_coords(depth, inv_calib_mat, uniform_pixel_coords=None, batch_shape=None, image_dims=None,
                        device=None):
    """Get camera-centric homogeneous co-ordinates image :math:`\mathbf{X}_c\in\mathbb{R}^{hxw×4}` from
    depth image :math:`\mathbf{X}_p\in\mathbb{R}^{hxw×1}`.\n

    Parameters
    ----------
    depth
        Depth image *[batch_shape,h,w,1]*
    inv_calib_mat
        Inverse calibration matrix *[batch_shape,3,3]*
    uniform_pixel_coords
        Image of homogeneous pixel co-ordinates. Created if None. *[batch_shape,h,w,3]* (Default value = None)
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_dims
        Image dimensions. Inferred from inputs in None. (Default value = None)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Camera-centric homogeneous co-ordinates image *[batch_shape,h,w,4]*

    """

    if batch_shape is None:
        batch_shape = inv_calib_mat.shape[:-2]

    if image_dims is None:
        image_dims = depth.shape[-3:-1]

    if device is None:
        device = _ivy.device(depth)

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x H x W x 3
    ds_pixel_coords = depth_to_ds_pixel_coords(depth, uniform_pixel_coords, batch_shape, image_dims)

    # BS x H x W x 4
    return ds_pixel_to_cam_coords(ds_pixel_coords, inv_calib_mat, batch_shape, image_dims, device)


def world_to_cam_coords(coords_wrt_world, ext_mat, batch_shape=None, image_shape=None, device=None):
    """Get camera-centric homogeneous co-ordinates image :math:`\mathbf{X}_c\in\mathbb{R}^{is×4}` from world-centric
    homogeneous co-ordinates image :math:`\mathbf{X}_w\in\mathbb{R}^{is×4}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=174>`_
    page 156, equation 6.6

    Parameters
    ----------
    coords_wrt_world
        World-centric homogeneous co-ordinates image *[batch_shape,image_shape,4]*
    ext_mat
        Extrinsic matrix *[batch_shape,3,4]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image shape. Inferred from inputs in None. (Default value = None)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Camera-centric homogeneous co-ordinates image *[batch_shape,image_shape,4]*

    """

    if batch_shape is None:
        batch_shape = ext_mat.shape[:-2]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        image_shape = coords_wrt_world.shape[num_batch_dims:-1]

    if device is None:
        device = _ivy.device(coords_wrt_world)

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    # BS x IS x 3
    cam_coords = _ivy_pg.transform(coords_wrt_world, ext_mat, batch_shape, image_shape)

    # BS x IS x 4
    return _ivy.concat([cam_coords, _ivy.ones(batch_shape + image_shape + [1], device=device)], -1)


def cam_to_world_coords(coords_wrt_cam, inv_ext_mat, batch_shape=None, image_shape=None, device=None):
    """Get world-centric homogeneous co-ordinates image :math:`\mathbf{X}_w\in\mathbb{R}^{is×4}` from camera-centric
    homogeneous co-ordinates image :math:`\mathbf{X}_c\in\mathbb{R}^{is×4}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=174>`_
    matrix inverse of page 156, equation 6.6

    Parameters
    ----------
    coords_wrt_cam
        Camera-centric homogeneous co-ordinates image *[batch_shape,image_shape,4]*
    inv_ext_mat
        Inverse extrinsic matrix *[batch_shape,3,4]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image shape. Inferred from inputs in None. (Default value = None)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None. (Default value = None)

    Returns
    -------
    ret
        World-centric homogeneous co-ordinates image *[batch_shape,image_shape,4]*

    """

    if batch_shape is None:
        batch_shape = inv_ext_mat.shape[:-2]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        image_shape = coords_wrt_cam.shape[num_batch_dims:-1]

    if device is None:
        device = _ivy.device(coords_wrt_cam)

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    # BS x IS x 3
    world_coords = _ivy_pg.transform(coords_wrt_cam, inv_ext_mat, batch_shape, image_shape)

    # BS x IS x 4
    return _ivy.concat([world_coords, _ivy.ones(batch_shape + image_shape + [1], device=device)], -1)


def world_to_ds_pixel_coords(coords_wrt_world, full_mat, batch_shape=None, image_shape=None):
    """Get depth scaled homogeneous pixel co-ordinates image :math:`\mathbf{X}_p\in\mathbb{R}^{is×3}` from world-centric
    homogeneous co-ordinates image :math:`\mathbf{X}_w\in\mathbb{R}^{is×4}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=173>`_
    combination of page 156, equation 6.6, and page 155, equation 6.3

    Parameters
    ----------
    coords_wrt_world
        World-centric homogeneous co-ordinates image *[batch_shape,image_shape,4]*
    full_mat
        Full projection matrix *[batch_shape,3,4]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image shape. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        Depth scaled homogeneous pixel co-ordinates image *[batch_shape,image_shape,3]*

    """

    if batch_shape is None:
        batch_shape = full_mat.shape[:-2]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        image_shape = coords_wrt_world.shape[num_batch_dims:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    # BS x IS x 3
    return _ivy_pg.transform(coords_wrt_world, full_mat, batch_shape, image_shape)


def world_coords_to_depth(coords_wrt_world, full_mat, batch_shape=None, image_shape=None):
    """Get depth image :math:`\mathbf{X}_d\in\mathbb{R}^{is×1}` from world-centric
    homogeneous co-ordinates image :math:`\mathbf{X}_w\in\mathbb{R}^{is×4}`.\n

    Parameters
    ----------
    coords_wrt_world
        World-centric homogeneous co-ordinates image *[batch_shape,image_shape,4]*
    full_mat
        Full projection matrix *[batch_shape,3,4]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image shape. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        Depth image *[batch_shape,image_shape,1]*

    """
    # BS x IS x 1
    return world_to_ds_pixel_coords(coords_wrt_world, full_mat, batch_shape, image_shape)[..., -1:]


def ds_pixel_to_world_coords(ds_pixel_coords, inv_full_mat, batch_shape=None, image_shape=None):
    """Get world-centric homogeneous co-ordinates image :math:`\mathbf{X}_w\in\mathbb{R}^{is×4}` from depth scaled
    homogeneous pixel co-ordinates image :math:`\mathbf{X}_p\in\mathbb{R}^{is×3}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=173>`_
    combination of page 155, matrix inverse of equation 6.3, and matrix inverse of page 156, equation 6.6

    Parameters
    ----------
    ds_pixel_coords
        Depth scaled homogeneous pixel co-ordinates image: *[batch_shape,image_shape,3]*
    inv_full_mat
        Inverse full projection matrix *[batch_shape,3,4]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image shape. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        World-centric homogeneous co-ordinates image *[batch_shape,image_shape,4]*

    """

    if batch_shape is None:
        batch_shape = inv_full_mat.shape[:-2]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        image_shape = ds_pixel_coords.shape[num_batch_dims:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    # BS x IS x 4
    ds_pixel_coords = _ivy_mec.make_coordinates_homogeneous(ds_pixel_coords, batch_shape)

    # BS x IS x 3
    world_coords = _ivy_pg.transform(ds_pixel_coords, inv_full_mat, batch_shape, image_shape)

    # BS x IS x 4
    return _ivy_mec.make_coordinates_homogeneous(world_coords, batch_shape + image_shape)


def depth_to_world_coords(depth, inv_full_mat, uniform_pixel_coords=None, batch_shape=None, image_dims=None):
    """Get world-centric homogeneous co-ordinates image :math:`\mathbf{X}_w\in\mathbb{R}^{hxw×4}` from depth image
    :math:`\mathbf{X}_d\in\mathbb{R}^{hxw×1}`.\n

    Parameters
    ----------
    depth
        Depth image: *[batch_shape,h,w,1]*
    inv_full_mat
        Inverse full projection matrix *[batch_shape,3,4]*
    uniform_pixel_coords
        Image of homogeneous pixel co-ordinates. Created if None. *[batch_shape,h,w,3]* (Default value = None)
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_dims
        Image shape. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        World-centric homogeneous co-ordinates image *[batch_shape,h,w,4]*

    """

    if batch_shape is None:
        batch_shape = depth.shape[:-3]

    if image_dims is None:
        image_dims = depth.shape[-3:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x H x W x 3
    ds_pixel_coords = depth_to_ds_pixel_coords(depth, uniform_pixel_coords, batch_shape, image_dims)

    # BS x H x W x 4
    return ds_pixel_to_world_coords(ds_pixel_coords, inv_full_mat, batch_shape, image_dims)


def pixel_coords_to_world_ray_vectors(inv_full_mat, pixel_coords=None, camera_center=None, batch_shape=None,
                                      image_shape=None):
    """Calculate world-centric ray vector image :math:`\mathbf{RV}\in\mathbb{R}^{is×3}` from homogeneous pixel co-ordinate
    image :math:`\mathbf{X}_p\in\mathbb{R}^{is×3}`. Each ray vector :math:`\mathbf{rv}_{i,j}\in\mathbb{R}^{3}` is
    represented as a unit vector from the camera center :math:`\overset{\sim}{\mathbf{C}}\in\mathbb{R}^{3×1}`, in the
    world frame. Co-ordinates :math:`\mathbf{x}_{i,j}\in\mathbb{R}^{3}` along the world ray can then be parameterized as
    :math:`\mathbf{x}_{i,j}=\overset{\sim}{\mathbf{C}} + λ\mathbf{rv}_{i,j}`, where :math:`λ` is a scalar who's
    magnitude dictates the position of the world co-ordinate along the world ray.

    Parameters
    ----------
    inv_full_mat
        Inverse full projection matrix *[batch_shape,3,4]*
    pixel_coords
        Homogeneous pixel co-ordinates image, created uniformly if None. *[batch_shape,image_shape,3]* (Default value = None)
    camera_center
        Camera centers, inferred from inv_full_mat if None *[batch_shape,3,1]* (Default value = None)
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image shape. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        World ray vectors *[batch_shape,image_shape,3]*

    """

    if batch_shape is None:
        batch_shape = inv_full_mat.shape[:-2]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        if pixel_coords is None:
            raise Exception('if pixel_coords is not specified, image_shape must be specified when calling'
                            'pixel_coords_to_world_ray_vectors')
        image_shape = pixel_coords.shape[num_batch_dims:-1]
    num_image_dims = len(image_shape)

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    if camera_center is None:
        camera_center = inv_ext_mat_to_camera_center(inv_full_mat)

    if pixel_coords is None:
        pixel_coords = create_uniform_pixel_coords_image(image_shape, batch_shape, device=_ivy.device(inv_full_mat))

    # BS x [1]xNID x 3
    camera_centers_reshaped = _ivy.reshape(camera_center, batch_shape + [1]*num_image_dims + [3])

    # BS x IS x 3
    vectors = ds_pixel_to_world_coords(pixel_coords, inv_full_mat, batch_shape, image_shape)[..., 0:3] \
              - camera_centers_reshaped

    # BS x H x W x 3
    return vectors / (_ivy.sum(vectors ** 2, -1, keepdims=True) ** 0.5 + MIN_DENOMINATOR)


def sphere_coords_to_world_ray_vectors(sphere_coords, inv_rotation_mat, batch_shape=None, image_shape=None):
    """Calculate world-centric ray vector image :math:`\mathbf{RV}\in\mathbb{R}^{is×3}` from camera-centric ego-sphere
    polar co-ordinates image :math:`\mathbf{S}_c\in\mathbb{R}^{is×3}`. Each ray vector
    :math:`\mathbf{rv}_{i,j}\in\mathbb{R}^{3}` is represented as a unit vector from the camera center
    :math:`\overset{\sim}{\mathbf{C}}\in\mathbb{R}^{3×1}`, in the world frame. Co-ordinates
    :math:`\mathbf{x}_{i,j}\in\mathbb{R}^{3}` along the world ray can then be parameterized as
    :math:`\mathbf{x}_{i,j}=\overset{\sim}{\mathbf{C}} + λ\mathbf{rv}_{i,j}`, where :math:`λ` is a scalar who's
    magnitude dictates the position of the world co-ordinate along the world ray.

    Parameters
    ----------
    sphere_coords
        Camera-centric ego-sphere polar co-ordinates image *[batch_shape,image_shape,3]*
    inv_rotation_mat
        Inverse rotation matrix *[batch_shape,3,3]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image dimensions. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        World ray vectors *[batch_shape,image_shape,3]*

    """

    if batch_shape is None:
        batch_shape = inv_rotation_mat.shape[:-2]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        image_shape = sphere_coords.shape[num_batch_dims:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    # BS x IS x 3
    cam_coords = sphere_to_cam_coords(sphere_coords, batch_shape=batch_shape + image_shape)[..., 0:3]
    vectors = _ivy_pg.transform(cam_coords, inv_rotation_mat, batch_shape, image_shape)
    return vectors / (_ivy.sum(vectors ** 2, -1, keepdims=True) ** 0.5 + MIN_DENOMINATOR)


def bilinearly_interpolate_image(image, sampling_pixel_coords, batch_shape=None, image_dims=None):
    """Bilinearly interpolate image :math:`\mathbf{X}\in\mathbb{R}^{h×w×d}` at sampling pixel locations
    :math:`\mathbf{S}\in\mathbb{R}^{h×w×2}`, to return interpolated image :math:`\mathbf{X}_I\in\mathbb{R}^{h×w×d}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Bilinear_interpolation>`_

    Parameters
    ----------
    image
        Image to be interpolated *[batch_shape,h,w,d]*
    sampling_pixel_coords
        Pixel co-ordinates to sample the image at *[batch_shape,h,w,2]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_dims
        Image dimensions. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        Interpolated image *[batch_shape,h,w,d]*

    """

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
    uniform_values_flat = _ivy.reshape(image, [batch_shape_product] + image_dims + [-1])

    # prod(BS) x H x W x 2
    sampling_pixel_coords_flat = _ivy.reshape(sampling_pixel_coords, [batch_shape_product] + image_dims + [2])

    # prod(BS) x H x W x D
    interpolation_flat = _ivy_img.bilinear_resample(uniform_values_flat, sampling_pixel_coords_flat)

    # BS x H x W x D
    return _ivy.reshape(interpolation_flat, batch_shape + image_dims + [-1])


# noinspection PyUnusedLocal
def inv_ext_mat_to_camera_center(inv_ext_mat):
    """Compute camera center :math:`\overset{\sim}{\mathbf{C}}\in\mathbb{R}^{3×1}` from camera extrinsic matrix
    :math:`\mathbf{E}\in\mathbb{R}^{3×4}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=174>`_
    matrix inverse of page 156, equation 6.6

    Parameters
    ----------
    inv_ext_mat
        Inverse extrinsic matrix *[batch_shape,3,4]*

    Returns
    -------
    ret
        Camera center *[batch_shape,3,1]*

    """

    # BS x 3 x 1
    return inv_ext_mat[..., -1:]


def calib_and_ext_to_full_mat(calib_mat, ext_mat):
    """Compute full projection matrix :math:`\mathbf{P}\in\mathbb{R}^{3×4}` from calibration
    :math:`\mathbf{K}\in\mathbb{R}^{3×3}` and extrinsic matrix :math:`\mathbf{E}\in\mathbb{R}^{3×4}`.\n

    Parameters
    ----------
    calib_mat
        Calibration matrix *[batch_shape,3,3]*
    ext_mat
        Extrinsic matrix *[batch_shape,3,4]*

    Returns
    -------
    ret
        Full projection matrix *[batch_shape,3,4]*

    """

    # BS x 3 x 4
    return _ivy.matmul(calib_mat, ext_mat)


def cam_to_sphere_coords(cam_coords, forward_facing_z=True):
    """Convert camera-centric homogeneous cartesian co-ordinates image :math:`\mathbf{X}_c\in\mathbb{R}^{h×w×4}` to
    camera-centric ego-sphere polar co-ordinates image :math:`\mathbf{S}_c\in\mathbb{R}^{h×w×3}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_

    Parameters
    ----------
    cam_coords
        Camera-centric homogeneous cartesian co-ordinates image *[batch_shape,h,w,4]*
    forward_facing_z
        Whether to use reference frame so z is forward facing. Default is False.

    Returns
    -------
    ret
        Camera-centric ego-sphere polar co-ordinates image *[batch_shape,h,w,3]*

    """

    # BS x H x W x 3
    if forward_facing_z:
        cam_coords = _ivy.concat([cam_coords[..., 2:3], cam_coords[..., 0:1], cam_coords[..., 1:2]], -1)
    else:
        cam_coords = cam_coords[..., 0:3]

    # BS x H x W x 3
    return _ivy_mec.cartesian_to_polar_coords(cam_coords)


def ds_pixel_to_sphere_coords(ds_pixel_coords, inv_calib_mat, batch_shape=None, image_shape=None):
    """Convert depth scaled homogeneous pixel co-ordinates image :math:`\mathbf{X}_p\in\mathbb{R}^{is×3}` to
    camera-centric ego-sphere polar co-ordinates image :math:`\mathbf{S}_c\in\mathbb{R}^{is×3}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_

    Parameters
    ----------
    ds_pixel_coords
        Depth scaled homogeneous pixel co-ordinates image *[batch_shape,image_shape,3]*
    inv_calib_mat
        Inverse calibration matrix *[batch_shape,3,3]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image dimensions. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        Camera-centric ego-sphere polar co-ordinates image *[batch_shape,image_shape,3]*

    """

    if batch_shape is None:
        batch_shape = inv_calib_mat.shape[:-2]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        image_shape = ds_pixel_coords.shape[num_batch_dims:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    # BS x IS x 4
    cam_coords = ds_pixel_to_cam_coords(ds_pixel_coords, inv_calib_mat, batch_shape, image_shape)

    # BS x IS x 3
    return cam_to_sphere_coords(cam_coords)


def angular_pixel_to_sphere_coords(angular_pixel_coords, pixels_per_degree):
    """Convert angular pixel co-ordinates image :math:`\mathbf{A}_p\in\mathbb{R}^{h×w×3}` to camera-centric ego-sphere
    polar co-ordinates image :math:`\mathbf{S}_c\in\mathbb{R}^{h×w×3}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_

    Parameters
    ----------
    angular_pixel_coords
        Angular pixel co-ordinates image *[batch_shape,h,w,3]*
    pixels_per_degree
        Number of pixels per angular degree

    Returns
    -------
    ret
        Camera-centric ego-sphere polar co-ordinates image *[batch_shape,h,w,3]*

    """

    # BS x H x W x 1
    sphere_x_coords = angular_pixel_coords[..., 0:1]
    sphere_y_coords = angular_pixel_coords[..., 1:2]
    radius_values = angular_pixel_coords[..., 2:3]

    sphere_x_angle_coords_in_degs = (180 - sphere_x_coords/(pixels_per_degree + MIN_DENOMINATOR) % 360)
    sphere_y_angle_coords_in_degs = (sphere_y_coords/(pixels_per_degree + MIN_DENOMINATOR) % 180)

    # BS x H x W x 2
    sphere_angle_coords_in_degs = _ivy.concat([sphere_x_angle_coords_in_degs, sphere_y_angle_coords_in_degs], -1)
    sphere_angle_coords = sphere_angle_coords_in_degs * np.pi / 180

    # BS x H x W x 3
    return _ivy.concat([sphere_angle_coords, radius_values], -1)


def sphere_to_cam_coords(sphere_coords, forward_facing_z=True, batch_shape=None, device=None):
    """Convert camera-centric ego-sphere polar co-ordinates image :math:`\mathbf{S}_c\in\mathbb{R}^{bs×3}` to
    camera-centric homogeneous cartesian co-ordinates image :math:`\mathbf{X}_c\in\mathbb{R}^{bs×4}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_

    Parameters
    ----------
    sphere_coords
        Camera-centric ego-sphere polar co-ordinates image *[batch_shape,3]*
    forward_facing_z
        Whether to use reference frame so z is forward facing. Default is False.
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Camera-centric homogeneous cartesian co-ordinates image *[batch_shape,4]*

    """

    if batch_shape is None:
        batch_shape = sphere_coords.shape[:-1]

    if device is None:
        device = _ivy.device(sphere_coords)

    # shapes as list
    batch_shape = list(batch_shape)

    # BS x 3
    cam_coords = _ivy_mec.polar_to_cartesian_coords(sphere_coords)
    if forward_facing_z:
        cam_coords = _ivy.concat(
            [cam_coords[..., 1:2], cam_coords[..., 2:3], cam_coords[..., 0:1]], -1)

    # BS x 4
    return _ivy_mec.make_coordinates_homogeneous(cam_coords, batch_shape)


def sphere_to_ds_pixel_coords(sphere_coords, calib_mat, batch_shape=None, image_shape=None):
    """Convert camera-centric ego-sphere polar co-ordinates image :math:`\mathbf{S}_c\in\mathbb{R}^{is×3}` to depth scaled
    homogeneous pixel co-ordinates image :math:`\mathbf{X}_p\in\mathbb{R}^{is×3}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_

    Parameters
    ----------
    sphere_coords
        Camera-centric ego-sphere polar co-ordinates image *[batch_shape,image_shape,3]*
    calib_mat
        Calibration matrix *[batch_shape,3,3]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image dimensions. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        Depth scaled homogeneous pixel co-ordinates image *[batch_shape,image_shape,3]*

    """

    if batch_shape is None:
        batch_shape = calib_mat.shape[:-2]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        image_shape = sphere_coords.shape[num_batch_dims:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    # BS x IS x 4
    cam_coords = sphere_to_cam_coords(sphere_coords, batch_shape=batch_shape + image_shape)

    # BS x IS x 3
    return cam_to_ds_pixel_coords(cam_coords, calib_mat, batch_shape, image_shape)


def sphere_to_angular_pixel_coords(sphere_coords, pixels_per_degree):
    """Convert camera-centric ego-sphere polar co-ordinates image :math:`\mathbf{S}_c\in\mathbb{R}^{h×w×3}` to angular
    pixel co-ordinates image :math:`\mathbf{A}_p\in\mathbb{R}^{h×w×3}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_

    Parameters
    ----------
    sphere_coords
        Camera-centric ego-sphere polar co-ordinates image *[batch_shape,h,w,3]*
    pixels_per_degree
        Number of pixels per angular degree

    Returns
    -------
    ret
        Angular pixel co-ordinates image *[batch_shape,h,w,3]*

    """

    # BS x H x W x 1
    sphere_radius_vals = sphere_coords[..., -1:]

    # BS x H x W x 2
    sphere_angle_coords = sphere_coords[..., 0:2]

    # BS x H x W x 2
    sphere_angle_coords_in_degs = sphere_angle_coords * 180 / np.pi

    # BS x H x W x 1
    sphere_x_coords = ((180 - sphere_angle_coords_in_degs[..., 0:1]) % 360) * pixels_per_degree
    sphere_y_coords = (sphere_angle_coords_in_degs[..., 1:2] % 180) * pixels_per_degree

    # BS x H x W x 3
    return _ivy.concat([sphere_x_coords, sphere_y_coords, sphere_radius_vals], -1)


# Camera Geometry Object Functions #
# ---------------------------------#


def persp_angles_and_pp_offsets_to_intrinsics_object(persp_angles, pp_offsets, image_dims, batch_shape=None):
    """Create camera intrinsics object from perspective angles :math:`θ_x, θ_y`, principal-point offsets :math:`p_x, p_y`
    and image dimensions [height, width].

    Parameters
    ----------
    persp_angles
        Perspective angles *[batch_shape,2]*
    pp_offsets
        Principal-point offsets *[batch_shape,2]*
    image_dims
        Image dimensions.
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)

    Returns
    -------
    ret
        Camera intrinsics object.

    """

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
    inv_calib_mat = _ivy.inv(calib_mat)

    # intrinsics object
    intrinsics = _Intrinsics(focal_lengths, persp_angles, pp_offsets, calib_mat, inv_calib_mat)
    return intrinsics


def focal_lengths_and_pp_offsets_to_intrinsics_object(focal_lengths, pp_offsets, image_dims, batch_shape=None):
    """Create camera intrinsics object from focal lengths :math:`f_x, f_y`, principal-point offsets :math:`p_x, p_y`, and
    image dimensions [height, width].

    Parameters
    ----------
    focal_lengths
        Focal lengths *[batch_shape,2]*
    pp_offsets
        Principal-point offsets *[batch_shape,2]*
    image_dims
        Image dimensions. Inferred from inputs in None.
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)

    Returns
    -------
    ret
        Camera intrinsics object

    """

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
    inv_calib_mat = _ivy.inv(calib_mat)

    # intrinsics object
    intrinsics = _Intrinsics(focal_lengths, persp_angles, pp_offsets, calib_mat, inv_calib_mat)
    return intrinsics


def calib_mat_to_intrinsics_object(calib_mat, image_dims, batch_shape=None):
    """Create camera intrinsics object from calibration matrix.

    Parameters
    ----------
    calib_mat
        Calibration matrices *[batch_shape,3,3]*
    image_dims
        Image dimensions. Inferred from inputs in None.
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)

    Returns
    -------
    ret
        Camera intrinsics object

    """

    if batch_shape is None:
        batch_shape = calib_mat.shape[:-2]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x 2
    focal_lengths = _ivy.concat([calib_mat[..., 0, 0:1], calib_mat[..., 1, 1:2]], -1)

    # BS x 2
    persp_angles = focal_lengths_to_persp_angles(focal_lengths, image_dims)

    # BS x 2
    pp_offsets = _ivy.concat([calib_mat[..., 0, -1:], calib_mat[..., 1, -1:]], -1)

    # BS x 3 x 3
    inv_calib_mat = _ivy.inv(calib_mat)

    # intrinsics object
    intrinsics = _Intrinsics(focal_lengths, persp_angles, pp_offsets, calib_mat, inv_calib_mat)
    return intrinsics


def ext_mat_and_intrinsics_to_cam_geometry_object(ext_mat, intrinsics, batch_shape=None, device=None):
    """Create camera geometry object from extrinsic matrix :math:`\mathbf{E}\in\mathbb{R}^{3×4}`, and camera intrinsics
    object.

    Parameters
    ----------
    ext_mat
        Extrinsic matrix *[batch_shape,3,4]*
    intrinsics
        camera intrinsics object
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Camera geometry object

    """

    if batch_shape is None:
        batch_shape = ext_mat.shape[:-2]

    if device is None:
        device = _ivy.device(ext_mat)

    # shapes as list
    batch_shape = list(batch_shape)

    # num batch dims
    num_batch_dims = len(batch_shape)

    # BS x 4 x 4
    ext_mat_homo = \
        _ivy.concat(
            [ext_mat, _ivy.tile(_ivy.reshape(_ivy.array([0., 0., 0., 1.], device=device),
                                             [1] * (num_batch_dims + 1) + [4]),
                                batch_shape + [1, 1])], -2)

    # BS x 4 x 4
    inv_ext_mat_homo = _ivy.inv(ext_mat_homo)

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
        _ivy.concat([
            full_mat, _ivy.tile(_ivy.reshape(_ivy.array([0., 0., 0., 1.], device=device),
                                             [1] * (num_batch_dims + 1) + [4]),
                                batch_shape + [1, 1])], -2)

    # BS x 4 x 4
    inv_full_mat_homo = _ivy.inv(full_mat_homo)

    # camera geometry object
    cam_geometry = _CameraGeometry(intrinsics, extrinsics, full_mat_homo, inv_full_mat_homo)
    return cam_geometry


def inv_ext_mat_and_intrinsics_to_cam_geometry_object(inv_ext_mat, intrinsics, batch_shape=None, device=None):
    """Create camera geometry object from inverse extrinsic matrix :math:`\mathbf{E}^{-1}\in\mathbb{R}^{3×4}`, and camera
    intrinsics object.

    Parameters
    ----------
    inv_ext_mat
        Inverse extrinsic matrix *[batch_shape,3,4]*
    intrinsics
        camera intrinsics object
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Camera geometry object

    """

    if batch_shape is None:
        batch_shape = inv_ext_mat.shape[:-2]

    if device is None:
        device = _ivy.device(inv_ext_mat)

    # shapes as list
    batch_shape = list(batch_shape)

    # num batch dims
    num_batch_dims = len(batch_shape)

    # BS x 4 x 4
    inv_ext_mat_homo = \
        _ivy.concat([inv_ext_mat, _ivy.tile(
            _ivy.reshape(_ivy.array([0., 0., 0., 1.], device=devie), [1] * (num_batch_dims + 1) + [4]),
            batch_shape + [1, 1])], -2)

    # BS x 4 x 4
    ext_mat_homo = _ivy.inv(inv_ext_mat_homo)

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
        _ivy.concat([
            full_mat, _ivy.tile(_ivy.reshape(_ivy.array([0., 0., 0., 1.], device=device),
                                             [1] * (num_batch_dims + 1) + [4]),
                                batch_shape + [1, 1])], -2)

    # BS x 4 x 4
    inv_full_mat_homo = _ivy.inv(full_mat_homo)

    # camera geometry object
    camera_geometry = _CameraGeometry(intrinsics, extrinsics, full_mat_homo, inv_full_mat_homo)
    return camera_geometry
