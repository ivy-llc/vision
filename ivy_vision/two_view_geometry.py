"""
Collection of Two-View-Geometry Functions
"""

# global
import ivy_mech as _ivy_mech
from ivy.framework_handler import get_framework as _get_framework

# local
from ivy_vision import projective_geometry as _ivy_pg
from ivy_vision import single_view_geometry as _ivy_svg

MIN_DENOMINATOR = 1e-12


def pixel_to_pixel_coords(pixel_coords1, cam1to2_full_mat, batch_shape=None, image_dims=None, dev=None, f=None):
    """
    Transform depth scaled homogeneous pixel co-ordinates image in first camera frame
    :math:`\mathbf{X}_{p1}\in\mathbb{R}^{h×w×3}` to depth scaled homogeneous pixel co-ordinates image in second camera
    frame :math:`\mathbf{X}_{p2}\in\mathbb{R}^{h×w×3}`, given camera to camera projection matrix
    :math:`\mathbf{P}_{1→2}\in\mathbb{R}^{3×4}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=174>`_

    :param pixel_coords1: Depth scaled homogeneous pixel co-ordinates image in frame 1 *[batch_shape,h,w,3]*
    :type pixel_coords1: array
    :param cam1to2_full_mat: Camera1-to-camera2 full projection matrix *[batch_shape,3,4]*
    :type cam1to2_full_mat: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Depth scaled homogeneous pixel co-ordinates image in frame 2 *[batch_shape,h,w,3]*
    """

    f = _get_framework(pixel_coords1, f=f)

    if batch_shape is None:
        batch_shape = pixel_coords1.shape[:-3]

    if image_dims is None:
        image_dims = pixel_coords1.shape[-3:-1]

    if dev is None:
        dev = f.get_device(pixel_coords1)

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x H x W x 4
    pixel_coords_homo = f.concatenate((pixel_coords1,
                                          f.ones(batch_shape + image_dims + [1], dev=dev)), -1)

    # BS x H x W x 3
    return _ivy_pg.transform(pixel_coords_homo, cam1to2_full_mat, batch_shape, image_dims, f=f)


def cam_to_cam_coords(cam_coords1, cam1to2_ext_mat, batch_shape=None, image_dims=None, dev=None, f=None):
    """
    Transform camera-centric homogeneous co-ordinates image for camera 1 :math:`\mathbf{X}_{c1}\in\mathbb{R}^{h×w×4}` to
    camera-centric homogeneous co-ordinates image for camera 2 :math:`\mathbf{X}_{c2}\in\mathbb{R}^{h×w×4}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=174>`_

    :param cam_coords1: Camera-centric homogeneous co-ordinates image in frame 1 *[batch_shape,h,w,4]*
    :type cam_coords1: array
    :param cam1to2_ext_mat: Camera1-to-camera2 extrinsic projection matrix *[batch_shape,3,4]*
    :type cam1to2_ext_mat: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Depth scaled homogeneous pixel co-ordinates image in frame 2 *[batch_shape,h,w,3]*
    """

    f = _get_framework(cam_coords1, f=f)

    if batch_shape is None:
        batch_shape = cam_coords1.shape[:-3]

    if image_dims is None:
        image_dims = cam_coords1.shape[-3:-1]

    if dev is None:
        dev = f.get_device(cam_coords1)

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x H x W x 3
    cam_coords2 = _ivy_pg.transform(cam_coords1, cam1to2_ext_mat, batch_shape, image_dims, f=f)

    # BS x H x W x 4
    return f.concatenate((cam_coords2, f.ones(batch_shape + image_dims + [1], dev=dev)), -1)


def sphere_to_sphere_coords(sphere_coords1, cam1to2_ext_mat, batch_shape=None, image_dims=None, f=None):
    """
    Convert camera-centric ego-sphere polar co-ordinates image in frame 1 :math:`\mathbf{S}_{c1}\in\mathbb{R}^{h×w×3}`
    to camera-centric ego-sphere polar co-ordinates image in frame 2 :math:`\mathbf{S}_{c2}\in\mathbb{R}^{h×w×3}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_
    
    :param sphere_coords1: Camera-centric ego-sphere polar co-ordinates image in frame 1 *[batch_shape,h,w,3]*
    :type sphere_coords1: array
    :param cam1to2_ext_mat: Camera1-to-camera2 extrinsic projection matrix *[batch_shape,3,4]*
    :type cam1to2_ext_mat: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Camera-centric ego-sphere polar co-ordinates image in frame 2 *[batch_shape,h,w,3]*
    """

    f = _get_framework(sphere_coords1, f=f)

    if batch_shape is None:
        batch_shape = sphere_coords1.shape[:-3]

    if image_dims is None:
        image_dims = sphere_coords1.shape[-3:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x H x W x 4
    cam_coords1 = _ivy_svg.sphere_to_cam_coords(sphere_coords1, batch_shape, image_dims, f=f)
    cam_coords2 = cam_to_cam_coords(cam_coords1, cam1to2_ext_mat, batch_shape, image_dims)

    # BS x H x W x 3
    return _ivy_svg.cam_to_sphere_coords(cam_coords2, batch_shape, image_dims, f=f)


def angular_pixel_to_angular_pixel_coords(angular_pixel_coords1, cam1to2_ext_mat, pixels_per_degree, batch_shape=None,
                                          image_dims=None, f=None):
    """
    Convert angular pixel co-ordinates image in frame 1 :math:`\mathbf{A}_{p1}\in\mathbb{R}^{h×w×3}` to angular pixel
    co-ordinates image in frame 2 :math:`\mathbf{A}_{p2}\in\mathbb{R}^{h×w×3}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_

    :param angular_pixel_coords1: Angular pixel co-ordinates image in frame 1 *[batch_shape,h,w,3]*
    :type angular_pixel_coords1: array
    :param cam1to2_ext_mat: Camera1-to-camera2 extrinsic projection matrix *[batch_shape,3,4]*
    :type cam1to2_ext_mat: array
    :param pixels_per_degree: Number of pixels per angular degree
    :type pixels_per_degree: float
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Camera-centric ego-sphere polar co-ordinates image in frame 2 *[batch_shape,h,w,3]*
    """

    f = _get_framework(angular_pixel_coords1, f=f)

    if batch_shape is None:
        batch_shape = angular_pixel_coords1.shape[:-3]

    if image_dims is None:
        image_dims = angular_pixel_coords1.shape[-3:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x H x W x 3
    sphere_coords1 = _ivy_svg.angular_pixel_to_sphere_coords(angular_pixel_coords1, pixels_per_degree, f=f)

    # BS x H x W x 3
    sphere_coords2 = sphere_to_sphere_coords(sphere_coords1, cam1to2_ext_mat, batch_shape, image_dims)

    # BS x H x W x 3
    return _ivy_svg.sphere_to_angular_pixel_coords(sphere_coords2, pixels_per_degree, f=f)


def get_fundamental_matrix(full_mat1, full_mat2, camera_center1=None, pinv_full_mat1=None, batch_shape=None, dev=None,
                           f=None):
    """
    Compute fundamental matrix :math:`\mathbf{F}\in\mathbb{R}^{3×3}` between two cameras, given their extrinsic
    matrices :math:`\mathbf{E}_1\in\mathbb{R}^{3×4}` and :math:`\mathbf{E}_2\in\mathbb{R}^{3×4}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=262>`_
    bottom of page 244, section 9.2.2, equation 9.1

    :param full_mat1: Frame 1 full projection matrix *[batch_shape,3,4]*
    :type full_mat1: array
    :param full_mat2: Frame 2 full projection matrix *[batch_shape,3,4]*
    :type full_mat2: array
    :param camera_center1: Frame 1 camera center, inferred from full_mat1 if None *[batch_shape,3,1]*
    :type camera_center1: array, optional
    :param pinv_full_mat1: Frame 1 full projection matrix pseudo-inverse, inferred from full_mat1 if None *[batch_shape,4,3]*
    :type pinv_full_mat1: array, optional
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Fundamental matrix connecting frames 1 and 2 *[batch_shape,3,3]*
    """

    f = _get_framework(full_mat1, f=f)

    if batch_shape is None:
        batch_shape = full_mat1.shape[:-2]

    if dev is None:
        dev = f.get_device(full_mat1)

    # shapes as list
    batch_shape = list(batch_shape)

    if camera_center1 is None:
        inv_full_mat1 = f.inv(_ivy_mech.make_transformation_homogeneous(full_mat1, batch_shape, dev, f=f))[..., 0:3, :]
        camera_center1 = _ivy_svg.inv_ext_mat_to_camera_center(inv_full_mat1, f=f)

    if pinv_full_mat1 is None:
        pinv_full_mat1 = f.pinv(full_mat1)

    # BS x 4 x 1
    camera_center1_homo = f.concatenate((camera_center1, f.ones(batch_shape + [1, 1], dev=dev)), -2)

    # BS x 3
    e2 = f.matmul(full_mat2, camera_center1_homo)[..., -1]

    # BS x 3 x 3
    e2_skew_symmetric = f.linalg.vector_to_skew_symmetric_matrix(e2, batch_shape)

    # BS x 3 x 3
    return f.matmul(e2_skew_symmetric, f.matmul(full_mat2, pinv_full_mat1))


def closest_mutual_points_along_two_skew_rays(camera_centers, world_ray_vectors, batch_shape=None, image_dims=None,
                                              dev=None, f=None):
    """
    Compute closest mutual homogeneous co-ordinates :math:`\mathbf{x}_{1,i,j}\in\mathbb{R}^{4}` and
    :math:`\mathbf{x}_{2,i,j}\in\mathbb{R}^{4}` along two world-centric rays
    :math:`\overset{\sim}{\mathbf{C}_1} + λ_1\mathbf{rv}_{1,i,j}` and
    :math:`\overset{\sim}{\mathbf{C}_2} + λ_2\mathbf{rv}_{2,i,j}`, for each index aligned pixel between two
    world-centric ray vector images :math:`\mathbf{RV}_1\in\mathbb{R}^{h×w×3}` and
    :math:`\mathbf{RV}_2\in\mathbb{R}^{h×w×3}`. The function returns two images of closest mutual homogeneous
    co-ordinates :math:`\mathbf{X}_1\in\mathbb{R}^{h×w×4}` and :math:`\mathbf{X}_2\in\mathbb{R}^{h×w×4}`,
    concatenated together into a single array.\n
    `[reference] <https://math.stackexchange.com/questions/1414285/location-of-shortest-distance-between-two-skew-lines-in-3d>`_
    second answer in forum

    :param camera_centers: Camera center *[batch_shape,2,3,1]*
    :type camera_centers: array
    :param world_ray_vectors: World ray vectors *[batch_shape,2,h,w,3]*
    :type world_ray_vectors: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Closest mutual points image *[batch_shape,2,h,w,4]*
    """

    f = _get_framework(camera_centers, f=f)

    if batch_shape is None:
        batch_shape = world_ray_vectors.shape[:-4]

    if image_dims is None:
        image_dims = world_ray_vectors.shape[-3:-1]

    if dev is None:
        dev = f.get_device(camera_centers)

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x 3 x 1
    camera_center0 = camera_centers[..., 0, :, :]
    camera_center1 = camera_centers[..., 1, :, :]

    # BS x 1 x 1 x 3
    cam1_to_cam2 = f.reshape(camera_center1 - camera_center0, batch_shape + [1, 1, 3])
    cam2_to_cam1 = f.reshape(camera_center0 - camera_center1, batch_shape + [1, 1, 3])

    # BS x 2 x H x W x 3
    ds = world_ray_vectors

    # BS x H x W x 3
    ds0 = ds[..., 0, :, :, :]
    ds1 = ds[..., 1, :, :, :]
    n = f.cross(ds0, ds1)
    n1 = f.cross(ds0, n)
    n2 = f.cross(ds1, n)

    # BS x 1 x H x W
    t1 = f.expand_dims(f.reduce_sum(cam1_to_cam2 * n2, -1) / (
            f.reduce_sum(ds0 * n2, -1) + MIN_DENOMINATOR), -3)
    t2 = f.expand_dims(f.reduce_sum(cam2_to_cam1 * n1, -1) / (
            f.reduce_sum(ds1 * n1, -1) + MIN_DENOMINATOR), -3)

    # BS x 2 x H x W
    ts = f.expand_dims(f.concatenate((t1, t2), -3), -1)

    # BS x 2 x H x W x 3
    world_coords = f.reshape(camera_centers[..., 0], batch_shape + [2, 1, 1, 3]) + ts * world_ray_vectors

    # BS x 2 x H x W x 4
    return f.concatenate((world_coords, f.ones(batch_shape + [2] + image_dims + [1], dev=dev)), -1)


def _triangulate_depth_by_closest_mutual_points(pixel_coords, full_mats, inv_full_mats, camera_centers, batch_shape,
                                                image_dims, f):

    # single view geom batch shape
    svg_batch_shape = batch_shape + [2]

    # BS x 2 x H x W x 3
    world_rays_flat = _ivy_svg.pixel_coords_to_world_ray_vectors(pixel_coords, inv_full_mats, camera_centers,
                                                                 svg_batch_shape, image_dims, f=f)

    # BS x 2 x H x W x 3
    world_rays = f.reshape(world_rays_flat, svg_batch_shape + image_dims + [3])

    # BS x 2 x H x W x 4
    world_points = closest_mutual_points_along_two_skew_rays(camera_centers, world_rays, batch_shape, image_dims, f=f)

    # BS x H x W x 3
    return _ivy_svg.world_to_pixel_coords(world_points[..., 0, :, :, :], full_mats[..., 0, :, :],
                                          batch_shape, image_dims, f=f)


def _triangulate_depth_by_homogeneous_dlt(pixel_coords, full_mats, _, _1, batch_shape, image_dims, f):

    # num batch dims
    num_batch_dims = len(batch_shape)

    # BS x 2 x H x W x 3
    pixel_coords_normalized = pixel_coords / (pixel_coords[..., -1:] + MIN_DENOMINATOR)

    # BS x 3 x 4
    P = full_mats[..., 0, :, :]
    P_dash = full_mats[..., 1, :, :]

    # BS x (HxW) x 4
    p1T = f.tile(P[..., 0:1, :], [1] * num_batch_dims + [image_dims[0] * image_dims[1], 1])
    p2T = f.tile(P[..., 1:2, :], [1] * num_batch_dims + [image_dims[0] * image_dims[1], 1])
    p3T = f.tile(P[..., 2:3, :], [1] * num_batch_dims + [image_dims[0] * image_dims[1], 1])

    p_dash_1T = f.tile(P_dash[..., 0:1, :], [1] * num_batch_dims + [image_dims[0] * image_dims[1], 1])
    p_dash_2T = f.tile(P_dash[..., 1:2, :], [1] * num_batch_dims + [image_dims[0] * image_dims[1], 1])
    p_dash_3T = f.tile(P_dash[..., 2:3, :], [1] * num_batch_dims + [image_dims[0] * image_dims[1], 1])

    # BS x (WxH) x 1
    x = f.reshape(pixel_coords_normalized[..., 0, :, :, 0], batch_shape + [-1, 1])
    y = f.reshape(pixel_coords_normalized[..., 0, :, :, 1], batch_shape + [-1, 1])
    x_dash = f.reshape(pixel_coords_normalized[..., 1, :, :, 0], batch_shape + [-1, 1])
    y_dash = f.reshape(pixel_coords_normalized[..., 1, :, :, 1], batch_shape + [-1, 1])

    # BS x (HxW) x 1 x 4
    A_row1 = f.expand_dims(x * p3T - p1T, -2)
    A_row2 = f.expand_dims(y * p3T - p2T, -2)
    A_row3 = f.expand_dims(x_dash * p_dash_3T - p_dash_1T, -2)
    A_row4 = f.expand_dims(y_dash * p_dash_3T - p_dash_2T, -2)

    # BS x (HxW) x 4 x 4
    A = f.concatenate((A_row1, A_row2, A_row3, A_row4), -2)

    # BS x (HxW) x 4
    X = _ivy_pg.solve_homogeneous_dlt(A, f=f)

    # BS x W x H x 4
    coords_wrt_world_homo_unscaled = f.reshape(X, batch_shape + image_dims + [4])
    coords_wrt_world = coords_wrt_world_homo_unscaled / (coords_wrt_world_homo_unscaled[..., -1:] + MIN_DENOMINATOR)

    # BS x W x H x 3
    return _ivy_svg.world_to_pixel_coords(coords_wrt_world, full_mats[..., 0, :, :], batch_shape, image_dims, f=f)


TRI_METHODS = {'cmp': _triangulate_depth_by_closest_mutual_points,
               'dlt': _triangulate_depth_by_homogeneous_dlt}


def triangulate_depth(pixel_coords, full_mats, inv_full_mats=None, camera_centers=None, method='cmp', batch_shape=None,
                      image_dims=None, f=None):
    """
    Triangulate depth in frame 1, returning depth scaled homogeneous pixel co-ordinate image
    :math:`\mathbf{X}\in\mathbb{R}^{h×w×3}` in frame 1.\n

    :param pixel_coords: Homogeneous pixel co-ordinate images: *[batch_shape,h,w,3]*
    :type pixel_coords: array
    :param full_mats: Full projection matrices *[batch_shape,2,3,4]*
    :type full_mats: array
    :param inv_full_mats: Inverse full projection matrices, required for closest_mutual_points method *[batch_shape,2,3,4]*
    :type inv_full_mats: array, optional
    :param camera_centers: Camera centers, required for closest_mutual_points method *[batch_shape,2,3,1]*
    :type camera_centers: array, optional
    :param method: Triangulation method, one of [cmp|dlt], for closest mutual points or homogeneous dlt approach, closest_mutual_points by default
    :type method: str, optional
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Depth scaled homogeneous pixel co-ordinates image in frame 1 *[batch_shape,h,w,3]*
    """

    f = _get_framework(pixel_coords, f=f)

    if batch_shape is None:
        batch_shape = pixel_coords.shape[:-4]

    if image_dims is None:
        image_dims = pixel_coords.shape[-3:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    if method == 'cmt':

        if inv_full_mats is None:
            inv_full_mats = f.inv(_ivy_mech.make_transformation_homogeneous(
                full_mats, batch_shape + [2], f=f))[..., 0:3, :]

        if camera_centers is None:
            camera_centers = _ivy_svg.inv_ext_mat_to_camera_center(inv_full_mats, f=f)

    try:
        return TRI_METHODS[method](pixel_coords, full_mats, inv_full_mats, camera_centers, batch_shape, image_dims, f)
    except KeyError:
        raise Exception('Triangulation method must be one of [cmp|dlt], but found {}'.format(method))
