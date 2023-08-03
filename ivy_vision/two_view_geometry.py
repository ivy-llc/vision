"""Collection of Two-View-Geometry Functions"""

# global
import ivy
import ivy_mech
from operator import mul
from functools import reduce

# local
from ivy_vision import projective_geometry as ivy_pg
from ivy_vision import single_view_geometry as ivy_svg

MIN_DENOMINATOR = 1e-12


def ds_pixel_to_ds_pixel_coords(
    ds_pixel_coords1, cam1to2_full_mat, batch_shape=None, image_shape=None, device=None
):
    r"""Transform depth scaled homogeneous pixel co-ordinates image in
    first camera frame
    :math:`\mathbf{X}_{p1}\in\mathbb{R}^{is×3}` to
    depth scaled homogeneous pixel co-ordinates image in second camera
    frame :math:`\mathbf{X}_{p2}\in\mathbb{R}^{is×3}`,
    given camera to camera projection matrix
    :math:`\mathbf{P}_{1→2}\in\mathbb{R}^{3×4}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=174>`_ # noqa

    Parameters
    ----------
    ds_pixel_coords1
        Depth scaled homogeneous pixel co-ordinates image in
        frame 1 *[batch_shape,image_shape,3]*
    cam1to2_full_mat
        Camera1-to-camera2 full projection matrix *[batch_shape,3,4]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image shape. Inferred from inputs in None. (Default value = None)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Depth scaled homogeneous pixel co-ordinates image in frame 2
        *[batch_shape,image_shape,3]*

    """
    if batch_shape is None:
        batch_shape = cam1to2_full_mat.shape[:-2]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        image_shape = ds_pixel_coords1.shape[num_batch_dims:-1]

    if device is None:
        device = ivy.dev(ds_pixel_coords1)

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    # BS x IS x 4
    pixel_coords_homo = ivy_mech.make_coordinates_homogeneous(ds_pixel_coords1)

    # BS x IS x 3
    return ivy_pg.transform(
        pixel_coords_homo, cam1to2_full_mat, batch_shape, image_shape
    )


def cam_to_cam_coords(
    cam_coords1, cam1to2_ext_mat, batch_shape=None, image_shape=None, device=None
):
    r"""Transform camera-centric homogeneous co-ordinates image for camera 1
    :math:`\mathbf{X}_{c1}\in\mathbb{R}^{is×4}` to
    camera-centric homogeneous co-ordinates image for camera 2 :
    math:`\mathbf{X}_{c2}\in\mathbb{R}^{is×4}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=174>`_ # noqa

    Parameters
    ----------
    cam_coords1
        Camera-centric homogeneous co-ordinates image in
        frame 1 *[batch_shape,image_shape,4]*
    cam1to2_ext_mat
        Camera1-to-camera2 extrinsic projection matrix *[batch_shape,3,4]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image shape. Inferred from inputs in None. (Default value = None)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Depth scaled homogeneous pixel co-ordinates image
        in frame 2 *[batch_shape,image_shape,3]*

    """
    if batch_shape is None:
        batch_shape = cam1to2_ext_mat.shape[:-2]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        image_shape = cam_coords1.shape[num_batch_dims:-1]

    if device is None:
        device = ivy.dev(cam_coords1)

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    # BS x IS x 3
    cam_coords2 = ivy_pg.transform(
        cam_coords1, cam1to2_ext_mat, batch_shape, image_shape
    )

    # BS x IS x 4
    return ivy_mech.make_coordinates_homogeneous(cam_coords2)


def sphere_to_sphere_coords(
    sphere_coords1, cam1to2_ext_mat, batch_shape=None, image_shape=None
):
    r"""Convert camera-centric ego-sphere polar co-ordinates image in frame 1
    :math:`\mathbf{S}_{c1}\in\mathbb{R}^{is×3}`
    to camera-centric ego-sphere polar co-ordinates image in frame 2
    :math:`\mathbf{S}_{c2}\in\mathbb{R}^{is×3}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_ # noqa

    Parameters
    ----------
    sphere_coords1
        Camera-centric ego-sphere polar co-ordinates image in frame 1
        *[batch_shape,image_shape,3]*
    cam1to2_ext_mat
        Camera1-to-camera2 extrinsic projection matrix *[batch_shape,3,4]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image shape. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        Camera-centric ego-sphere polar co-ordinates image in frame 2
        *[batch_shape,image_shape,3]*

    """
    if batch_shape is None:
        batch_shape = cam1to2_ext_mat.shape[:-2]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        image_shape = sphere_coords1.shape[num_batch_dims:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    # BS x IS x 4
    cam_coords1 = ivy_svg.sphere_to_cam_coords(
        sphere_coords1, batch_shape=batch_shape + image_shape
    )
    cam_coords2 = cam_to_cam_coords(
        cam_coords1, cam1to2_ext_mat, batch_shape, image_shape
    )

    # BS x IS x 3
    return ivy_svg.cam_to_sphere_coords(cam_coords2)


def angular_pixel_to_angular_pixel_coords(
    angular_pixel_coords1,
    cam1to2_ext_mat,
    pixels_per_degree,
    batch_shape=None,
    image_shape=None,
):
    r"""Convert angular pixel co-ordinates image in frame 1
    :math:`\mathbf{A}_{p1}\in\mathbb{R}^{is×3}` to angular pixel
    co-ordinates image in frame 2 :math:`\mathbf{A}_{p2}\in\mathbb{R}^{is×3}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_ # noqa

    Parameters
    ----------
    angular_pixel_coords1
        Angular pixel co-ordinates image in frame 1 *[batch_shape,image_shape,3]*
    cam1to2_ext_mat
        Camera1-to-camera2 extrinsic projection matrix *[batch_shape,3,4]*
    pixels_per_degree
        Number of pixels per angular degree
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image dimensions. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        Camera-centric ego-sphere polar co-ordinates image in frame 2
        *[batch_shape,image_shape,3]*

    """
    if batch_shape is None:
        batch_shape = cam1to2_ext_mat.shape[:-2]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        image_shape = angular_pixel_coords1.shape[num_batch_dims:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    # BS x IS x 3
    sphere_coords1 = ivy_svg.angular_pixel_to_sphere_coords(
        angular_pixel_coords1, pixels_per_degree
    )

    # BS x IS x 3
    sphere_coords2 = sphere_to_sphere_coords(
        sphere_coords1, cam1to2_ext_mat, batch_shape, image_shape
    )

    # BS x IS x 3
    return ivy_svg.sphere_to_angular_pixel_coords(sphere_coords2, pixels_per_degree)


def get_fundamental_matrix(
    full_mat1,
    full_mat2,
    camera_center1=None,
    pinv_full_mat1=None,
    batch_shape=None,
    device=None,
):
    r"""Compute fundamental matrix
    :math:`\mathbf{F}\in\mathbb{R}^{3×3}` between two cameras,
    given their extrinsic
    matrices :math:`\mathbf{E}_1\in\mathbb{R}^{3×4}` and
    :math:`\mathbf{E}_2\in\mathbb{R}^{3×4}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf#page=262>`_ # noqa
    bottom of page 244, section 9.2.2, equation 9.1

    Parameters
    ----------
    full_mat1
        Frame 1 full projection matrix *[batch_shape,3,4]*
    full_mat2
        Frame 2 full projection matrix *[batch_shape,3,4]*
    camera_center1
        Frame 1 camera center, inferred from full_mat1 if None *[batch_shape,3,1]*
        (Default value = None)
    pinv_full_mat1
        Frame 1 full projection matrix pseudo-inverse, inferred from full_mat1 if None
        *[batch_shape,4,3]* (Default value = None)
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Fundamental matrix connecting frames 1 and 2 *[batch_shape,3,3]*

    """
    if batch_shape is None:
        batch_shape = full_mat1.shape[:-2]

    if device is None:
        device = ivy.dev(full_mat1)

    # shapes as list
    batch_shape = list(batch_shape)

    if camera_center1 is None:
        inv_full_mat1 = ivy.inv(
            ivy_mech.make_transformation_homogeneous(full_mat1, batch_shape, device)
        )[..., 0:3, :]
        camera_center1 = ivy_svg.inv_ext_mat_to_camera_center(inv_full_mat1)

    if pinv_full_mat1 is None:
        pinv_full_mat1 = ivy.pinv(full_mat1)

    # BS x 4 x 1
    camera_center1_homo = ivy.concat(
        (camera_center1, ivy.ones(batch_shape + [1, 1], device=device)), axis=-2
    )

    # BS x 3
    e2 = ivy.matmul(full_mat2, camera_center1_homo)[..., -1]

    # BS x 3 x 3
    e2_skew_symmetric = ivy.vector_to_skew_symmetric_matrix(e2)

    # BS x 3 x 3
    return ivy.matmul(e2_skew_symmetric, ivy.matmul(full_mat2, pinv_full_mat1))


# noinspection PyUnresolvedReferences
def closest_mutual_points_along_two_skew_rays(
    camera_centers, world_ray_vectors, batch_shape=None, image_shape=None, device=None
):
    r"""Compute closest mutual homogeneous co-ordinates
    :math:`\mathbf{x}_{1,i,j}\in\mathbb{R}^{4}` and
    :math:`\mathbf{x}_{2,i,j}\in\mathbb{R}^{4}` along two world-centric rays
    :math:`\overset{\sim}{\mathbf{C}_1} + λ_1\mathbf{rv}_{1,i,j}` and
    :math:`\overset{\sim}{\mathbf{C}_2} + λ_2\mathbf{rv}_{2,i,j}`,
    for each index aligned pixel between two
    world-centric ray vector images
    :math:`\mathbf{RV}_1\in\mathbb{R}^{is×3}` and
    :math:`\mathbf{RV}_2\in\mathbb{R}^{is×3}`.
    The function returns two images of closest mutual homogeneous
    co-ordinates :math:`\mathbf{X}_1\in\mathbb{R}^{is×4}` and
    :math:`\mathbf{X}_2\in\mathbb{R}^{is×4}`,
    concatenated together into a single array.\n
    `[reference] <https://math.stackexchange.com/questions/1414285/location-of-shortest-distance-between-two-skew-lines-in-3d>`_ # noqa
    second answer in forum

    Parameters
    ----------
    camera_centers
        Camera center *[batch_shape,2,3,1]*
    world_ray_vectors
        World ray vectors *[batch_shape,2,image_shape,3]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image dimensions. Inferred from inputs in None. (Default value = None)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Closest mutual points image *[batch_shape,2,image_shape,4]*

    """
    if batch_shape is None:
        batch_shape = camera_centers.shape[:-3]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        image_shape = world_ray_vectors.shape[num_batch_dims + 1 : -1]
    num_image_dims = len(image_shape)

    if device is None:
        device = ivy.dev(camera_centers)

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    # BS x 3 x 1
    camera_center0 = camera_centers[..., 0, :, :]
    camera_center1 = camera_centers[..., 1, :, :]

    # BS x [1]*NID x 3
    cam1_to_cam2 = ivy.reshape(
        camera_center1 - camera_center0, batch_shape + [1] * num_image_dims + [3]
    )
    cam2_to_cam1 = ivy.reshape(
        camera_center0 - camera_center1, batch_shape + [1] * num_image_dims + [3]
    )

    # BS x 2 x IS x 3
    ds = world_ray_vectors

    # BS x IS x 3
    ds0 = ds[tuple([slice(None)] * num_batch_dims + [0])]
    ds1 = ds[tuple([slice(None)] * num_batch_dims + [1])]
    n = ivy.cross(ds0, ds1)
    n1 = ivy.cross(ds0, n)
    n2 = ivy.cross(ds1, n)

    # BS x 1 x IS
    t1 = ivy.expand_dims(
        ivy.sum(cam1_to_cam2 * n2, axis=-1)
        / (ivy.sum(ds0 * n2, axis=-1) + MIN_DENOMINATOR),
        axis=num_batch_dims,
    )
    t2 = ivy.expand_dims(
        ivy.sum(cam2_to_cam1 * n1, axis=-1)
        / (ivy.sum(ds1 * n1, axis=-1) + MIN_DENOMINATOR),
        axis=num_batch_dims,
    )

    # BS x 2 x IS x 1
    ts = ivy.expand_dims(ivy.concat((t1, t2), axis=num_batch_dims), axis=-1)

    # BS x 2 x IS x 3
    world_coords = (
        ivy.reshape(
            camera_centers[..., 0], batch_shape + [2] + [1] * num_image_dims + [3]
        )
        + ts * world_ray_vectors
    )

    # BS x 2 x IS x 4
    return ivy_mech.make_coordinates_homogeneous(
        world_coords, batch_shape + [2] + image_shape
    )


def _triangulate_depth_by_closest_mutual_points(
    ds_pixel_coords, full_mats, inv_full_mats, camera_centers, batch_shape, image_shape
):
    """

    Parameters
    ----------
    ds_pixel_coords

    full_mats

    inv_full_mats

    camera_centers

    batch_shape

    image_shape

    """
    # single view geom batch shape
    svg_batch_shape = batch_shape + [2]
    num_batch_dims = len(batch_shape)

    # BS x 2 x IS x 3
    world_rays_flat = ivy_svg.pixel_coords_to_world_ray_vectors(
        inv_full_mats, ds_pixel_coords, camera_centers, svg_batch_shape, image_shape
    )

    # BS x 2 x IS x 3
    world_rays = ivy.reshape(world_rays_flat, svg_batch_shape + image_shape + [3])

    # BS x 2 x IS x 4
    world_points = closest_mutual_points_along_two_skew_rays(
        camera_centers, world_rays, batch_shape, image_shape
    )

    # BS x IS x 3
    return ivy_svg.world_to_ds_pixel_coords(
        world_points[tuple([slice(None)] * num_batch_dims + [0])],
        full_mats[..., 0, :, :],
        batch_shape,
        image_shape,
    )


def _triangulate_depth_by_homogeneous_dlt(
    ds_pixel_coords, full_mats, _, _1, batch_shape, image_shape
):
    """

    Parameters
    ----------
    ds_pixel_coords

    full_mats

    _

    _1

    batch_shape

    image_shape

    """
    # num batch dims
    num_batch_dims = len(batch_shape)
    num_image_dims = len(image_shape)
    image_size = reduce(mul, image_shape)

    # BS x 2 x IS x 3
    pixel_coords_normalized = ds_pixel_coords / (
        ds_pixel_coords[..., -1:] + MIN_DENOMINATOR
    )

    # BS x 3 x 4
    P = full_mats[..., 0, :, :]
    P_dash = full_mats[..., 1, :, :]

    # BS x (IS) x 4
    p1T = ivy.tile(P[..., 0:1, :], [1] * num_batch_dims + [image_size, 1])
    p2T = ivy.tile(P[..., 1:2, :], [1] * num_batch_dims + [image_size, 1])
    p3T = ivy.tile(P[..., 2:3, :], [1] * num_batch_dims + [image_size, 1])

    p_dash_1T = ivy.tile(P_dash[..., 0:1, :], [1] * num_batch_dims + [image_size, 1])
    p_dash_2T = ivy.tile(P_dash[..., 1:2, :], [1] * num_batch_dims + [image_size, 1])
    p_dash_3T = ivy.tile(P_dash[..., 2:3, :], [1] * num_batch_dims + [image_size, 1])

    # BS x (IS) x 1
    x = ivy.reshape(
        pixel_coords_normalized[
            tuple(
                [slice(None)] * num_batch_dims
                + [0]
                + [slice(None)] * num_image_dims
                + [0]
            )
        ],
        batch_shape + [image_size, 1],
    )
    y = ivy.reshape(
        pixel_coords_normalized[
            tuple(
                [slice(None)] * num_batch_dims
                + [0]
                + [slice(None)] * num_image_dims
                + [1]
            )
        ],
        batch_shape + [image_size, 1],
    )
    x_dash = ivy.reshape(
        pixel_coords_normalized[
            tuple(
                [slice(None)] * num_batch_dims
                + [1]
                + [slice(None)] * num_image_dims
                + [0]
            )
        ],
        batch_shape + [image_size, 1],
    )
    y_dash = ivy.reshape(
        pixel_coords_normalized[
            tuple(
                [slice(None)] * num_batch_dims
                + [1]
                + [slice(None)] * num_image_dims
                + [1]
            )
        ],
        batch_shape + [image_size, 1],
    )

    # BS x (IS) x 1 x 4
    A_row1 = ivy.expand_dims(x * p3T - p1T, axis=-2)
    A_row2 = ivy.expand_dims(y * p3T - p2T, axis=-2)
    A_row3 = ivy.expand_dims(x_dash * p_dash_3T - p_dash_1T, axis=-2)
    A_row4 = ivy.expand_dims(y_dash * p_dash_3T - p_dash_2T, axis=-2)

    # BS x (IS) x 4 x 4
    A = ivy.concat((A_row1, A_row2, A_row3, A_row4), axis=-2)

    # BS x (IS) x 4
    X = ivy_pg.solve_homogeneous_dlt(A)

    # BS x IS x 4
    coords_wrt_world_homo_unscaled = ivy.reshape(X, batch_shape + image_shape + [4])
    coords_wrt_world = coords_wrt_world_homo_unscaled / (
        coords_wrt_world_homo_unscaled[..., -1:] + MIN_DENOMINATOR
    )

    # BS x IS x 3
    return ivy_svg.world_to_ds_pixel_coords(
        coords_wrt_world, full_mats[..., 0, :, :], batch_shape, image_shape
    )


TRI_METHODS = {
    "cmp": _triangulate_depth_by_closest_mutual_points,
    "dlt": _triangulate_depth_by_homogeneous_dlt,
}


def triangulate_depth(
    ds_pixel_coords,
    full_mats,
    inv_full_mats=None,
    camera_centers=None,
    method="cmp",
    batch_shape=None,
    image_shape=None,
):
    r"""Triangulate depth in frame 1, returning depth scaled homogeneous pixel
    co-ordinate image :math:`\mathbf{X}\in\mathbb{R}^{is×3}` in frame 1.\n

    Parameters
    ----------
    ds_pixel_coords
        Homogeneous pixel co-ordinate images: *[batch_shape,image_shape,3]*
    full_mats
        Full projection matrices *[batch_shape,2,3,4]*
    inv_full_mats
        Inverse full projection matrices, required for closest_mutual_points method
        *[batch_shape,2,3,4]* (Default value = None)
    camera_centers
        Camera centers, required for closest_mutual_points method *[batch_shape,2,3,1]*
        (Default value = None)
    method
        Triangulation method, one of [cmp|dlt], for closest mutual points or
        homogeneous dlt approach, closest_mutual_points by default
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image dimensions. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        Depth scaled homogeneous pixel co-ordinates image in frame 1
        *[batch_shape,image_shape,3]*

    """
    if batch_shape is None:
        batch_shape = ds_pixel_coords.shape[:-4]

    if image_shape is None:
        image_shape = ds_pixel_coords.shape[-3:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    if method == "cmt":
        if inv_full_mats is None:
            inv_full_mats = ivy.inv(
                ivy_mech.make_transformation_homogeneous(full_mats, batch_shape + [2])
            )[..., 0:3, :]

        if camera_centers is None:
            camera_centers = ivy_svg.inv_ext_mat_to_camera_center(inv_full_mats)

    try:
        return TRI_METHODS[method](
            ds_pixel_coords,
            full_mats,
            inv_full_mats,
            camera_centers,
            batch_shape,
            image_shape,
        )
    except KeyError:
        raise Exception(
            "Triangulation method must be one of [cmp|dlt], but found {}".format(method)
        )
