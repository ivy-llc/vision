"""Collection of Optical Flow Functions"""

# global
import ivy
import ivy_mech

# local
from ivy_vision import image as ivy_img
from ivy_vision import two_view_geometry as ivy_tvg
from ivy_vision import projective_geometry as ivy_pg
from ivy_vision import single_view_geometry as ivy_svg

MIN_DENOMINATOR = 1e-12


def depth_from_flow_and_cam_mats(
    flow,
    full_mats,
    inv_full_mats=None,
    camera_centers=None,
    uniform_pixel_coords=None,
    triangulation_method="cmp",
    batch_shape=None,
    image_dims=None,
    dev_str=None,
):
    """Compute depth map :math:`\mathbf{X}\in\mathbb{R}^{h×w×1}` in frame 1 using
    optical flow :math:`\mathbf{U}_{1→2}\in\mathbb{R}^{h×w×2}` from frame 1 to 2,
    and the camera geometry.\n

    Parameters
    ----------
    flow
        Optical flow from frame 1 to 2 *[batch_shape,h,w,2]*
    full_mats
        Full projection matrices *[batch_shape,2,3,4]*
    inv_full_mats
        Inverse full projection matrices, inferred from full_mats if None and
        'cmp' triangulation method *[batch_shape,2,3,4]* (Default value = None)
    camera_centers
        Camera centers, inferred from inv_full_mats if None and
        'cmp' triangulation method *[batch_shape,2,3,1]* (Default value = None)
    uniform_pixel_coords
        Homogeneous uniform (integer) pixel co-ordinate images,
        inferred from image_dims if None *[batch_shape,h,w,3]* (Default value = None)
    triangulation_method
        Triangulation method, one of [cmp|dlt], for closest mutual points or
        homogeneous dlt approach, closest_mutual_points by default
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_dims
        Image dimensions. Inferred from inputs in None. (Default value = None)
    dev_str
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Depth map in frame 1 *[batch_shape,h,w,1]*

    """

    if batch_shape is None:
        batch_shape = flow.shape[:-3]

    if image_dims is None:
        image_dims = flow.shape[-3:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    if dev_str is None:
        dev_str = ivy.dev(flow)

    if inv_full_mats is None:
        inv_full_mats = ivy.inv(
            ivy_mech.make_transformation_homogeneous(
                full_mats, batch_shape + [2], dev_str
            )
        )[..., 0:3, :]

    if camera_centers is None:
        camera_centers = ivy_svg.inv_ext_mat_to_camera_center(inv_full_mats)

    if uniform_pixel_coords is None:
        uniform_pixel_coords = ivy_svg.create_uniform_pixel_coords_image(
            image_dims, batch_shape, dev_str=dev_str
        )

    # BS x H x W x 3
    flow_homo = ivy.concat(
        (flow, ivy.zeros(batch_shape + image_dims + [1], device=dev_str)), axis=-1
    )

    # BS x H x W x 3
    transformed_pixel_coords = uniform_pixel_coords + flow_homo

    # BS x 2 x H x W x 3
    pixel_coords = ivy.concat(
        (
            ivy.expand_dims(uniform_pixel_coords, axis=-4),
            ivy.expand_dims(transformed_pixel_coords, axis=-4),
        ),
        axis=-4,
    )

    # BS x H x W x 1
    return ivy_tvg.triangulate_depth(
        pixel_coords,
        full_mats,
        inv_full_mats,
        camera_centers,
        triangulation_method,
        batch_shape,
        image_dims,
    )[..., -1:]


def flow_from_depth_and_cam_mats(
    pixel_coords1, cam1to2_full_mat, batch_shape=None, image_shape=None
):
    """Compute optical flow :math:`\mathbf{U}_{1→2}\in\mathbb{R}^{is×2}` from frame 1 to 2, using depth map
    :math:`\mathbf{X}\in\mathbb{R}^{is×1}` in frame 1, and the camera geometry.\n

    Parameters
    ----------
    pixel_coords1
        Depth scaled homogeneous pixel co-ordinates image in frame 1
        *[batch_shape,image_shape,3]*
    cam1to2_full_mat
        Camera1-to-camera2 full projection matrix *[batch_shape,3,4]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image dimensions. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        Optical flow from frame 1 to 2 *[batch_shape,image_shape,2]*

    """

    if batch_shape is None:
        batch_shape = cam1to2_full_mat.shape[:-2]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        image_shape = pixel_coords1.shape[num_batch_dims:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    # BS x IS x 3
    projected_pixel_coords = ivy_tvg.ds_pixel_to_ds_pixel_coords(
        pixel_coords1, cam1to2_full_mat, batch_shape, image_shape
    )
    projected_pixel_coords_normalized = projected_pixel_coords / (
        projected_pixel_coords[..., -1:] + MIN_DENOMINATOR
    )

    # BS x IS x 3
    pixel_coords1_normalized = pixel_coords1 / (
        pixel_coords1[..., -1:] + MIN_DENOMINATOR
    )

    # BS x IS x 2
    return (
        projected_pixel_coords_normalized[..., 0:2] - pixel_coords1_normalized[..., 0:2]
    )


def project_flow_to_epipolar_line(
    flow,
    fund_mat,
    uniform_pixel_coords=None,
    batch_shape=None,
    image_dims=None,
    dev_str=None,
):
    """Project optical flow :math:`\mathbf{U}_{1→2}\in\mathbb{R}^{h×w×2}` to epipolar
    line in frame 1.\n `[reference]
    <https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    #Line_defined_by_an_equation>`_

    Parameters
    ----------
    flow
        Optical flow from frame 1 to 2 *[batch_shape,h,w,2]*
    fund_mat
        Fundamental matrix connecting frames 1 and 2 *[batch_shape,3,3]*
    uniform_pixel_coords
        Homogeneous uniform (integer) pixel co-ordinate images,
        inferred from image_dims if None *[batch_shape,h,w,3]* (Default value = None)
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_dims
        Image dimensions. Inferred from inputs in None. (Default value = None)
    dev_str
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Optical flow from frame 1 to 2, projected to frame 1 epipolar line
        *[batch_shape,h,w,2]*

    """

    if batch_shape is None:
        batch_shape = flow.shape[:-3]

    if image_dims is None:
        image_dims = flow.shape[-3:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    if uniform_pixel_coords is None:
        uniform_pixel_coords = ivy_svg.create_uniform_pixel_coords_image(
            image_dims, batch_shape, dev_str
        )

    # BS x H x W x 3
    epipolar_lines = ivy_pg.transform(
        uniform_pixel_coords, fund_mat, batch_shape, image_dims
    )

    # BS x H x W x 2
    flow_pixels = uniform_pixel_coords[..., 0:2] + flow

    # BS x H x W x 1
    a = epipolar_lines[..., 0:1]
    b = epipolar_lines[..., 1:2]
    c = epipolar_lines[..., 2:3]

    x0 = flow_pixels[..., 0:1]
    y0 = flow_pixels[..., 1:2]

    bx0 = b * x0
    ay0 = a * y0
    a_sqrd = a**2
    b_sqrd = b**2
    ac = a * c
    bc = b * c
    a_sqrd_plus_b_sqrd = a_sqrd + b_sqrd

    x = (b * (bx0 - ay0) - ac) / (a_sqrd_plus_b_sqrd + MIN_DENOMINATOR)
    y = (a * (-bx0 + ay0) - bc) / (a_sqrd_plus_b_sqrd + MIN_DENOMINATOR)

    # BS x H x W x 2
    projected_pixels = ivy.concat((x, y), axis=-1)
    projected_flow = projected_pixels - uniform_pixel_coords[..., 0:2]
    return projected_flow


# noinspection PyUnresolvedReferences
def pixel_cost_volume(image1, image2, search_range, batch_shape=None):
    """Compute cost volume from image feature patch comparisons between first image
    :math:`\mathbf{X}_1\in\mathbb{R}^{h×w×d}` and second image :math:`\mathbf{
    X}_2\in\mathbb{R}^{h×w×d}`, as used in FlowNet paper.\n `[reference]
    <https://www.cv-foundation.org/openaccess/content_iccv_2015/papers
    /Dosovitskiy_FlowNet_Learning_Optical_ICCV_2015_paper.pdf>`_

    Parameters
    ----------
    image1
        Image 1 *[batch_shape,h,w,D]*
    image2
        Image 2 *[batch_shape,h,w,D]*
    search_range
        Search range for patch comparisons.
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)

    Returns
    -------
    ret
        Cost volume between the images *[batch_shape,h,w,(search_range*2+1)^2]*

    """

    if batch_shape is None:
        batch_shape = image1.shape[:-3]

    # shapes as list
    batch_shape = list(batch_shape)

    # shape info
    shape = image1.shape
    h = shape[-3]
    w = shape[-2]
    max_offset = search_range * 2 + 1

    # pad dims
    pad_dims = (
        [[0, 0]] * len(batch_shape) + [[search_range, search_range]] * 2 + [[0, 0]]
    )

    # BS x (H+2*SR) x (W+2*SR) x D
    padded_lvl = ivy.zero_pad(image2, pad_dims)

    # create list
    cost_vol = []

    # iterate through patches
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            # BS x H x W x D
            tensor_slice = padded_lvl[..., y : y + h, x : x + w, :]

            # BS x H x W x 1
            cost = ivy.mean(image1 * tensor_slice, axis=-1, keepdims=True)

            # append to list
            cost_vol.append(cost)

    # BS x H x W x (max_offset^2)
    return ivy.concat(cost_vol, axis=-1)


# noinspection PyUnresolvedReferences
def velocity_from_flow_cam_coords_and_cam_mats(
    flow_t_to_tm1,
    cam_coords_t,
    cam_coords_tm1,
    cam_tm1_to_t_ext_mat,
    delta_t,
    uniform_pixel_coords=None,
    batch_shape=None,
    image_dims=None,
    dev_str=None,
):
    """Compute relative cartesian velocity from optical flow, camera co-ordinates, and
    camera extrinsics.

    Parameters
    ----------
    flow_t_to_tm1
        Optical flow from frame t to t-1 *[batch_shape,h,w,2]*
    cam_coords_t
        Camera-centric homogeneous co-ordinates image in frame t *[batch_shape,h,w,4]*
    cam_coords_tm1
        Camera-centric homogeneous co-ordinates image in frame t-1 *[batch_shape,h,w,4]*
    cam_tm1_to_t_ext_mat
        Camera t-1 to camera t extrinsic projection matrix *[batch_shape,3,4]*
    delta_t
        Time difference between frame at timestep t-1 and t *[batch_shape,1]*
    uniform_pixel_coords
        Homogeneous uniform (integer) pixel co-ordinate images,
        inferred from image_dims if None *[batch_shape,h,w,3]* (Default value = None)
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_dims
        Image dimensions. Inferred from inputs in None. (Default value = None)
    dev_str
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Cartesian velocity measurements relative to the camera *[batch_shape,h,w,3]*

    """

    if batch_shape is None:
        batch_shape = flow_t_to_tm1.shape[:-3]

    if image_dims is None:
        image_dims = flow_t_to_tm1.shape[-3:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    if dev_str is None:
        dev_str = ivy.dev(flow_t_to_tm1)

    if uniform_pixel_coords is None:
        uniform_pixel_coords = ivy_svg.create_uniform_pixel_coords_image(
            image_dims, batch_shape, dev_str
        )

    # Interpolate cam coords from frame t-1

    # BS x H x W x 2
    warp = uniform_pixel_coords[..., 0:2] + flow_t_to_tm1

    # BS x H x W x 4
    cam_coords_tm1_interp = ivy_img.bilinear_resample(cam_coords_tm1, warp)

    # Project to frame t

    # BS x H x W x 4
    cam_coords_t_proj = ivy_tvg.cam_to_cam_coords(
        cam_coords_tm1_interp, cam_tm1_to_t_ext_mat, batch_shape, image_dims
    )

    # delta co-ordinates

    # BS x H x W x 3
    delta_cam_coords_t = (cam_coords_t - cam_coords_t_proj)[..., 0:3]

    # velocity

    # BS x H x W x 3
    vel = delta_cam_coords_t / ivy.reshape(delta_t, batch_shape + [1] * 3)

    # Validity mask

    # BS x H x W x 1
    validity_mask = (
        ivy.sum(
            ivy.astype(
                warp
                < ivy.array(
                    [image_dims[1], image_dims[0]], dtype="float32", device=dev_str
                ),
                "int32",
            ),
            axis=-1,
            keepdims=True,
        )
        == 2
    )

    # pruned

    # BS x H x W x 3,    BS x H x W x 1
    return (
        ivy.where(validity_mask, vel, ivy.zeros_like(vel, device=dev_str)),
        validity_mask,
    )


def project_cam_coords_with_object_transformations(
    cam_coords_1,
    id_image,
    obj_ids,
    obj_trans,
    cam_1_to_2_ext_mat,
    batch_shape=None,
    image_shape=None,
):
    """Compute velocity image from co-ordinate image, id image, and object
    transformations.

    Parameters
    ----------
    cam_coords_1
        Camera-centric homogeneous co-ordinates image in frame t
        *[batch_shape,image_shape,4]*
    id_image
        Image containing per-pixel object ids *[batch_shape,h,w,1]*
    obj_ids
        Object ids *[batch_shape,num_obj,1]*
    obj_trans
        Object transformations for this frame over time *[batch_shape,num_obj,3,4]*
    cam_1_to_2_ext_mat
        Camera 1 to camera 2 extrinsic projection matrix *[batch_shape,3,4]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image dimensions. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        Relative velocity image *[batch_shape,image_shape,3]*

    """

    if batch_shape is None:
        batch_shape = cam_1_to_2_ext_mat.shape[:-2]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        image_shape = cam_coords_1.shape[num_batch_dims:-1]
    num_image_dims = len(image_shape)

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)
    num_batch_dims = len(batch_shape)

    # Transform the co-ordinate image by each transformation

    # BS x (num_obj x 3) x 4
    obj_trans = ivy.reshape(obj_trans, batch_shape + [-1, 4])

    # BS x 4 x IS
    cam_coords_1_ = ivy.permute_dims(
        cam_coords_1,
        axes=list(range(num_batch_dims))
        + [
            i + num_batch_dims for i in ([num_image_dims] + list(range(num_image_dims)))
        ],
    )

    # BS x 4 x (IS)
    cam_coords_1_ = ivy.reshape(cam_coords_1_, batch_shape + [4, -1])

    # BS x (num_obj x 3) x (IS)
    cam_coords_2_all_obj_trans = ivy.matmul(obj_trans, cam_coords_1_)

    # BS x (IS) x (num_obj x 3)
    cam_coords_2_all_obj_trans = ivy.swapaxes(cam_coords_2_all_obj_trans, -1, -2)

    # BS x IS x num_obj x 3
    cam_coords_2_all_obj_trans = ivy.reshape(
        cam_coords_2_all_obj_trans, batch_shape + image_shape + [-1, 3]
    )

    # Multiplier

    # BS x [1]*NID x num_obj
    obj_ids = ivy.reshape(obj_ids, batch_shape + [1] * num_image_dims + [-1])

    # BS x IS x num_obj x 1
    multiplier = ivy.astype(ivy.expand_dims(obj_ids == id_image, axis=-1), "float32")

    # compute validity mask, for pixels which are on moving objects

    # BS x IS x 1
    motion_mask = ivy.sum(multiplier, axis=-2) > 0

    # make invalid transformations equal to zero

    # BS x IS x num_obj x 3
    cam_coords_2_all_obj_trans_w_zeros = cam_coords_2_all_obj_trans * multiplier

    # reduce to get only valid transformations

    # BS x IS x 3
    cam_coords_2_all_obj_trans = ivy.sum(cam_coords_2_all_obj_trans_w_zeros, axis=-2)

    # find cam coords to for zero motion pixels

    # BS x IS x 3
    cam_coords_2_wo_motion = ivy_tvg.cam_to_cam_coords(
        cam_coords_1, cam_1_to_2_ext_mat, batch_shape, image_shape
    )

    # BS x IS x 4
    cam_coords_2_all_trans_homo = ivy_mech.make_coordinates_homogeneous(
        cam_coords_2_all_obj_trans, batch_shape + image_shape
    )
    cam_coords_2 = ivy.where(
        motion_mask, cam_coords_2_all_trans_homo, cam_coords_2_wo_motion
    )

    # return

    # BS x IS x 3,    BS x IS x 1
    return cam_coords_2, motion_mask


def velocity_from_cam_coords_id_image_and_object_trans(
    cam_coords_t,
    id_image,
    obj_ids,
    obj_trans,
    delta_t,
    batch_shape=None,
    image_shape=None,
    dev_str=None,
):
    """Compute velocity image from co-ordinate image, id image, and object
    transformations.

    Parameters
    ----------
    cam_coords_t
        Camera-centric homogeneous co-ordinates image in frame t
        *[batch_shape,image_shape,4]*
    id_image
        Image containing per-pixel object ids *[batch_shape,image_shape,1]*
    obj_ids
        Object ids *[batch_shape,num_obj,1]*
    obj_trans
        Object transformations for this frame over time *[batch_shape,num_obj,3,4]*
    delta_t
        Time difference between frame at timestep t-1 and t *[batch_shape,1]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image dimensions. Inferred from inputs in None. (Default value = None)
    dev_str
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Relative velocity image *[batch_shape,image_shape,3]*

    """

    if batch_shape is None:
        batch_shape = delta_t.shape[:-1]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        image_shape = cam_coords_t.shape[num_batch_dims:-1]

    if dev_str is None:
        dev_str = ivy.dev(cam_coords_t)

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    # get co-ordinate re-projections

    # BS x IS x 4
    (
        cam_coords_t_all_trans,
        motion_mask,
    ) = project_cam_coords_with_object_transformations(
        cam_coords_t,
        id_image,
        obj_ids,
        obj_trans,
        ivy.eye(4, batch_shape=batch_shape)[..., 0:3, :],
        batch_shape,
        image_shape,
    )

    # BS x IS x 4
    cam_coords_t_all_trans = ivy.where(
        motion_mask,
        cam_coords_t_all_trans,
        ivy.zeros_like(cam_coords_t_all_trans, device=dev_str),
    )

    # compute velocities

    # BS x IS x 3
    vel = (cam_coords_t[..., 0:3] - cam_coords_t_all_trans[..., 0:3]) / delta_t

    # prune velocities

    # BS x IS x 3
    return ivy.where(motion_mask, vel, ivy.zeros_like(vel, device=dev_str))


def flow_from_cam_coords_id_image_and_object_trans(
    cam_coords_f1,
    id_image,
    obj_ids,
    obj_trans,
    calib_mat,
    cam_1_to_2_ext_mat,
    batch_shape=None,
    image_shape=None,
):
    """Compute optical flow from co-ordinate image, id image, and object
    transformations.

    Parameters
    ----------
    cam_coords_f1
        Camera-centric homogeneous co-ordinates image in frame t
        *[batch_shape,image_shape,4]*
    id_image
        Image containing per-pixel object ids *[batch_shape,image_shape,1]*
    obj_ids
        Object ids *[batch_shape,num_obj,1]*
    obj_trans
        Object transformations for this frame over time *[batch_shape,num_obj,3,4]*
    calib_mat
        Calibration matrix *[batch_shape,3,3]*
    cam_1_to_2_ext_mat
        Camera 1 to camera 2 extrinsic projection matrix *[batch_shape,3,4]*
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_shape
        Image dimensions. Inferred from inputs in None. (Default value = None)

    Returns
    -------
    ret
        Relative velocity image *[batch_shape,image_shape,3]*

    """

    if batch_shape is None:
        batch_shape = calib_mat.shape[:-2]
    num_batch_dims = len(batch_shape)

    if image_shape is None:
        image_shape = cam_coords_f1.shape[num_batch_dims:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_shape = list(image_shape)

    # get co-ordinate re-projections

    # BS x IS x 3
    cam_coords_trans_f2, _ = project_cam_coords_with_object_transformations(
        cam_coords_f1,
        id_image,
        obj_ids,
        obj_trans,
        cam_1_to_2_ext_mat,
        batch_shape,
        image_shape,
    )

    # co-ordinates to pixel co-ordinates

    # BS x IS x 3
    pixel_coords_f1 = ivy_svg.cam_to_ds_pixel_coords(
        cam_coords_f1, calib_mat, batch_shape, image_shape
    )
    pixel_coords_trans_f2 = ivy_svg.cam_to_ds_pixel_coords(
        cam_coords_trans_f2, calib_mat, batch_shape, image_shape
    )

    # unscaled pixel coords
    unscaled_pixel_coords_f1 = pixel_coords_f1[..., 0:2] / (
        pixel_coords_f1[..., -1:] + MIN_DENOMINATOR
    )
    unscaled_pixel_coords_f2 = pixel_coords_trans_f2[..., 0:2] / (
        pixel_coords_trans_f2[..., -1:] + MIN_DENOMINATOR
    )

    # optical flow

    # BS x IS x 2
    return unscaled_pixel_coords_f2 - unscaled_pixel_coords_f1
