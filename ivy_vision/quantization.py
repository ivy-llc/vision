"""Collection of Quantization Functions"""

# global
import ivy

# local
from ivy_vision import single_view_geometry as ivy_svg

MIN_DENOMINATOR = 1e-12
MIN_DEPTH_DIFF = 1e-2


def quantize_to_image(
    pixel_coords,
    final_image_dims,
    feat=None,
    feat_prior=None,
    with_db=False,
    pixel_coords_var=1e-3,
    feat_var=1e-3,
    pixel_coords_prior_var=1e12,
    feat_prior_var=1e12,
    var_threshold=(1e-3, 1e12),
    uniform_pixel_coords=None,
    batch_shape=None,
    dev_str=None,
):
    r"""Quantize pixel co-ordinates with d feature channels (for depth, rgb, normals
    etc.), from images :math:`\mathbf{X}\in\mathbb{R}^{input\_images\_shape×(2+d)}`,
    which may have been reprojected from a host of different cameras (leading to
    non-integer pixel values), to a new quantized pixel co-ordinate image with the
    same feature channels :math:`\mathbf{X}\in\mathbb{R}^{h×w×(2+d)}`, and with
    integer pixel co-ordinates. Duplicates during the quantization are either
    probabilistically fused based on variance, or the minimum depth is chosen when
    using depth buffer mode.

    Parameters
    ----------
    pixel_coords
        Pixel co-ordinates *[batch_shape,input_size,2]*
    final_image_dims
        Image dimensions of the final image.
    feat
        Features (i.e. depth, rgb, encoded), default is None.
        *[batch_shape,input_size,d]*
    feat_prior
        Prior feature image mean, default is None. *[batch_shape,input_size,d]*
    with_db
        Whether or not to use depth buffer in rendering, default is false
    pixel_coords_var
        Pixel coords variance *[batch_shape,input_size,2]* (Default value = 1e-3)
    feat_var
        Feature variance *[batch_shape,input_size,d]* (Default value = 1e-3)
    pixel_coords_prior_var
        Pixel coords prior variance *[batch_shape,h,w,2]* (Default value = 1e12)
    feat_prior_var
        Features prior variance *[batch_shape,h,w,3]* (Default value = 1e12)
    var_threshold
        Variance threshold, for projecting valid coords and clipping
        *[batch_shape,2+d,2]* (Default value = (1e-3)
    uniform_pixel_coords
        Homogeneous uniform (integer) pixel co-ordinate images,
        inferred from final_image_dims
        if None *[batch_shape,h,w,3]* (Default value = None)
    batch_shape
        Shape of batch. Assumed no batches if None. (Default value = None)
    dev_str
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Quantized pixel co-ordinates image with d feature channels
        (for depth, rgb, normals etc.) *[batch_shape,h,w,2+d]*,
        maybe the quantized variance, *[batch_shape,h,w,2+d]*, and scatter counter image
        *[batch_shape,h,w,1]*

    """
    # ToDo: make variance fully optional. If not specified,
    #  then do not compute and scatter during function call for better efficiency.
    # config
    if batch_shape is None:
        batch_shape = pixel_coords.shape[:-2]

    if dev_str is None:
        dev_str = ivy.dev(pixel_coords)

    if feat is None:
        d = 0
    else:
        d = feat.shape[-1]
    min_depth_diff = ivy.array([MIN_DEPTH_DIFF], device=dev_str)
    red = "min" if with_db else "sum"

    # shapes as list
    batch_shape = list(batch_shape)
    final_image_dims = list(final_image_dims)
    num_batch_dims = len(batch_shape)

    # variance threshold
    if isinstance(var_threshold, tuple) or isinstance(var_threshold, list):
        ones = ivy.ones(batch_shape + [1, 2 + d, 1])
        var_threshold = ivy.concat(
            (ones * var_threshold[0], ones * var_threshold[1]), axis=-1
        )
    else:
        var_threshold = ivy.reshape(var_threshold, batch_shape + [1, 2 + d, 2])

    # uniform pixel coords
    if uniform_pixel_coords is None:
        uniform_pixel_coords = ivy_svg.create_uniform_pixel_coords_image(
            final_image_dims, batch_shape, dev_str=dev_str
        )
    uniform_pixel_coords = uniform_pixel_coords[..., 0:2]

    # Extract Values #

    feat_prior = (
        ivy.ones_like(feat) * feat_prior
        if isinstance(feat_prior, float)
        else feat_prior
    )
    pixel_coords_var = (
        ivy.ones_like(pixel_coords) * pixel_coords_var
        if isinstance(pixel_coords_var, float)
        else pixel_coords_var
    )
    feat_var = (
        ivy.ones_like(feat) * feat_var if isinstance(feat_var, float) else feat_var
    )
    pixel_coords_prior_var = (
        ivy.ones(batch_shape + final_image_dims + [2]) * pixel_coords_prior_var
        if isinstance(pixel_coords_prior_var, float)
        else pixel_coords_prior_var
    )
    feat_prior_var = (
        ivy.ones(batch_shape + final_image_dims + [d]) * feat_prior_var
        if isinstance(feat_prior_var, float)
        else feat_prior_var
    )

    # Quantize #

    # BS x N x 2
    quantized_pixel_coords = ivy.reshape(
        ivy.astype(ivy.round(pixel_coords), "int32"), batch_shape + [-1, 2]
    )

    # Combine #

    # BS x N x (2+D)
    pc_n_feat = ivy.reshape(
        ivy.concat((pixel_coords, feat), axis=-1), batch_shape + [-1, 2 + d]
    )
    pc_n_feat_var = ivy.reshape(
        ivy.concat((pixel_coords_var, feat_var), axis=-1), batch_shape + [-1, 2 + d]
    )

    # BS x H x W x (2+D)
    prior = ivy.concat((uniform_pixel_coords, feat_prior), axis=-1)
    prior_var = ivy.concat((pixel_coords_prior_var, feat_prior_var), axis=-1)

    # Validity Mask #

    # BS x N x 1
    var_validity_mask = (
        ivy.sum(
            ivy.astype(pc_n_feat_var < var_threshold[..., 1], "int32"),
            axis=-1,
            keepdims=True,
        )
        == 2 + d
    )
    bounds_validity_mask = ivy.logical_and(
        ivy.logical_and(
            quantized_pixel_coords[..., 0:1] >= 0, quantized_pixel_coords[..., 1:2] >= 0
        ),
        ivy.logical_and(
            quantized_pixel_coords[..., 0:1] <= final_image_dims[1] - 1,
            quantized_pixel_coords[..., 1:2] <= final_image_dims[0] - 1,
        ),
    )
    validity_mask = ivy.logical_and(var_validity_mask, bounds_validity_mask)

    # num_valid_indices x len(BS)+2
    validity_indices = ivy.reshape(
        ivy.astype(ivy.argwhere(validity_mask), "int32"), [-1, num_batch_dims + 2]
    )
    num_valid_indices = validity_indices.shape[-2]

    if num_valid_indices == 0:
        return (
            ivy.concat((uniform_pixel_coords[..., 0:2], feat_prior), axis=-1),
            ivy.concat((pixel_coords_prior_var, feat_prior_var), axis=-1),
            ivy.zeros_like(feat[..., 0:1], device=dev_str),
        )

    # Depth Based Scaling #

    mean_depth_min = None
    mean_depth_range = None
    pc_n_feat_wo_depth_range = None
    pc_n_feat_wo_depth_min = None
    var_vals_range = None
    var_vals_min = None

    if with_db:
        # BS x N x 1
        mean_depth = pc_n_feat[..., 2:3]

        # BS x 1 x 1
        mean_depth_min = ivy.min(mean_depth, axis=-2, keepdims=True)
        mean_depth_max = ivy.max(mean_depth, axis=-2, keepdims=True)
        mean_depth_range = mean_depth_max - mean_depth_min

        # BS x N x 1
        scaled_depth = (mean_depth - mean_depth_min) / (
            mean_depth_range * min_depth_diff + MIN_DENOMINATOR
        )

        if d == 1:
            # BS x 1 x 1+D
            pc_n_feat_wo_depth_min = ivy.zeros(batch_shape + [1, 0], device=dev_str)
            pc_n_feat_wo_depth_range = ivy.ones(batch_shape + [1, 0], device=dev_str)

        else:
            # feat without depth

            # BS x N x 1+D
            pc_n_feat_wo_depth = ivy.concat(
                (pc_n_feat[..., 0:2], pc_n_feat[..., 3:]), axis=-1
            )

            # find the min and max of each value

            # BS x 1 x 1+D
            pc_n_feat_wo_depth_max = (
                ivy.max(pc_n_feat_wo_depth, axis=-2, keepdims=True) + 1
            )
            pc_n_feat_wo_depth_min = (
                ivy.min(pc_n_feat_wo_depth, axis=-2, keepdims=True) - 1
            )
            pc_n_feat_wo_depth_range = pc_n_feat_wo_depth_max - pc_n_feat_wo_depth_min

            # BS x N x 1+D
            normed_pc_n_feat_wo_depth = (
                pc_n_feat_wo_depth - pc_n_feat_wo_depth_min
            ) / (pc_n_feat_wo_depth_range + MIN_DENOMINATOR)

            # combine with scaled depth

            # BS x N x 1+D
            pc_n_feat_wo_depth_scaled = normed_pc_n_feat_wo_depth + scaled_depth

            # BS x N x (2+D)
            pc_n_feat = ivy.concat(
                (
                    pc_n_feat_wo_depth_scaled[..., 0:2],
                    mean_depth,
                    pc_n_feat_wo_depth_scaled[..., 2:],
                ),
                axis=-1,
            )

        # scale variance

        # BS x 1 x (2+D)
        var_vals_max = ivy.max(pc_n_feat_var, axis=-2, keepdims=True) + 1
        var_vals_min = ivy.min(pc_n_feat_var, axis=-2, keepdims=True) - 1
        var_vals_range = var_vals_max - var_vals_min

        # BS x N x (2+D)
        normed_var_vals = (pc_n_feat_var - var_vals_min) / (
            var_vals_range + MIN_DENOMINATOR
        )
        pc_n_feat_var = normed_var_vals + scaled_depth

        # ready for later reversal with full image dimensions

        # BS x 1 x 1 x D
        var_vals_min = ivy.expand_dims(var_vals_min, axis=-2)
        var_vals_range = ivy.expand_dims(var_vals_range, axis=-2)

    # Validity Pruning #

    # num_valid_indices x (2+D)
    pc_n_feat = ivy.gather_nd(pc_n_feat, validity_indices[..., 0 : num_batch_dims + 1])
    pc_n_feat_var = ivy.gather_nd(
        pc_n_feat_var, validity_indices[..., 0 : num_batch_dims + 1]
    )

    # num_valid_indices x 2
    quantized_pixel_coords = ivy.gather_nd(
        quantized_pixel_coords, validity_indices[..., 0 : num_batch_dims + 1]
    )

    if with_db:
        means_to_scatter = pc_n_feat
        vars_to_scatter = pc_n_feat_var
    else:
        # num_valid_indices x (2+D)
        vars_to_scatter = 1 / (pc_n_feat_var + MIN_DENOMINATOR)
        means_to_scatter = pc_n_feat * vars_to_scatter

    # Scatter #

    # num_valid_indices x 1
    counter = ivy.ones_like(pc_n_feat[..., 0:1], device=dev_str)
    if with_db:
        counter *= -1

    # num_valid_indices x 2(2+D)+1
    values_to_scatter = ivy.concat(
        (means_to_scatter, vars_to_scatter, counter), axis=-1
    )

    # num_valid_indices x (num_batch_dims + 2)
    all_indices = ivy.flip(quantized_pixel_coords, axis=-1)
    if num_batch_dims > 0:
        all_indices = ivy.concat((validity_indices[..., :-2], all_indices), axis=-1)

    # BS x H x W x (2(2+D) + 1)
    quantized_img = ivy.scatter_nd(
        ivy.reshape(all_indices, [-1, num_batch_dims + 2]),
        ivy.reshape(values_to_scatter, [-1, 2 * (2 + d) + 1]),
        shape=batch_shape + final_image_dims + [2 * (2 + d) + 1],
        reduction=red,
    )

    # BS x H x W x 1
    quantized_counter = quantized_img[..., -1:]
    if with_db:
        invalidity_mask = quantized_counter != -1
    else:
        invalidity_mask = quantized_counter == 0

    if with_db:
        # BS x H x W x (2+D)
        quantized_mean_scaled = quantized_img[..., 0 : 2 + d]
        quantized_var_scaled = quantized_img[..., 2 + d : 2 * (2 + d)]

        # BS x H x W x 1
        quantized_depth_mean = quantized_mean_scaled[..., 2:3]

        # BS x 1 x 1 x 1
        mean_depth_min = ivy.expand_dims(mean_depth_min, axis=-2)
        mean_depth_range = ivy.expand_dims(mean_depth_range, axis=-2)

        # BS x 1 x 1 x (1+D)
        pc_n_feat_wo_depth_min = ivy.expand_dims(pc_n_feat_wo_depth_min, axis=-2)
        pc_n_feat_wo_depth_range = ivy.expand_dims(pc_n_feat_wo_depth_range, axis=-2)

        # BS x 1 x 1 x (2+D) x 2
        var_threshold = ivy.expand_dims(var_threshold, axis=-3)

        # BS x H x W x (1+D)
        quantized_mean_wo_depth_scaled = ivy.concat(
            (quantized_mean_scaled[..., 0:2], quantized_mean_scaled[..., 3:]), axis=-1
        )
        quantized_mean_wo_depth_normed = quantized_mean_wo_depth_scaled - (
            quantized_depth_mean - mean_depth_min
        ) / (mean_depth_range * min_depth_diff + MIN_DENOMINATOR)
        quantized_mean_wo_depth = (
            quantized_mean_wo_depth_normed * pc_n_feat_wo_depth_range
            + pc_n_feat_wo_depth_min
        )
        prior_wo_depth = ivy.concat((prior[..., 0:2], prior[..., 3:]), axis=-1)
        quantized_mean_wo_depth = ivy.where(
            invalidity_mask, prior_wo_depth, quantized_mean_wo_depth
        )

        # BS x H x W x (2+D)
        quantized_mean = ivy.concat(
            (
                quantized_mean_wo_depth[..., 0:2],
                quantized_depth_mean,
                quantized_mean_wo_depth[..., 2:],
            ),
            axis=-1,
        )

        # BS x H x W x (2+D)
        quantized_var_normed = quantized_var_scaled - (
            quantized_depth_mean - mean_depth_min
        ) / (mean_depth_range * min_depth_diff + MIN_DENOMINATOR)
        quantized_var = ivy.maximum(
            quantized_var_normed * var_vals_range + var_vals_min, var_threshold[..., 0]
        )
        quantized_var = ivy.where(invalidity_mask, prior_var, quantized_var)
    else:
        # BS x H x W x (2+D)
        quantized_sum_mean_x_recip_var = quantized_img[..., 0 : 2 + d]
        quantized_var_wo_increase = ivy.where(
            invalidity_mask,
            prior_var,
            (1 / (quantized_img[..., 2 + d : 2 * (2 + d)] + MIN_DENOMINATOR)),
        )
        quantized_var = ivy.maximum(
            quantized_var_wo_increase * quantized_counter,
            ivy.expand_dims(var_threshold[..., 0], axis=-2),
        )
        quantized_var = ivy.where(invalidity_mask, prior_var, quantized_var)
        quantized_mean = ivy.where(
            invalidity_mask,
            prior,
            quantized_var_wo_increase * quantized_sum_mean_x_recip_var,
        )

    # BS x H x W x (2+D)    BS x H x W x (2+D)     BS x H x W x 1
    return quantized_mean, quantized_var, quantized_counter
