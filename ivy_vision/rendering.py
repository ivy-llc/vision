"""
Collection of Rendering Functions
"""

# global
import ivy as _ivy
import math as _math
from functools import reduce as _reduce
from operator import mul as _mul

# local
from ivy_vision import single_view_geometry as _ivy_svg

MIN_DENOMINATOR = 1e-12
MIN_DEPTH_DIFF = 1e-2


# ToDo: refactor the various render implementations, to reduce duplicate code blocks


def _quantize_proj_pixel_coords_with_var(pixel_coords, prior, final_image_dims, pixel_coords_var, prior_var,
                                         var_threshold, uniform_pixel_coords, batch_shape, dev_str):
    # shapes
    num_batch_dims = len(batch_shape)
    d = prior.shape[-1] - 1

    # Quantization #

    # BS x N x (1+D)
    mean_vals = _ivy.reshape(pixel_coords[..., 2:], batch_shape + [-1, 1 + d])

    # BS x N x 1
    mean_depth = mean_vals[..., 0:1]

    # BS x N x 2
    pixel_xy_coords = _ivy.reshape(pixel_coords[..., 0:2], batch_shape + [-1, 2]) / (mean_depth + MIN_DENOMINATOR)

    # BS x N x 2
    quantized_pixel_xy_coords = _ivy.cast(_ivy.round(pixel_xy_coords), 'int32')

    # BS x N x (1+D)
    var_vals = _ivy.reshape(pixel_coords_var, batch_shape + [-1, 1 + d])

    # Validity Mask #

    # BS x N x 1
    var_validity_mask = \
        _ivy.reduce_sum(_ivy.cast(var_vals < var_threshold[..., 1], 'int32'), -1, keepdims=True) == d + 1
    bounds_validity_mask = _ivy.logical_and(
        _ivy.logical_and(quantized_pixel_xy_coords[..., 0:1] >= 0, quantized_pixel_xy_coords[..., 1:2] >= 0),
        _ivy.logical_and(quantized_pixel_xy_coords[..., 0:1] <= final_image_dims[1] - 1,
                         quantized_pixel_xy_coords[..., 1:2] <= final_image_dims[0] - 1)
    )
    validity_mask = _ivy.logical_and(var_validity_mask, bounds_validity_mask)

    # num_valid_indices x len(BS)+2
    validity_indices = _ivy.reshape(_ivy.cast(_ivy.indices_where(validity_mask), 'int32'), [-1, num_batch_dims + 2])
    num_valid_indices = validity_indices.shape[-2]

    if num_valid_indices == 0:
        return _ivy.concatenate((uniform_pixel_coords[..., 0:2], prior), -1), \
               prior_var, _ivy.zeros_like(prior_var[..., 0:1], dev_str=dev_str)

    # Validity Pruning #

    # num_valid_indices x (1+D)
    var_vals = _ivy.gather_nd(var_vals, validity_indices[..., 0:num_batch_dims + 1])
    mean_vals = _ivy.gather_nd(mean_vals, validity_indices[..., 0:num_batch_dims + 1])

    # num_valid_indices x 2
    quantized_pixel_xy_coords = _ivy.gather_nd(quantized_pixel_xy_coords, validity_indices[..., 0:num_batch_dims + 1])

    # num_valid_indices x (1+D)
    recip_vars = 1 / (var_vals + MIN_DENOMINATOR)
    means_x_recip_vars = mean_vals * recip_vars

    # Scatter #

    # num_valid_indices x 2(1+D)+1
    values_to_scatter = _ivy.concatenate((means_x_recip_vars, recip_vars,
                                          _ivy.ones_like(mean_vals[..., 0:1], dev_str=dev_str)), -1)

    # num_valid_indices x (num_batch_dims + 2)
    if num_batch_dims == 0:
        all_indices = _ivy.flip(quantized_pixel_xy_coords, -1)
    else:
        all_indices = _ivy.concatenate((validity_indices[..., :-2], _ivy.flip(quantized_pixel_xy_coords, -1)), -1)

    # BS x H x W x (2(1+D) + 1)
    quantized_img = _ivy.scatter_nd(_ivy.reshape(all_indices, [-1, num_batch_dims + 2]),
                                    _ivy.reshape(values_to_scatter, [-1, 2 * (1 + d) + 1]),
                                    batch_shape + final_image_dims + [2 * (1 + d) + 1],
                                    reduction='replace' if _ivy.backend == 'mxnd' else 'sum')

    # BS x H x W x 1
    quantized_counter = quantized_img[..., -1:]
    invalidity_mask = quantized_counter == 0

    # BS x H x W x (1+D)
    quantized_sum_mean_x_recip_var = quantized_img[..., 0:1 + d]

    # BS x H x W x D
    quantized_var_wo_increase = _ivy.where(invalidity_mask, prior_var,
                                           (1 / (quantized_img[..., 1 + d:2 * (1 + d)] + MIN_DENOMINATOR)))
    quantized_var = _ivy.maximum(quantized_var_wo_increase * quantized_counter,
                                 _ivy.expand_dims(var_threshold[..., 0], -2))
    quantized_var = _ivy.where(invalidity_mask, prior_var, quantized_var)
    quantized_mean = _ivy.where(invalidity_mask, prior, quantized_var_wo_increase * quantized_sum_mean_x_recip_var)

    # BS x H x W x 1
    quantized_depth_mean = quantized_mean[..., 0:1]

    # BS x H x W x 3
    quantized_pixel_coords = uniform_pixel_coords * quantized_depth_mean

    # BS x H x W x (1+D)
    if d == 0:
        quantized_mean = quantized_pixel_coords
    else:
        quantized_mean = _ivy.concatenate((quantized_pixel_coords, quantized_mean[..., 1:]), -1)

    # BS x H x W x (3+D)    BS x H x W x (1+D)     BS x H x W x 1
    return quantized_mean, quantized_var, quantized_counter


def _quantize_proj_pixel_coords_with_depth_buffer_and_var(pixel_coords, prior, final_image_dims, pixel_coords_var,
                                                          prior_var, var_threshold, uniform_pixel_coords, batch_shape,
                                                          dev_str):
    # shapes
    num_batch_dims = len(batch_shape)
    d = prior.shape[-1] - 1
    min_depth_diff = _ivy.array(MIN_DEPTH_DIFF)

    # Quantization #
    # -------------#

    # BS x N x (1+D)
    mean_vals = _ivy.reshape(pixel_coords[..., 2:], batch_shape + [-1, 1 + d])

    # BS x N x 1
    mean_depth = mean_vals[..., 0:1]

    # BS x N x 2
    pixel_xy_coords = _ivy.reshape(pixel_coords[..., 0:2], batch_shape + [-1, 2]) / (mean_depth + MIN_DENOMINATOR)

    # BS x N x 2
    quantized_pixel_xy_coords = _ivy.round(pixel_xy_coords)

    # BS x N x 2
    quantized_pixel_xy_coords = _ivy.cast(quantized_pixel_xy_coords, 'int32')

    # BS x N x (1+D)
    var_vals = _ivy.reshape(pixel_coords_var, batch_shape + [-1, 1 + d])

    # Validity Mask #
    # --------------#

    # BS x N x 1
    var_validity_mask = \
        _ivy.reduce_sum(_ivy.cast(var_vals < var_threshold[..., 1], 'int32'), -1, keepdims=True) == d + 1
    bounds_validity_mask = _ivy.logical_and(
        _ivy.logical_and(quantized_pixel_xy_coords[..., 0:1] >= 0, quantized_pixel_xy_coords[..., 1:2] >= 0),
        _ivy.logical_and(quantized_pixel_xy_coords[..., 0:1] <= final_image_dims[1] - 1,
                         quantized_pixel_xy_coords[..., 1:2] <= final_image_dims[0] - 1)
    )
    validity_mask = _ivy.logical_and(var_validity_mask, bounds_validity_mask)

    # num_valid_indices x len(BS)+2
    validity_indices = _ivy.reshape(_ivy.cast(_ivy.indices_where(
        validity_mask), 'int32'), [-1, num_batch_dims + 2])
    num_valid_indices = validity_indices.shape[0]

    if num_valid_indices == 0:
        return _ivy.concatenate((uniform_pixel_coords[..., 0:2], prior), -1), \
               prior_var, _ivy.zeros_like(prior_var[..., 0:1], dev_str=dev_str)

    # Depth Based Scaling #
    # --------------------#

    # BS x N x 1
    mean_depth = mean_vals[..., 0:1]

    # BS x 1 x 1
    mean_depth_min = _ivy.reduce_min(mean_depth, -2, keepdims=True)
    mean_depth_max = _ivy.reduce_max(mean_depth, -2, keepdims=True)
    mean_depth_range = mean_depth_max - mean_depth_min

    # BS x N x 1
    scaled_depth = (mean_depth - mean_depth_min) / (mean_depth_range * min_depth_diff + MIN_DENOMINATOR)

    if d != 0:
        # means vals without depth

        # BS x N x D
        mean_vals_wo_depth = mean_vals[..., 1:]

        # find the min and max of each value

        # BS x 1 x D
        mean_vals_wo_depth_max = _ivy.reduce_max(mean_vals_wo_depth, -2, keepdims=True) + 1
        mean_vals_wo_depth_min = _ivy.reduce_min(mean_vals_wo_depth, -2, keepdims=True) - 1
        mean_vals_wo_depth_range = mean_vals_wo_depth_max - mean_vals_wo_depth_min

        # BS x N x D
        normed_mean_vals_wo_depth = (mean_vals_wo_depth - mean_vals_wo_depth_min) / \
                                    (mean_vals_wo_depth_range + MIN_DENOMINATOR)

        # combine with scaled depth

        # BS x N x D
        mean_vals_wo_depth_scaled = normed_mean_vals_wo_depth + scaled_depth

        # BS x N x (1+D)
        mean_vals_scaled = _ivy.concatenate((mean_depth, mean_vals_wo_depth_scaled), -1)

    else:

        # BS x 1 x D
        mean_vals_wo_depth_min = _ivy.zeros(batch_shape + [1, d], dev_str=dev_str)
        mean_vals_wo_depth_range = _ivy.ones(batch_shape + [1, d], dev_str=dev_str)

        # BS x N x (1+D)
        mean_vals_scaled = mean_vals

    # scale variance

    # num_valid_indices x (1+D)
    var_vals_max = _ivy.reduce_max(var_vals, -2, keepdims=True) + 1
    var_vals_min = _ivy.reduce_min(var_vals, -2, keepdims=True) - 1
    var_vals_range = var_vals_max - var_vals_min

    # num_valid_indices x (1+D)
    normed_var_vals = (var_vals - var_vals_min) / (var_vals_range + MIN_DENOMINATOR)
    var_vals_scaled = normed_var_vals + scaled_depth

    # Validity Pruning #
    # -----------------#

    # num_valid_indices x (1+D)
    var_vals_scaled = _ivy.gather_nd(var_vals_scaled, validity_indices[..., 0:num_batch_dims + 1])
    mean_vals_scaled = _ivy.gather_nd(mean_vals_scaled, validity_indices[..., 0:num_batch_dims + 1])

    # num_valid_indices x 2
    quantized_pixel_xy_coords = _ivy.gather_nd(quantized_pixel_xy_coords,
                                               validity_indices[..., 0:num_batch_dims + 1])

    # Scatter #
    # --------#

    # num_valid_indices x 1
    minus_ones = -_ivy.ones_like(mean_vals_scaled[..., 0:1], dev_str=dev_str)

    # num_valid_indices x 2(1+D)+1
    values_to_scatter = _ivy.concatenate((mean_vals_scaled, var_vals_scaled, minus_ones), -1)

    # num_valid_indices x (num_batch_dims + 2)
    all_indices = _ivy.concatenate((validity_indices[..., :-2], _ivy.flip(quantized_pixel_xy_coords, -1)), -1)

    # BS x H x W x 2(1+D)+1
    quantized_img = _ivy.scatter_nd(_ivy.reshape(all_indices, [-1, num_batch_dims + 2]),
                                    _ivy.reshape(values_to_scatter, [-1, 2 * (1 + d) + 1]),
                                    batch_shape + final_image_dims + [2 * (1 + d) + 1], reduction='min')

    # BS x H x W x (1+D)
    quantized_mean_scaled = quantized_img[..., 0:1 + d]
    quantized_var_scaled = quantized_img[..., 1 + d:2 * (1 + d)]

    # BS x H x W x 1
    validity_mask = quantized_img[..., -1:] == -1

    # BS x H x W x 1
    quantized_depth_mean = quantized_mean_scaled[..., 0:1]

    # BS x H x W x D

    quantized_mean_wo_depth_normed = quantized_mean_scaled[..., 1:] - (quantized_depth_mean - mean_depth_min) / \
                                     (mean_depth_range * min_depth_diff + MIN_DENOMINATOR)
    quantized_mean_wo_depth = quantized_mean_wo_depth_normed * mean_vals_wo_depth_range + mean_vals_wo_depth_min
    quantized_mean_wo_depth = _ivy.where(validity_mask, quantized_mean_wo_depth, prior[..., 1:])

    # BS x H x W x 3
    quantized_pixel_coords = uniform_pixel_coords * quantized_depth_mean

    # BS x H x W x (3+D)
    quantized_mean = _ivy.concatenate((quantized_pixel_coords, quantized_mean_wo_depth), -1)

    # BS x H x W x (1+D)
    quantized_var_normed = quantized_var_scaled - (quantized_depth_mean - mean_depth_min) / \
                           (mean_depth_range * min_depth_diff + MIN_DENOMINATOR)
    quantized_var = _ivy.maximum(quantized_var_normed * var_vals_range + var_vals_min, var_threshold[..., 0])
    quantized_var = _ivy.where(validity_mask, quantized_var, prior_var)

    # BS x H x W x (3+D)    BS x H x W x (1+D)     BS x H x W x 1
    return quantized_mean, quantized_var, validity_mask


def _quantize_omni_pixel_coords_with_var(pixel_coords, prior, final_image_dims, pixel_coords_var, prior_var,
                                         var_threshold, uniform_pixel_coords, batch_shape, dev_str):
    # shapes
    num_batch_dims = len(batch_shape)
    d = prior.shape[-1]

    # Quantization #
    # -------------#

    # BS x N x 2
    pixel_xy_coords = _ivy.reshape(pixel_coords[..., 0:2], batch_shape + [-1, 2])

    # BS x N x 2
    quantized_pixel_xy_coords = _ivy.floormod(_ivy.round(pixel_xy_coords),
                                              _ivy.array([float(final_image_dims[1]),
                                                          float(final_image_dims[0])], dev_str=dev_str))

    # BS x N x 2
    quantized_pixel_xy_coords = _ivy.cast(quantized_pixel_xy_coords, 'int32')

    # BS x N x D
    scatter_vals = _ivy.reshape(pixel_coords[..., 2:], batch_shape + [-1, d])
    scatter_val_vars = _ivy.reshape(pixel_coords_var, batch_shape + [-1, d])

    # Validity Mask #
    # --------------#

    # BS x N x 1
    validity_mask = _ivy.reduce_sum(_ivy.cast(scatter_val_vars < var_threshold[..., 1], 'int32'), -1,
                                    keepdims=True) == d

    # num_valid_indices x len(BS)+2
    validity_indices = _ivy.reshape(_ivy.cast(_ivy.indices_where(
        validity_mask), 'int32'), [-1, num_batch_dims + 2])
    num_valid_indices = validity_indices.shape[0]

    if num_valid_indices == 0:
        return _ivy.concatenate((uniform_pixel_coords[..., 0:2], prior), -1), \
               prior_var, _ivy.zeros_like(prior_var[..., 0:1], dev_str=dev_str)

    # Validity Pruning #
    # -----------------#

    # num_valid_indices x D
    scatter_val_vars = _ivy.gather_nd(scatter_val_vars, validity_indices[..., 0:num_batch_dims + 1])
    scatter_vals = _ivy.gather_nd(scatter_vals, validity_indices[..., 0:num_batch_dims + 1])

    # num_valid_indices x 2
    quantized_pixel_xy_coords = _ivy.gather_nd(quantized_pixel_xy_coords,
                                               validity_indices[..., 0:num_batch_dims + 1])

    # num_valid_indices x D
    recip_scatter_val_vars = 1 / (scatter_val_vars + MIN_DENOMINATOR)
    scatter_vals_times_recip_vars = scatter_vals * recip_scatter_val_vars

    # Scatter #
    # --------#

    # num_valid_indices x (D+D+1)
    scatter_vals_and_var_and_counter = \
        _ivy.concatenate((scatter_vals_times_recip_vars, recip_scatter_val_vars,
                          _ivy.ones_like(scatter_vals[..., 0:1], dev_str=dev_str)), -1)

    # num_valid_indices x (num_batch_dims + 2)
    all_indices = _ivy.concatenate((validity_indices[..., :-2],
                                    _ivy.flip(quantized_pixel_xy_coords, -1)), -1)

    # BS x H x W x (D+D+1)
    quantized_img = _ivy.scatter_nd(_ivy.reshape(all_indices, [-1, num_batch_dims + 2]),
                                    _ivy.reshape(scatter_vals_and_var_and_counter, [-1, d + d + 1]),
                                    batch_shape + final_image_dims + [d + d + 1],
                                    reduction='replace' if _ivy.backend == 'mxnd' else 'sum')

    # BS x H x W x 1
    quantized_counter = quantized_img[..., -1:]
    quantized_sum_scat_vals_x_recip_var = quantized_img[..., 0:d]
    invalidity_mask = quantized_counter == 0
    quantized_scat_vals_var_wo_increase = _ivy.where(invalidity_mask, prior_var,
                                                     (1 / (quantized_img[..., d:-1] + MIN_DENOMINATOR)))
    quantized_scat_vals_var = _ivy.maximum(quantized_scat_vals_var_wo_increase * quantized_counter,
                                           var_threshold[..., 0])
    quantized_scat_vals_var = _ivy.where(invalidity_mask, prior_var, quantized_scat_vals_var)
    quantized_scat_vals_mean = _ivy.where(invalidity_mask, prior,
                                          quantized_scat_vals_var_wo_increase * quantized_sum_scat_vals_x_recip_var)

    # BS x H x W x (2+D)
    quantized_pixel_coords = _ivy.concatenate((uniform_pixel_coords[..., 0:2], quantized_scat_vals_mean), -1)

    # BS x H x W x (2+D)    BS x H x W x D    BS x H x W x 1
    return quantized_pixel_coords, quantized_scat_vals_var, quantized_counter


def _quantize_omni_pixel_coords_with_depth_buffer_and_var(pixel_coords, prior, final_image_dims, pixel_coords_var,
                                                          prior_var, var_threshold, uniform_pixel_coords,
                                                          batch_shape, dev_str):
    # shapes as list
    batch_shape = list(batch_shape)
    num_batch_dims = len(batch_shape)
    final_image_dims = list(final_image_dims)
    input_size = pixel_coords.shape[-2]
    d = prior.shape[-1]
    min_depth_diff = _ivy.array(MIN_DEPTH_DIFF)

    # Quantization #
    # -------------#

    # BS x N x 2
    pixel_xy_coords = _ivy.reshape(pixel_coords[..., 0:2], batch_shape + [input_size, 2])

    # BS x N x 2
    quantized_pixel_xy_coords = _ivy.floormod(_ivy.round(pixel_xy_coords),
                                              _ivy.array([float(final_image_dims[1]),
                                                          float(final_image_dims[0])], dev_str=dev_str))

    # BS x N x 2
    quantized_pixel_xy_coords = _ivy.cast(quantized_pixel_xy_coords, 'int32')

    # BS x N x D
    mean_vals = _ivy.reshape(pixel_coords[..., 2:], batch_shape + [input_size, d])
    var_vals = _ivy.reshape(pixel_coords_var, batch_shape + [input_size, d])

    # Validity Mask #
    # --------------#

    # num_valid_indices x 1
    validity_mask = _ivy.reduce_sum(_ivy.cast(var_vals < var_threshold[..., 0, :, 1], 'int32'), -1, keepdims=True) == d

    # num_valid_indices x len(BS)+2
    validity_indices = _ivy.reshape(_ivy.cast(_ivy.indices_where(
        validity_mask), 'int32'), [-1, num_batch_dims + 2])
    num_valid_indices = _ivy.shape(validity_indices, True)[0]

    if _ivy.reduce_sum(_ivy.array(num_valid_indices)) == _ivy.array(0):
        return _ivy.concatenate((uniform_pixel_coords[..., 0:2], prior), -1), \
               prior_var, _ivy.cast(_ivy.zeros_like(prior_var[..., 0:1], dev_str=dev_str), 'bool')

    # Depth Based Scaling #
    # --------------------#

    # BS x N x 1
    mean_depth = mean_vals[..., 0:1]

    # BS x 1 x 1
    mean_depth_min = _ivy.expand_dims(_ivy.reduce_min(mean_depth, -2, keepdims=True), -1)
    mean_depth_max = _ivy.expand_dims(_ivy.reduce_max(mean_depth, -2, keepdims=True), -1)
    mean_depth_range = mean_depth_max - mean_depth_min

    # BS x N x 1
    scaled_depth = (mean_depth - mean_depth_min[..., 0]) / (mean_depth_range[..., 0] * min_depth_diff + MIN_DENOMINATOR)

    if d > 1:

        # scale means vals after depth channel

        # BS x N x (D-1)
        mean_vals_wo_depth = mean_vals[..., 1:]

        # find the min and max of each value

        # BS x 1 x (D-1)
        mean_vals_wo_depth_max = _ivy.reduce_max(mean_vals_wo_depth, -2, keepdims=True) + 1
        mean_vals_wo_depth_min = _ivy.reduce_min(mean_vals_wo_depth, -2, keepdims=True) - 1
        mean_vals_wo_depth_range = mean_vals_wo_depth_max - mean_vals_wo_depth_min

        # BS x N x (D-1)
        normed_mean_vals_wo_depth = (mean_vals_wo_depth - mean_vals_wo_depth_min) / \
                                    (mean_vals_wo_depth_range + MIN_DENOMINATOR)

        # combine with scaled depth

        # BS x N x (D-1)
        mean_vals_wo_depth_scaled = normed_mean_vals_wo_depth + scaled_depth

        # BS x N x D
        mean_vals_scaled = _ivy.concatenate((mean_depth, mean_vals_wo_depth_scaled), -1)

        # ready for later reversal with full image dimensions

        # BS x 1 x 1 x (D-1)
        mean_vals_wo_depth_min = _ivy.expand_dims(mean_vals_wo_depth_min, -2)
        mean_vals_wo_depth_range = _ivy.expand_dims(mean_vals_wo_depth_range, -2)

    else:

        # ready for later reversal with full image dimensions

        # BS x 1 x 1 x (D-1)
        mean_vals_wo_depth_min = _ivy.zeros(batch_shape + [1, 1, d], dev_str=dev_str)
        mean_vals_wo_depth_range = _ivy.ones(batch_shape + [1, 1, d], dev_str=dev_str)

        # BS x N x D
        mean_vals_scaled = mean_vals

    # scale variance

    # BS x 1 x D
    var_vals_max = _ivy.reduce_max(var_vals, -2, keepdims=True) + 1
    var_vals_min = _ivy.reduce_min(var_vals, -2, keepdims=True) - 1
    var_vals_range = var_vals_max - var_vals_min

    # BS x N x D
    normed_var_vals = (var_vals - var_vals_min) / (var_vals_range + MIN_DENOMINATOR)
    var_vals_scaled = normed_var_vals + scaled_depth

    # ready for later reversal with full image dimensions

    # BS x 1 x 1 x D
    var_vals_min = _ivy.expand_dims(var_vals_min, -2)
    var_vals_range = _ivy.expand_dims(var_vals_range, -2)

    # Validity Pruning #
    # -----------------#

    # num_valid_indices x D
    mean_vals_scaled = _ivy.gather_nd(mean_vals_scaled, validity_indices[..., 0:num_batch_dims + 1])
    var_vals_scaled = _ivy.gather_nd(var_vals_scaled, validity_indices[..., 0:num_batch_dims + 1])

    # num_valid_indices x 2
    quantized_pixel_xy_coords = _ivy.gather_nd(quantized_pixel_xy_coords,
                                               validity_indices[..., 0:num_batch_dims + 1])

    # num_valid_indices x 1
    minus_ones = -_ivy.ones_like(mean_vals_scaled[..., 0:1], dev_str=dev_str)

    # Scatter #
    # --------#

    # num_valid_indices x (2D+1)
    values_to_scatter = _ivy.concatenate((mean_vals_scaled, var_vals_scaled, minus_ones), -1)

    # num_valid_indices x (num_batch_dims + 2)
    all_indices = _ivy.concatenate((validity_indices[..., :-2], _ivy.flip(quantized_pixel_xy_coords, -1)), -1)

    # BS x H x W x (D+D+1)
    quantized_img = _ivy.scatter_nd(_ivy.reshape(all_indices, [-1, num_batch_dims + 2]),
                                    _ivy.reshape(values_to_scatter, [-1, d + d + 1]),
                                    batch_shape + final_image_dims + [d + d + 1], reduction='min')

    # BS x H x W x D
    quantized_mean_scaled = quantized_img[..., 0:d]
    quantized_var_scaled = quantized_img[..., d:d + d]

    # BS x H x W x 1
    validity_mask = quantized_img[..., -1:] == -1
    quantized_depth_mean = quantized_mean_scaled[..., 0:1]

    # BS x H x W x (D-1)
    quantized_mean_wo_depth_normed = quantized_mean_scaled[..., 1:] - (quantized_depth_mean - mean_depth_min) / \
                                     (mean_depth_range * min_depth_diff + MIN_DENOMINATOR)
    quantized_mean_wo_depth = quantized_mean_wo_depth_normed * mean_vals_wo_depth_range + mean_vals_wo_depth_min

    # BS x H x W x D
    quantized_mean = _ivy.concatenate((quantized_depth_mean, quantized_mean_wo_depth), -1)
    quantized_mean = _ivy.where(validity_mask, quantized_mean, prior)

    # BS x H x W x (2+D)
    quantized_pixel_coords = _ivy.concatenate((uniform_pixel_coords[..., 0:2], quantized_mean), -1)

    # BS x H x W x D
    quantized_var_normed = quantized_var_scaled - (quantized_depth_mean - mean_depth_min) / \
                           (mean_depth_range * min_depth_diff + MIN_DENOMINATOR)
    quantized_var = _ivy.maximum(quantized_var_normed * var_vals_range + var_vals_min, var_threshold[..., 0])
    quantized_var = _ivy.where(validity_mask, quantized_var, prior_var)

    # BS x H x W x (2+D)    BS x H x W x D    BS x H x W x 1
    return quantized_pixel_coords, quantized_var, validity_mask


RENDER_METHODS = {'proj':
                      {True: _quantize_proj_pixel_coords_with_depth_buffer_and_var,
                       False: _quantize_proj_pixel_coords_with_var},
                  'omni':
                      {True: _quantize_omni_pixel_coords_with_depth_buffer_and_var,
                       False: _quantize_omni_pixel_coords_with_var}
                  }


def quantize_pixel_coords(pixel_coords, prior, final_image_dims, mode='proj', with_db=False,
                          pixel_coords_var=1e-3, prior_var=1e12, var_threshold=(1e-3, 1e12), uniform_pixel_coords=None,
                          batch_shape=None, dev_str=None):
    """
    Quantize pixel co-ordinates with d feature channels (for depth, rgb, normals etc.), from
    images :math:`\mathbf{X}\in\mathbb{R}^{input\_images\_shape×(2+d)}`, which may have been reprojected from a host of
    different cameras (leading to non-quantized values), to a new quantized pixel co-ordinate image with the same
    feature channels :math:`\mathbf{X}\in\mathbb{R}^{h×w×(3+d)}`, and with integer pixel co-ordinates.
    Duplicates during the quantization are mean averaged.

    :param pixel_coords: Coordinates to scatter (depth, rgb, normals, etc.) *[batch_shape,input_size,2+d]*
    :type pixel_coords: array
    :param prior: Coords prior *[batch_shape,h,w,d]*
    :type prior: array or float to fill with
    :param final_image_dims: Image dimensions of the final image.
    :type final_image_dims: sequence of ints
    :param mode: Rendering mode, be one [proj|omni] for projective or omni-directional rendering, default is proj
    :type mode: str, optional
    :param with_db: Whether or not to use depth buffer in rendering, default is false
    :type with_db: bool, optional
    :param pixel_coords_var: Spherical polar pixel co-ordinates diagonal covariance *[batch_shape,input_size,d]*
    :type pixel_coords_var: array or float to fill with
    :param prior_var: Coords prior diagonal covariance *[batch_shape,h,w,d]*
    :type prior_var: array or float to fill with
    :param var_threshold: Variance threshold, for projecting valid coords and clipping *[batch_shape,d,2]*
    :type var_threshold: array or sequence of floats to fill with
    :param uniform_pixel_coords: Homogeneous uniform (integer) pixel co-ordinate images, inferred from final_image_dims if None *[batch_shape,h,w,3]*
    :type uniform_pixel_coords: array, optional
    :param batch_shape: Shape of batch. Assumed no batches if None.
    :type batch_shape: sequence of ints, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev_str: str, optional
    :return: Quantized pixel co-ordinates image with d feature channels (for depth, rgb, normals etc.) *[batch_shape,h,w,2+d]* with other d scattered, and scatter counter image *[batch_shape,h,w,1]*
    """

    if batch_shape is None:
        batch_shape = pixel_coords.shape[:-2]

    if dev_str is None:
        dev_str = _ivy.dev_str(pixel_coords)

    # shapes as list
    batch_shape = list(batch_shape)
    final_image_dims = list(final_image_dims)
    d = prior.shape[-1] - 1

    if uniform_pixel_coords is None:
        uniform_pixel_coords = _ivy_svg.create_uniform_pixel_coords_image(final_image_dims, batch_shape, dev_str=dev_str)

    # mode
    if isinstance(pixel_coords_var, float):
        pixel_coords_var = _ivy.ones_like(pixel_coords[..., 2:]) * pixel_coords_var
    if isinstance(prior_var, float):
        prior_var = _ivy.ones(batch_shape + final_image_dims + [1 + d]) * prior_var
    if isinstance(var_threshold, tuple) or isinstance(var_threshold, list):
        ones = _ivy.ones(batch_shape + [1, 1 + d, 1])
        var_threshold = _ivy.concatenate((ones * var_threshold[0], ones * var_threshold[1]), -1)
    else:
        var_threshold = _ivy.reshape(var_threshold, batch_shape + [1, 1 + d, 2])

    try:
        return RENDER_METHODS[mode][with_db](
            pixel_coords, prior, final_image_dims, pixel_coords_var, prior_var, var_threshold, uniform_pixel_coords,
            batch_shape, dev_str)
    except KeyError:
        raise Exception('Invalid render method called. Mode must be one of [proj|omni], but found {}'.format(mode))


def rasterize_triangles(pixel_coords_triangles, image_dims, batch_shape=None, dev_str=None):
    """
    Rasterize image-projected triangles
    based on: https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/rasterization-stage
    and: https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/rasterization-practical-implementation

    :param pixel_coords_triangles: Projected image-space triangles to be rasterized
                                    *[batch_shape,input_size,3,3]*
    :type pixel_coords_triangles: array
    :param image_dims: Image dimensions.
    :type image_dims: sequence of ints
    :param batch_shape: Shape of batch. Inferred from Inputs if None.
    :type batch_shape: sequence of ints, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev_str: str, optional
    :return: Rasterized triangles
    """

    if batch_shape is None:
        batch_shape = []

    if dev_str is None:
        dev_str = _ivy.dev_str(pixel_coords_triangles)

    # shapes as list
    batch_shape = list(batch_shape)
    num_batch_dims = len(batch_shape)
    image_dims = list(image_dims)
    input_image_dims = pixel_coords_triangles.shape[num_batch_dims:-2]
    input_image_dims_prod = _reduce(_mul, input_image_dims, 1)

    # BS x 3 x 2
    pixel_xy_coords = pixel_coords_triangles[..., 0:2]

    # BS x 3 x 1
    pixel_x_coords = pixel_coords_triangles[..., 0:1]
    pixel_y_coords = pixel_coords_triangles[..., 1:2]

    # 1
    x_min = _ivy.reshape(_ivy.reduce_min(pixel_x_coords, keepdims=True), (-1,))
    x_max = _ivy.reshape(_ivy.reduce_max(pixel_x_coords, keepdims=True), (-1,))
    x_range = x_max - x_min
    y_min = _ivy.reshape(_ivy.reduce_min(pixel_y_coords, keepdims=True), (-1,))
    y_max = _ivy.reshape(_ivy.reduce_max(pixel_y_coords, keepdims=True), (-1,))
    y_range = y_max - y_min

    # 2
    bbox = _ivy.concatenate((x_range, y_range), 0)
    img_bbox_list = [int(item) for item in _ivy.to_list(_ivy.concatenate((y_range + 1, x_range + 1), 0))]

    # BS x 2
    v0 = pixel_xy_coords[..., 0, :]
    v1 = pixel_xy_coords[..., 1, :]
    v2 = pixel_xy_coords[..., 2, :]
    tri_centres = (v0 + v1 + v2) / 3

    # BS x 1
    v0x = v0[..., 0:1]
    v0y = v0[..., 1:2]
    v1x = v1[..., 0:1]
    v1y = v1[..., 1:2]
    v2x = v2[..., 0:1]
    v2y = v2[..., 1:2]

    # BS x BBX x BBY x 2
    uniform_sample_coords = _ivy_svg.create_uniform_pixel_coords_image(img_bbox_list, batch_shape)[..., 0:2]
    P = _ivy.round(uniform_sample_coords + tri_centres - bbox / 2)

    # BS x BBX x BBY x 1
    Px = P[..., 0:1]
    Py = P[..., 1:2]
    v0v1_edge_func = ((Px - v0x) * (v1y - v0y) - (Py - v0y) * (v1x - v0x)) >= 0
    v1v2_edge_func = ((Px - v1x) * (v2y - v1y) - (Py - v1y) * (v2x - v1x)) >= 0
    v2v0_edge_func = ((Px - v2x) * (v0y - v2y) - (Py - v2y) * (v0x - v2x)) >= 0
    edge_func = _ivy.logical_and(_ivy.logical_and(v0v1_edge_func, v1v2_edge_func), v2v0_edge_func)

    batch_indices_list = list()
    for i, batch_dim in enumerate(batch_shape):
        # get batch shape
        batch_dims_before = batch_shape[:i]
        num_batch_dims_before = len(batch_dims_before)
        batch_dims_after = batch_shape[i + 1:]
        num_batch_dims_after = len(batch_dims_after)

        # [batch_dim]
        batch_indices = _ivy.arange(batch_dim, dtype_str='int32', dev_str=dev_str)

        # [1]*num_batch_dims_before x batch_dim x [1]*num_batch_dims_after x 1 x 1
        reshaped_batch_indices = _ivy.reshape(batch_indices, [1] * num_batch_dims_before + [batch_dim] +
                                              [1] * num_batch_dims_after + [1, 1])

        # BS x N x 1
        tiled_batch_indices = _ivy.tile(reshaped_batch_indices, batch_dims_before + [1] + batch_dims_after +
                                        [input_image_dims_prod * 9, 1])
        batch_indices_list.append(tiled_batch_indices)

    # BS x N x (num_batch_dims + 2)
    all_indices = _ivy.concatenate(
        batch_indices_list + [_ivy.cast(_ivy.flip(_ivy.reshape(P, batch_shape + [-1, 2]), -1),
                                        'int32')], -1)

    # offset uniform images
    return _ivy.cast(_ivy.flip(_ivy.scatter_nd(_ivy.reshape(all_indices, [-1, num_batch_dims + 2]),
                                               _ivy.reshape(_ivy.cast(edge_func, 'int32'), (-1, 1)),
                                               batch_shape + image_dims + [1],
                                               reduction='replace' if _ivy.backend == 'mxnd' else 'sum'), -3), 'bool')


# noinspection PyUnresolvedReferences
def weighted_image_smooth(mean, weights, kernel_dim):
    """
    Smooth an image using weight values from a weight image of the same size.

    :param mean: Image to smooth *[batch_shape,h,w,d]*
    :type mean: array
    :param weights: Variance image, with the variance values of each pixel in the image *[batch_shape,h,w,d]*
    :type weights: array
    :param kernel_dim: The dimension of the kernel
    :type kernel_dim: int
    :return: Image smoothed based on variance image and smoothing kernel.
    """

    # shapes as list
    kernel_shape = [kernel_dim, kernel_dim]
    dim = mean.shape[-1]

    # KW x KW x D
    kernel = _ivy.ones(kernel_shape + [dim])

    # D
    kernel_sum = _ivy.reduce_sum(kernel, [0, 1])[0]

    # BS x H x W x D
    mean_x_weights = mean * weights
    mean_x_weights_sum = _ivy.abs(_ivy.depthwise_conv2d(mean_x_weights, kernel, 1, "VALID"))
    sum_of_weights = _ivy.depthwise_conv2d(weights, kernel, 1, "VALID")
    new_mean = mean_x_weights_sum / (sum_of_weights + MIN_DENOMINATOR)

    new_weights = sum_of_weights / (kernel_sum + MIN_DENOMINATOR)

    # BS x H x W x D,  # BS x H x W x D
    return new_mean, new_weights


def smooth_image_fom_var_image(mean, var, kernel_dim, kernel_scale, dev_str=None):
    """
    Smooth an image using variance values from a variance image of the same size, and a spatial smoothing kernel.

    :param mean: Image to smooth *[batch_shape,h,w,d]*
    :type mean: array
    :param var: Variance image, with the variance values of each pixel in the image *[batch_shape,h,w,d]*
    :type var: array
    :param kernel_dim: The dimension of the kernel
    :type kernel_dim: int
    :param kernel_scale: The scale of the kernel along the channel dimension *[d]*
    :type kernel_scale: array
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev_str: str, optional
    :return: Image smoothed based on variance image and smoothing kernel.
    """

    if dev_str is None:
        dev_str = _ivy.dev_str(mean)

    # shapes as list
    kernel_shape = [kernel_dim, kernel_dim]
    kernel_size = kernel_dim ** 2
    dims = mean.shape[-1]

    # KH x KW x 2
    uniform_pixel_coords = _ivy_svg.create_uniform_pixel_coords_image(kernel_shape, dev_str=dev_str)[..., 0:2]

    # 2
    kernel_central_pixel_coord = _ivy.array([float(_math.floor(kernel_shape[0] / 2)),
                                             float(_math.floor(kernel_shape[1] / 2))], dev_str=dev_str)

    # KH x KW x 2
    kernel_xy_dists = kernel_central_pixel_coord - uniform_pixel_coords
    kernel_xy_dists_sqrd = kernel_xy_dists ** 2

    # KW x KW x D x D
    unit_kernel = _ivy.tile(_ivy.reduce_sum(kernel_xy_dists_sqrd, -1, keepdims=True) ** 0.5, (1, 1, dims))
    kernel = 1 + unit_kernel * kernel_scale
    recip_kernel = 1 / (kernel + MIN_DENOMINATOR)

    # D
    kernel_sum = _ivy.reduce_sum(kernel, [0, 1])[0]
    recip_kernel_sum = _ivy.reduce_sum(recip_kernel, [0, 1])

    # BS x H x W x D
    recip_var = 1 / (var + MIN_DENOMINATOR)
    recip_var_scaled = recip_var + 1

    recip_new_var_scaled = _ivy.depthwise_conv2d(recip_var_scaled, recip_kernel, 1, "VALID")
    # This 0.99 prevents float32 rounding errors leading to -ve variances, the true equation would use 1.0
    recip_new_var = recip_new_var_scaled - recip_kernel_sum * 0.99
    new_var = 1 / (recip_new_var + MIN_DENOMINATOR)

    mean_x_recip_var = mean * recip_var
    mean_x_recip_var_sum = _ivy.abs(_ivy.depthwise_conv2d(mean_x_recip_var, recip_kernel, 1, "VALID"))
    new_mean = new_var * mean_x_recip_var_sum

    new_var = new_var * kernel_size ** 2 / (kernel_sum + MIN_DENOMINATOR)
    # prevent overconfidence from false meas independence assumption

    # BS x H x W x D,        # BS x H x W x D
    return new_mean, new_var


def pad_omni_image(image, pad_size, image_dims=None):
    """
    Pad an omni-directional image with the correct image wrapping at the edges.

    :param image: Image to perform the padding on *[batch_shape,h,w,d]*
    :type image: array
    :param pad_size: Number of pixels to pad.
    :type pad_size: int
    :param image_dims: Image dimensions. Inferred from Inputs if None.
    :type image_dims: sequence of ints, optional
    :return: New padded omni-directional image *[batch_shape,h+ps,w+ps,d]*
    """

    if image_dims is None:
        image_dims = image.shape[-3:-1]

    # BS x PS x W/2 x D
    top_left = image[..., 0:pad_size, int(image_dims[1] / 2):, :]
    top_right = image[..., 0:pad_size, 0:int(image_dims[1] / 2), :]

    # BS x PS x W x D
    top_border = _ivy.flip(_ivy.concatenate((top_left, top_right), -2), -3)

    # BS x PS x W/2 x D
    bottom_left = image[..., -pad_size:, int(image_dims[1] / 2):, :]
    bottom_right = image[..., -pad_size:, 0:int(image_dims[1] / 2), :]

    # BS x PS x W x D
    bottom_border = _ivy.flip(_ivy.concatenate((bottom_left, bottom_right), -2), -3)

    # BS x H+2PS x W x D
    image_expanded = _ivy.concatenate((top_border, image, bottom_border), -3)

    # BS x H+2PS x PS x D
    left_border = image_expanded[..., -pad_size:, :]
    right_border = image_expanded[..., 0:pad_size, :]

    # BS x H+2PS x W+2PS x D
    return _ivy.concatenate((left_border, image_expanded, right_border), -2)


def create_trimesh_indices_for_image(batch_shape, image_dims, dev_str='cpu:0'):
    """
    Create triangle mesh for image with given image dimensions

    :param batch_shape: Shape of batch.
    :type batch_shape: sequence of ints
    :param image_dims: Image dimensions.
    :type image_dims: sequence of ints
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    :type dev_str: str, optional
    :return: Triangle mesh indices for image *[batch_shape,h*w*some_other_stuff,3]*
    """

    # shapes as lists
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # other shape specs
    num_batch_dims = len(batch_shape)
    tri_dim = 2 * (image_dims[0] - 1) * (image_dims[1] - 1)
    flat_shape = [1] * num_batch_dims + [tri_dim] + [3]
    tile_shape = batch_shape + [1] * 2

    # 1 x W-1
    t00_ = _ivy.reshape(_ivy.arange(image_dims[1] - 1, dtype_str='float32', dev_str=dev_str), (1, -1))

    # H-1 x 1
    k_ = _ivy.reshape(_ivy.arange(image_dims[0] - 1, dtype_str='float32', dev_str=dev_str), (-1, 1)) * image_dims[1]

    # H-1 x W-1
    t00_ = _ivy.matmul(_ivy.ones((image_dims[0] - 1, 1), dev_str=dev_str), t00_)
    k_ = _ivy.matmul(k_, _ivy.ones((1, image_dims[1] - 1), dev_str=dev_str))

    # (H-1xW-1) x 1
    t00 = _ivy.expand_dims(t00_ + k_, -1)
    t01 = t00 + 1
    t02 = t00 + image_dims[1]
    t10 = t00 + image_dims[1] + 1
    t11 = t01
    t12 = t02

    # (H-1xW-1) x 3
    t0 = _ivy.concatenate((t00, t01, t02), -1)
    t1 = _ivy.concatenate((t10, t11, t12), -1)

    # BS x 2x(H-1xW-1) x 3
    return _ivy.tile(_ivy.reshape(_ivy.concatenate((t0, t1), 0),
                                  flat_shape), tile_shape)


def coord_image_to_trimesh(coord_img, validity_mask=None, batch_shape=None, image_dims=None, dev_str=None):
    """
    Create trimesh, with vertices and triangle indices, from co-ordinate image.

    :param coord_img: Image of co-ordinates *[batch_shape,h,w,3]*
    :type coord_img: array
    :param validity_mask: Boolean mask of where the coord image contains valid values *[batch_shape,h,w,1]*
    :type validity_mask: array, optional
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev_str: str, optional
    :return: Vertices *[batch_shape,(hxw),3]* amd Trimesh indices *[batch_shape,n,3]*
    """

    if dev_str is None:
        dev_str = _ivy.dev_str(coord_img)

    if batch_shape is None:
        batch_shape = _ivy.shape(coord_img)[:-3]

    if image_dims is None:
        image_dims = _ivy.shape(coord_img)[-3:-1]

    # shapes as lists
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x (HxW) x 3
    vertices = _ivy.reshape(coord_img, batch_shape + [image_dims[0] * image_dims[1], 3])

    if validity_mask is not None:

        # BS x H-1 x W-1 x 1
        t00_validity = validity_mask[..., 0:image_dims[0] - 1, 0:image_dims[1] - 1, :]
        t01_validity = validity_mask[..., 0:image_dims[0] - 1, 1:image_dims[1], :]
        t02_validity = validity_mask[..., 1:image_dims[0], 0:image_dims[1] - 1, :]
        t10_validity = validity_mask[..., 1:image_dims[0], 1:image_dims[1], :]
        t11_validity = t01_validity
        t12_validity = t02_validity

        # BS x H-1 x W-1 x 1
        t0_validity = _ivy.logical_and(t00_validity, _ivy.logical_and(t01_validity, t02_validity))
        t1_validity = _ivy.logical_and(t10_validity, _ivy.logical_and(t11_validity, t12_validity))

        # BS x (H-1xW-1)
        t0_validity_flat = _ivy.reshape(t0_validity, batch_shape + [-1])
        t1_validity_flat = _ivy.reshape(t1_validity, batch_shape + [-1])

        # BS x 2x(H-1xW-1)
        trimesh_index_validity = _ivy.concatenate((t0_validity_flat, t1_validity_flat), -1)

        # BS x N
        trimesh_valid_indices = _ivy.indices_where(trimesh_index_validity)

        # BS x 2x(H-1xW-1) x 3
        all_trimesh_indices = create_trimesh_indices_for_image(batch_shape, image_dims, dev_str)

        # BS x N x 3
        trimesh_indices = _ivy.gather_nd(all_trimesh_indices, trimesh_valid_indices)

    else:

        # BS x N=2x(H-1xW-1) x 3
        trimesh_indices = create_trimesh_indices_for_image(batch_shape, image_dims)

    # BS x (HxW) x 3,    BS x N x 3
    return vertices, trimesh_indices
