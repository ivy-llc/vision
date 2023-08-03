"""Collection of Voxel-Grid Functions"""

# global
import ivy

MIN_DENOMINATOR = 1e-12


def coords_to_voxel_grid(
    coords,
    voxel_shape_spec,
    mode="DIMS",
    coord_bounds=None,
    features=None,
    batch_shape=None,
    device=None,
):
    r"""Create voxel grid :math:`\mathbf{X}_v\in\mathbb{R}^{x×y×z×(3+N+1)}` from
    homogeneous co-ordinates :math:`\mathbf{X}_w\in\mathbb{R}^{num\_coords×4}`. Each
    voxel contains 3+N+1 values: the mean world co-ordinate inside the voxel for the
    projected pixels, N coordinte features (optional), and also the number of
    projected pixels inside the voxel. Grid resolutions and dimensions are also
    returned separately for each entry in the batch. Note that the final batched
    voxel grid returned uses the maximum grid dimensions across the batch, this means
    some returned grids may contain redundant space, with all but the single largest
    batched grid occupying a subset of the grid space, originating from the corner of
    minimum :math:`x,y,z` values.\n `[reference]
    <https://en.wikipedia.org/wiki/Voxel>`_

    Parameters
    ----------
    coords
        Homogeneous co-ordinates *[batch_shape,c,4]*
    voxel_shape_spec
        Either the number of voxels in x,y,z directions, or the resolutions (metres) in
        x,y,z
        directions, depending on mode. Batched or unbatched. *[batch_shape,3]* or *[3]*
    mode
        Shape specification mode, either "DIMS" or "RES" (Default value = 'DIMS')
    coord_bounds
        Co-ordinate x, y, z boundaries *[batch_shape,6]* or *[6]* (Default value = None)
    features
        Co-ordinate features *[batch_shape,c,4]*.
        E.g. RGB values, low-dimensional features, etc.
        Features mapping to the same voxel are averaged. (Default value = None)
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Voxel grid *[batch_shape,x_max,v_max,z_max,3+feature_size+1]*, dimensions
        *[batch_shape,3]*,
        resolutions *[batch_shape,3]*, voxel_grid_lower_corners *[batch_shape,3]*

    """
    if batch_shape is None:
        batch_shape = coords.shape[:-2]

    if device is None:
        device = ivy.dev(coords)

    # shapes as list
    batch_shape = list(batch_shape)
    num_batch_dims = len(batch_shape)
    num_coords_per_batch = coords.shape[-2]

    # voxel shape spec as array
    if len(voxel_shape_spec) == 3:
        # BS x 1 x 3
        voxel_shape_spec = ivy.expand_dims(
            ivy.tile(
                ivy.reshape(ivy.array(voxel_shape_spec), [1] * num_batch_dims + [3]),
                batch_shape + [1],
            ),
            axis=-2,
        )

    # coord bounds spec as array
    if coord_bounds is not None:
        if len(coord_bounds) == 6:
            # BS x 6
            coord_bounds = ivy.tile(
                ivy.reshape(
                    ivy.array(coord_bounds, dtype="float32"), [1] * num_batch_dims + [6]
                ),
                batch_shape + [1],
            )

        # BS x 1 x 6
        coord_bounds = ivy.expand_dims(coord_bounds, axis=-2)

    # BS x N x 3
    coords = coords[..., 0:3]

    if coord_bounds is not None:
        # BS x 1 x 1
        x_min = coord_bounds[..., 0:1]
        y_min = coord_bounds[..., 1:2]
        z_min = coord_bounds[..., 2:3]
        x_max = coord_bounds[..., 3:4]
        y_max = coord_bounds[..., 4:5]
        z_max = coord_bounds[..., 5:6]

        # BS x N x 1
        x_coords = coords[..., 0:1]
        y_coords = coords[..., 1:2]
        z_coords = coords[..., 2:3]

        x_validity_mask = ivy.logical_and(x_coords > x_min, x_coords < x_max)
        y_validity_mask = ivy.logical_and(y_coords > y_min, y_coords < y_max)
        z_validity_mask = ivy.logical_and(z_coords > z_min, z_coords < z_max)

        # BS x N
        full_validity_mask = ivy.logical_and(
            ivy.logical_and(x_validity_mask, y_validity_mask), z_validity_mask
        )[..., 0]

        # BS x 1 x 3
        bb_mins = coord_bounds[..., 0:3]
        bb_maxs = coord_bounds[..., 3:6]
        bb_ranges = bb_maxs - bb_mins
    else:
        # BS x N
        full_validity_mask = ivy.astype(
            ivy.ones(batch_shape + [num_coords_per_batch], device=device), "bool"
        )

        # BS x 1 x 3
        bb_mins = ivy.min(coords, axis=-2, keepdims=True)
        bb_maxs = ivy.max(coords, axis=-2, keepdims=True)
        bb_ranges = bb_maxs - bb_mins

    # get voxel dimensions
    if mode == "DIMS":
        # BS x 1 x 3
        dims = ivy.astype(voxel_shape_spec, "int32")
    elif mode == "RES":
        # BS x 1 x 3
        res = ivy.astype(voxel_shape_spec, "float32")
        dims = ivy.astype(ivy.ceil(bb_ranges / (res + MIN_DENOMINATOR)), "int32")
    else:
        raise Exception('Invalid mode selection. Must be either "DIMS" or "RES"')
    dims_m_one = ivy.astype(dims - 1, "int32")

    # BS x 1 x 3
    res = bb_ranges / (ivy.astype(dims, "float32") + MIN_DENOMINATOR)

    # BS x NC x 3
    voxel_indices = ivy.minimum(
        ivy.astype(ivy.floor((coords - bb_mins) / (res + MIN_DENOMINATOR)), "int32"),
        dims_m_one,
    )

    # BS x NC x 3
    voxel_values = coords

    feature_size = 0
    if features is not None:
        feature_size = features.shape[-1]
        voxel_values = ivy.concat([voxel_values, features], axis=-1)

    # TNVC x len(BS)+1
    valid_coord_indices = ivy.astype(ivy.argwhere(full_validity_mask), "int32")

    # scalar
    total_num_valid_coords = valid_coord_indices.shape[0]

    # TNVC x 3
    voxel_values_pruned_flat = ivy.gather_nd(voxel_values, valid_coord_indices)
    voxel_indices_pruned_flat = ivy.gather_nd(voxel_indices, valid_coord_indices)

    # TNVC x len(BS)+2
    if num_batch_dims == 0:
        all_indices_pruned_flat = voxel_indices_pruned_flat
    else:
        batch_indices = valid_coord_indices[..., :-1]
        all_indices_pruned_flat = ivy.concat(
            [batch_indices] + [voxel_indices_pruned_flat], axis=-1
        )

    # TNVC x 4
    voxel_values_pruned_flat = ivy.concat(
        (
            voxel_values_pruned_flat,
            ivy.ones([total_num_valid_coords, 1], device=device),
        ),
        axis=-1,
    )

    # get max dims list for scatter
    if num_batch_dims > 0:
        max_dims = ivy.max(
            ivy.reshape(dims, batch_shape + [3]), axis=list(range(num_batch_dims))
        )
    else:
        max_dims = ivy.reshape(dims, batch_shape + [3])
    batch_shape_array_list = (
        [ivy.array(batch_shape, dtype="int32", device=device)]
        if num_batch_dims != 0
        else []
    )
    total_dims_list = ivy.to_list(
        ivy.concat(
            batch_shape_array_list
            + [max_dims, ivy.array([4 + feature_size], dtype="int32", device=device)],
            axis=-1,
        )
    )

    # BS x x_max x y_max x z_max x 4
    scattered = ivy.scatter_nd(
        all_indices_pruned_flat,
        voxel_values_pruned_flat,
        shape=total_dims_list,
        reduction="sum",
    )

    # BS x x_max x y_max x z_max x 4 + feature_size, BS x 3, BS x 3, BS x 3
    return (
        ivy.concat(
            (
                scattered[..., :-1]
                / (ivy.maximum(scattered[..., -1:], 1.0) + MIN_DENOMINATOR),
                scattered[..., -1:],
            ),
            axis=-1,
        ),
        dims[..., 0, :],
        res[..., 0, :],
        bb_mins[..., 0, :],
    )
