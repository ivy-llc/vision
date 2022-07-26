"""Collection of Singed Distance Functions"""
# ToDo: extend to include cylinders and cones
# https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm

# global
import ivy as _ivy


def sphere_signed_distances(sphere_positions, sphere_radii, query_positions):
    """Return the signed distances of a set of query points from the sphere
    surfaces.\n `[reference] <https://www.iquilezles.org/www/articles/distfunctions
    /distfunctions.htm>`_

    Parameters
    ----------
    sphere_positions
        Positions of the spheres *[batch_shape,num_spheres,3]*
    sphere_radii
        Radii of the spheres *[batch_shape,num_spheres,1]*
    query_positions
        Points for which to query the signed distances *[batch_shape,num_points,3]*

    Returns
    -------
    ret
        The distances of the query points from the closest sphere surface
        *[batch_shape,num_points,1]*

    """

    # BS x NS x 1 x 3
    sphere_positions = _ivy.expand_dims(sphere_positions, -2)

    # BS x 1 x NP x 3
    query_positions = _ivy.expand_dims(query_positions, -3)

    # BS x NS x NP x 1
    distances_to_centre = _ivy.sum((query_positions - sphere_positions) ** 2, -1, keepdims=True)**0.5

    # BS x NS x NP x 1
    all_sdfs = distances_to_centre - _ivy.expand_dims(sphere_radii, -2)

    # BS x NP x 1
    return _ivy.min(all_sdfs, -3)


def cuboid_signed_distances(cuboid_ext_mats, cuboid_dims, query_positions, batch_shape=None):
    """Return the signed distances of a set of query points from the cuboid
    surfaces.\n `[reference] <https://www.iquilezles.org/www/articles/distfunctions
    /distfunctions.htm>`_

    Parameters
    ----------
    cuboid_ext_mats
        Extrinsic matrices of the cuboids *[batch_shape,num_cuboids,3,4]*
    cuboid_dims
        Dimensions of the cuboids, in the order x, y, z *[batch_shape,num_cuboids,3]*
    query_positions
        Points for which to query the signed distances *[batch_shape,num_points,3]*
    batch_shape
        Shape of batch. Assumed no batches if None. (Default value = None)

    Returns
    -------
    ret
        The distances of the query points from the closest cuboid surface
        *[batch_shape,num_points,1]*

    """

    if batch_shape is None:
        batch_shape = cuboid_ext_mats.shape[:-3]

    # shapes as list
    batch_shape = list(batch_shape)
    num_batch_dims = len(batch_shape)
    batch_dims_for_trans = list(range(num_batch_dims))
    num_cuboids = cuboid_ext_mats.shape[-3]
    num_points = query_positions.shape[-2]

    # BS x 3 x NP
    query_positions_trans = _ivy.transpose(
        query_positions, batch_dims_for_trans + [num_batch_dims+1, num_batch_dims])

    # BS x 1 x NP
    ones = _ivy.ones_like(query_positions_trans[..., 0:1, :])

    # BS x 4 x NP
    query_positions_trans_homo = _ivy.concat([query_positions_trans, ones], -2)

    # BS x NCx3 x 4
    cuboid_ext_mats_flat = _ivy.reshape(cuboid_ext_mats, batch_shape + [-1, 4])

    # BS x NCx3 x NP
    rel_query_positions_trans_flat = _ivy.matmul(cuboid_ext_mats_flat, query_positions_trans_homo)

    # BS x NC x 3 x NP
    rel_query_positions_trans = _ivy.reshape(rel_query_positions_trans_flat, batch_shape + [num_cuboids, 3, num_points])

    # BS x NC x NP x 3
    rel_query_positions = _ivy.transpose(rel_query_positions_trans,
                                         batch_dims_for_trans + [num_batch_dims, num_batch_dims+2, num_batch_dims+1])
    q = _ivy.abs(rel_query_positions) - _ivy.expand_dims(cuboid_dims/2, -2)
    q_max_clipped = _ivy.max(q, 1e-12)

    # BS x NC x NP x 1
    q_min_clipped = _ivy.min(_ivy.max(q, -1, keepdims=True), 0.)
    q_max_clipped_len = _ivy.sum(q_max_clipped**2, -1, keepdims=True)**0.5
    sdfs = q_max_clipped_len + q_min_clipped

    # BS x NP x 1
    return _ivy.min(sdfs, -3)
