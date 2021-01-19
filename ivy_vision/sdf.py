"""
Collection of Singed Distance Functions
"""
# ToDo: extend to include cylinders and cones
# https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm

# global
from ivy.framework_handler import get_framework as _get_framework


def sphere_signed_distances(sphere_positions, sphere_radii, query_positions, f=None):
    """
    Return the signed distances of a set of query points from the sphere surfaces.\n
    `[reference] <https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm>`_

    :param sphere_positions: Positions of the spheres *[batch_shape,num_spheres,3]*
    :type sphere_positions: array
    :param sphere_radii: Radii of the spheres *[batch_shape,num_spheres,1]*
    :type sphere_radii: array
    :param query_positions: Points for which to query the signed distances *[batch_shape,num_points,3]*
    :type query_positions: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The distances of the query points from the closest sphere surface *[batch_shape,num_points,1]*
    """

    f = _get_framework(sphere_positions, f=f)

    # BS x NS x 1 x 3
    sphere_positions = f.expand_dims(sphere_positions, -2)

    # BS x 1 x NP x 3
    query_positions = f.expand_dims(query_positions, -3)

    # BS x NS x NP x 1
    distances_to_centre = f.reduce_sum((query_positions - sphere_positions) ** 2, -1, keepdims=True)**0.5

    # BS x NS x NP x 1
    all_sdfs = distances_to_centre - f.expand_dims(sphere_radii, -2)

    # BS x NP x 1
    return f.reduce_min(all_sdfs, -3)


def cuboid_signed_distances(cuboid_ext_mats, cuboid_dims, query_positions, batch_shape=None, f=None):
    """
    Return the signed distances of a set of query points from the cuboid surfaces.\n
    `[reference] <https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm>`_

    :param cuboid_ext_mats: Extrinsic matrices of the cuboids *[batch_shape,num_cuboids,3,4]*
    :type cuboid_ext_mats: array
    :param cuboid_dims: Dimensions of the cuboids, in the order x, y, z *[batch_shape,num_cuboids,3]*
    :type cuboid_dims: array
    :param query_positions: Points for which to query the signed distances *[batch_shape,num_points,3]*
    :type query_positions: array
    :param batch_shape: Shape of batch. Assumed no batches if None.
    :type batch_shape: sequence of ints, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The distances of the query points from the closest cuboid surface *[batch_shape,num_points,1]*
    """

    f = _get_framework(cuboid_ext_mats, f=f)

    if batch_shape is None:
        batch_shape = cuboid_ext_mats.shape[:-3]

    # shapes as list
    batch_shape = list(batch_shape)
    num_batch_dims = len(batch_shape)
    batch_dims_for_trans = list(range(num_batch_dims))
    num_cuboids = cuboid_ext_mats.shape[-3]
    num_points = query_positions.shape[-2]

    # BS x 3 x NP
    query_positions_trans = f.transpose(
        query_positions, batch_dims_for_trans + [num_batch_dims+1, num_batch_dims])

    # BS x 1 x NP
    ones = f.ones_like(query_positions_trans[..., 0:1, :])

    # BS x 4 x NP
    query_positions_trans_homo = f.concatenate((query_positions_trans, ones), -2)

    # BS x NCx3 x 4
    cuboid_ext_mats_flat = f.reshape(cuboid_ext_mats, batch_shape + [-1, 4])

    # BS x NCx3 x NP
    rel_query_positions_trans_flat = f.matmul(cuboid_ext_mats_flat, query_positions_trans_homo)

    # BS x NC x 3 x NP
    rel_query_positions_trans = f.reshape(rel_query_positions_trans_flat, batch_shape + [num_cuboids, 3, num_points])

    # BS x NC x NP x 3
    rel_query_positions = f.transpose(rel_query_positions_trans,
                                      batch_dims_for_trans + [num_batch_dims, num_batch_dims+2, num_batch_dims+1])
    q = f.abs(rel_query_positions) - f.expand_dims(cuboid_dims/2, -2)
    q_max_clipped = f.maximum(q, 1e-12)

    # BS x NC x NP x 1
    q_min_clipped = f.minimum(f.reduce_max(q, -1, keepdims=True), 0.)
    q_max_clipped_len = f.reduce_sum(q_max_clipped**2, -1, keepdims=True)**0.5
    sdfs = q_max_clipped_len + q_min_clipped

    # BS x NP x 1
    return f.reduce_min(sdfs, -3)
