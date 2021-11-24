"""
Collection of Mesh Functions
"""

# global
import ivy as _ivy
from functools import reduce as _reduce
from operator import mul as _mul

# local
from ivy_vision import single_view_geometry as _ivy_svg

MIN_DENOMINATOR = 1e-12


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
                                               reduction='replace' if _ivy.backend == 'mxnet' else 'sum'), -3), 'bool')


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
