"""Collection of Mesh Functions"""

# global
import ivy 
from functools import reduce 
from operator import mul 

# local
from ivy_vision import single_view_geometry as ivy_svg

MIN_DENOMINATOR = 1e-12


def rasterize_triangles(pixel_coords_triangles, image_dims, batch_shape=None, dev_str=None):
    """Rasterize image-projected triangles based on:
    https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical
    -implementation/rasterization-stage and:
    https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical
    -implementation/rasterization-practical-implementation

    Parameters
    ----------
    pixel_coords_triangles
        Projected image-space triangles to be rasterized
        *[batch_shape,input_size,3,3]*
    image_dims
        Image dimensions.
    batch_shape
        Shape of batch. Inferred from Inputs if None. (Default value = None)
    dev_str
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        Same as x if None. (Default value = None)

    Returns
    -------
    ret
        Rasterized triangles

    """

    if batch_shape is None:
        batch_shape = []

    if dev_str is None:
        dev_str = ivy.dev(pixel_coords_triangles)

    # shapes as list
    batch_shape = list(batch_shape)
    num_batch_dims = len(batch_shape)
    image_dims = list(image_dims)
    input_image_dims = pixel_coords_triangles.shape[num_batch_dims:-2]
    input_image_dims_prod = reduce(mul, input_image_dims, 1)

    # BS x 3 x 2
    pixel_xy_coords = pixel_coords_triangles[..., 0:2]

    # BS x 3 x 1
    pixel_x_coords = pixel_coords_triangles[..., 0:1]
    pixel_y_coords = pixel_coords_triangles[..., 1:2]

    # 1
    x_min = ivy.reshape(ivy.min(pixel_x_coords, keepdims=True), (-1,))
    x_max = ivy.reshape(ivy.max(pixel_x_coords, keepdims=True), (-1,))
    x_range = x_max - x_min
    y_min = ivy.reshape(ivy.min(pixel_y_coords, keepdims=True), (-1,))
    y_max = ivy.reshape(ivy.max(pixel_y_coords, keepdims=True), (-1,))
    y_range = y_max - y_min

    # 2
    bbox = ivy.concat((x_range, y_range), axis=0)
    img_bbox_list = [int(item) for item in ivy.to_list(ivy.concat((y_range + 1, x_range + 1), axis=0))]

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
    uniform_sample_coords = ivy_svg.create_uniform_pixel_coords_image(img_bbox_list, batch_shape)[..., 0:2]
    P = ivy.round(uniform_sample_coords + tri_centres - bbox / 2)

    # BS x BBX x BBY x 1
    Px = P[..., 0:1]
    Py = P[..., 1:2]
    v0v1_edge_func = ((Px - v0x) * (v1y - v0y) - (Py - v0y) * (v1x - v0x)) >= 0
    v1v2_edge_func = ((Px - v1x) * (v2y - v1y) - (Py - v1y) * (v2x - v1x)) >= 0
    v2v0_edge_func = ((Px - v2x) * (v0y - v2y) - (Py - v2y) * (v0x - v2x)) >= 0
    edge_func = ivy.logical_and(ivy.logical_and(v0v1_edge_func, v1v2_edge_func), v2v0_edge_func)

    batch_indices_list = list()
    for i, batch_dim in enumerate(batch_shape):
        # get batch shape
        batch_dims_before = batch_shape[:i]
        num_batch_dims_before = len(batch_dims_before)
        batch_dims_after = batch_shape[i + 1:]
        num_batch_dims_after = len(batch_dims_after)

        # [batch_dim]
        batch_indices = ivy.arange(batch_dim, dtype='int32', device=dev_str)

        # [1]*num_batch_dims_before x batch_dim x [1]*num_batch_dims_after x 1 x 1
        reshaped_batch_indices = ivy.reshape(batch_indices, [1] * num_batch_dims_before + [batch_dim] +
                                              [1] * num_batch_dims_after + [1, 1])

        # BS x N x 1
        tiled_batch_indices = ivy.tile(reshaped_batch_indices, batch_dims_before + [1] + batch_dims_after +
                                        [input_image_dims_prod * 9, 1])
        batch_indices_list.append(tiled_batch_indices)

    # BS x N x (num_batch_dims + 2)
    all_indices = ivy.concat(
        batch_indices_list + [ivy.astype(ivy.flip(ivy.reshape(P, batch_shape + [-1, 2]), axis=-1),
                                        'int32')], axis=-1)

    # offset uniform images
    return ivy.astype(ivy.flip(ivy.scatter_nd(ivy.reshape(all_indices, [-1, num_batch_dims + 2]),
                                               ivy.reshape(ivy.astype(edge_func, 'int32'), (-1, 1)),
                                               shape=batch_shape + image_dims + [1],
                                               reduction='replace' if ivy.backend == 'mxnet' else 'sum'), axis=-3), 'bool')


def create_trimesh_indices_for_image(batch_shape, image_dims, dev_str=None):
    """Create triangle mesh for image with given image dimensions

    Parameters
    ----------
    batch_shape
        Shape of batch.
    image_dims
        Image dimensions.
    dev_str
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        (Default value = None)

    Returns
    -------
    ret
        Triangle mesh indices for image *[batch_shape,h*w*some_other_stuff,3]*

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
    t00_ = ivy.reshape(ivy.arange(image_dims[1] - 1, dtype='float32', device=dev_str), (1, -1))

    # H-1 x 1
    k_ = ivy.reshape(ivy.arange(image_dims[0] - 1, dtype='float32', device=dev_str), (-1, 1)) * image_dims[1]

    # H-1 x W-1
    t00_ = ivy.matmul(ivy.ones((image_dims[0] - 1, 1), device=dev_str), t00_)
    k_ = ivy.matmul(k_, ivy.ones((1, image_dims[1] - 1), device=dev_str))

    # (H-1xW-1) x 1
    t00 = ivy.expand_dims(t00_ + k_, axis=-1)
    t01 = t00 + 1
    t02 = t00 + image_dims[1]
    t10 = t00 + image_dims[1] + 1
    t11 = t01
    t12 = t02

    # (H-1xW-1) x 3
    t0 = ivy.concat((t00, t01, t02), axis=-1)
    t1 = ivy.concat((t10, t11, t12), axis=-1)

    # BS x 2x(H-1xW-1) x 3
    return ivy.tile(ivy.reshape(ivy.concat((t0, t1), axis=0),
                                  flat_shape), tile_shape)


def coord_image_to_trimesh(coord_img, validity_mask=None, batch_shape=None, image_dims=None, dev_str=None):
    """Create trimesh, with vertices and triangle indices, from co-ordinate image.

    Parameters
    ----------
    coord_img
        Image of co-ordinates *[batch_shape,h,w,3]*
    validity_mask
        Boolean mask of where the coord image contains valid values
        *[batch_shape,h,w,1]* (Default value = None)
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
        Vertices *[batch_shape,(hxw),3]* amd Trimesh indices *[batch_shape,n,3]*

    """

    if dev_str is None:
        dev_str = ivy.dev(coord_img)

    if batch_shape is None:
        batch_shape = ivy.shape(coord_img)[:-3]

    if image_dims is None:
        image_dims = ivy.shape(coord_img)[-3:-1]

    # shapes as lists
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # BS x (HxW) x 3
    vertices = ivy.reshape(coord_img, batch_shape + [image_dims[0] * image_dims[1], 3])

    if validity_mask is not None:

        # BS x H-1 x W-1 x 1
        t00_validity = validity_mask[..., 0:image_dims[0] - 1, 0:image_dims[1] - 1, :]
        t01_validity = validity_mask[..., 0:image_dims[0] - 1, 1:image_dims[1], :]
        t02_validity = validity_mask[..., 1:image_dims[0], 0:image_dims[1] - 1, :]
        t10_validity = validity_mask[..., 1:image_dims[0], 1:image_dims[1], :]
        t11_validity = t01_validity
        t12_validity = t02_validity

        # BS x H-1 x W-1 x 1
        t0_validity = ivy.logical_and(t00_validity, ivy.logical_and(t01_validity, t02_validity))
        t1_validity = ivy.logical_and(t10_validity, ivy.logical_and(t11_validity, t12_validity))

        # BS x (H-1xW-1)
        t0_validity_flat = ivy.reshape(t0_validity, batch_shape + [-1])
        t1_validity_flat = ivy.reshape(t1_validity, batch_shape + [-1])

        # BS x 2x(H-1xW-1)
        trimesh_index_validity = ivy.concat((t0_validity_flat, t1_validity_flat), axis=-1)

        # BS x N
        trimesh_valid_indices = ivy.argwhere(trimesh_index_validity)

        # BS x 2x(H-1xW-1) x 3
        all_trimesh_indices = create_trimesh_indices_for_image(batch_shape, image_dims, dev_str)

        # BS x N x 3
        trimesh_indices = ivy.gather_nd(all_trimesh_indices, trimesh_valid_indices)

    else:

        # BS x N=2x(H-1xW-1) x 3
        trimesh_indices = create_trimesh_indices_for_image(batch_shape, image_dims)

    # BS x (HxW) x 3,    BS x N x 3
    return vertices, trimesh_indices
