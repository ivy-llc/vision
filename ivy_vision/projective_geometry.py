"""
Collection of General Projective-Geometry Functions
"""

# global
import ivy as _ivy


def transform(coords, trans, batch_shape=None, image_dims=None):
    """
    Transform image of :math:`n`-dimensional co-ordinates :math:`\mathbf{x}\in\mathbb{R}^{h×w×n}` by
    transformation matrix :math:`\mathbf{f}\in\mathbb{R}^{m×n}`, to produce image of transformed co-ordinates
    :math:`\mathbf{x}_{trans}\in\mathbb{R}^{h×w×m}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Matrix_multiplication>`_

    :param coords: Co-ordinate image *[batch_shape,height,width,n]*
    :type coords: array
    :param trans: Transformation matrix *[batch_shape,m,n]*
    :type trans: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs if None.
    :type image_dims: sequence of ints
    :return: Transformed co-ordinate image *[batch_shape,height,width,m]*
    """

    if batch_shape is None:
        batch_shape = coords.shape[:-3]

    if image_dims is None:
        image_dims = coords.shape[-3:-1]

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)

    # transpose idxs
    num_batch_dims = len(batch_shape)
    transpose_idxs = list(range(num_batch_dims)) + [num_batch_dims + 1, num_batch_dims]

    # BS x (HxW) x N
    coords_flattened = _ivy.reshape(coords, batch_shape + [image_dims[0] * image_dims[1], -1])

    # BS x N x (HxW)
    coords_reshaped = _ivy.transpose(coords_flattened, transpose_idxs)

    # BS x M x (HxW)
    transformed_coords_vector = _ivy.matmul(trans, coords_reshaped)

    # BS x (HxW) x M
    transformed_coords_vector_transposed = _ivy.transpose(transformed_coords_vector, transpose_idxs)

    # BS x H x W x M
    return _ivy.reshape(transformed_coords_vector_transposed, batch_shape + image_dims + [-1])


def projection_matrix_pseudo_inverse(proj_mat, batch_shape=None):
    """
    Given projection matrix :math:`\mathbf{P}\in\mathbb{R}^{3×4}`, compute it's pseudo-inverse
    :math:`\mathbf{P}^+\in\mathbb{R}^{4×3}`.\n
    `[reference] <localhost:63342/ivy/docs/source/references/mvg_textbook.pdf?_ijt=25ihpil89dmfo4da975v402ogc#page=179>`_
    bottom of page 161, section 6.2.2
    
    :param proj_mat: Projection matrix *[batch_shape,3,4]*
    :type proj_mat: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :return: Projection matrix pseudo-inverse *[batch_shape,4,3]*
    """

    if batch_shape is None:
        batch_shape = proj_mat.shape[:-2]

    # shapes as list
    batch_shape = list(batch_shape)

    # transpose idxs
    num_batch_dims = len(batch_shape)
    transpose_idxs = list(range(num_batch_dims)) + [num_batch_dims + 1, num_batch_dims]

    # BS x 4 x 3
    matrix_transposed = _ivy.transpose(proj_mat, transpose_idxs)

    # BS x 4 x 3
    return _ivy.matmul(matrix_transposed, _ivy.inv(_ivy.matmul(proj_mat, matrix_transposed)))


def projection_matrix_inverse(proj_mat):
    """
    Given projection matrix :math:`\mathbf{P}\in\mathbb{R}^{3×4}`, compute it's inverse
    :math:`\mathbf{P}^{-1}\in\mathbb{R}^{3×4}`.\n
    `[reference] <https://github.com/pranjals16/cs676/blob/master/Hartley%2C%20Zisserman%20-%20Multiple%20View%20Geometry%20in%20Computer%20Vision.pdf#page=174>`_
    middle of page 156, section 6.1, eq 6.6

    :param proj_mat: Projection matrix *[batch_shape,3,4]*
    :type proj_mat: array
    :return: Projection matrix inverse *[batch_shape,3,4]*
    """

    # BS x 3 x 3
    rotation_matrix = proj_mat[..., 0:3]

    # BS x 3 x 3
    rotation_matrix_inverses = _ivy.inv(rotation_matrix)

    # BS x 3 x 1
    translations = proj_mat[..., 3:4]

    # BS x 3 x 1
    translation_inverses = -_ivy.matmul(rotation_matrix_inverses, translations)

    # BS x 3 x 4
    return _ivy.concatenate((rotation_matrix_inverses, translation_inverses), -1)


def solve_homogeneous_dlt(A):
    """
    Given :math:`\mathbf{A}\in\mathbb{R}^{d×d}`, solve the system of :math:`d` equations in :math:`d` unknowns
    :math:`\mathbf{Ax} = \mathbf{0}` using the homogeneous DLT method, to return :math:`\mathbf{x}\in\mathbb{R}^d`.\n
    `[reference] <https://github.com/pranjals16/cs676/blob/master/Hartley%2C%20Zisserman%20-%20Multiple%20View%20Geometry%20in%20Computer%20Vision.pdf#page=106>`_
    bottom of page 88, section 4.1

    :param A: Matrix representing system of equations to solve *[batch_shape,d,d]*
    :type A: array
    :return: Solution to the system of equations *[batch_shape,d]*
    """

    # BS x D x D,    BS x D,    BS x D x D
    U, D, VT = _ivy.svd(A)

    # BS x D
    return VT[..., -1, :]
