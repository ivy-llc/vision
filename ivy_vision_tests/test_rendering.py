# global
import ivy
import pytest
import numpy as np
import ivy_tests.helpers as helpers

# local
from ivy_vision import rendering as ivy_ren
from ivy_vision_tests.data import TestData


class RenderingTestData(TestData):

    def __init__(self):
        super().__init__()

        # Pixel Quantization #
        # -------------------#

        self.simple_uniform_pixel_coords = np.array([[[0., 0., 1.], [1., 0., 1.], [2., 0., 1.]],
                                                     [[0., 1., 1.], [1., 1., 1.], [2., 1., 1.]],
                                                     [[0., 2., 1.], [1., 2., 1.], [2., 2., 1.]]])

        pixel_coords = np.array([[[0., 0., 1.], [0., 0., 1.], [2., 0., 1.]],
                                 [[2., 1., 1.], [1., 1., 1.], [2., 1., 1.]],
                                 [[1., 2., 1.], [1., 2., 1.], [1., 2., 1.]]])

        depth_values = np.array([[[1.], [2.], [3.]],
                                 [[4.], [5.], [6.]],
                                 [[7.], [8.], [9.]]])

        intensity_values = np.array([[[0.9], [0.8], [0.7]],
                                     [[0.6], [0.5], [0.4]],
                                     [[0.3], [0.2], [0.1]]])

        self.coords_to_scatter = np.reshape(np.concatenate(
            (pixel_coords * depth_values, intensity_values), -1), (-1, 4))
        self.vars_to_scatter = np.reshape(np.array([[[2.] * 2, [4.] * 2, [1.] * 2],
                                                    [[0.5] * 2, [3.] * 2, [3.] * 2],
                                                    [[1.] * 2, [1.] * 2, [2.] * 2]]), (-1, 2))

        # duplicate averaging

        quantized_cov_values = np.array([[[1 / (1 / 2 + 1 / 4)] * 2, [10.] * 2, [1.] * 2],
                                         [[10.] * 2, [3.] * 2, [1 / (1 / 0.5 + 1 / 3)] * 2],
                                         [[10.] * 2, [1 / (1 / 1 + 1 / 1 + 1 / 2)] * 2, [10.] * 2]])

        quantized_sum_mean_x_recip_cov_values = \
            np.array([[[((1 / 2) * 1 + (1 / 4) * 2), ((1 / 2) * 0.9 + (1 / 4) * 0.8)],
                       [0., 0.],
                       [(1 / 1) * 3, (1 / 1) * 0.7]],

                      [[0., 0],
                       [(1 / 3) * 5, (1 / 3) * 0.5],
                       [((1 / 0.5) * 4 + (1 / 3) * 6), ((1 / 0.5) * 0.6 + (1 / 3) * 0.4)]],

                      [[0., 0.],
                       [((1 / 1) * 7 + (1 / 1) * 8 + (1 / 2) * 9), ((1 / 1) * 0.3 + (1 / 1) * 0.2 + (1 / 2) * 0.1)],
                       [0., 0.]]])

        self.quantized_counter = np.array([[[2.], [0.], [1.]],
                                           [[0.], [1.], [2.]],
                                           [[0.], [3.], [0.]]])

        quantized_mean_vals_from_cov = quantized_cov_values * quantized_sum_mean_x_recip_cov_values
        self.quantized_pixel_coords_from_cov = \
            np.concatenate((self.simple_uniform_pixel_coords * quantized_mean_vals_from_cov[..., 0:1],
                            quantized_mean_vals_from_cov[..., 1:]), -1)
        self.quantized_cov_values = np.where(self.quantized_counter > 0, quantized_cov_values * self.quantized_counter,
                                             quantized_cov_values)

        # duplicate minimizing

        quantized_cov_values_db = np.array([[[2.] * 2, [10.] * 2, [1.] * 2],
                                            [[10.] * 2, [3.] * 2, [0.5] * 2],
                                            [[10.] * 2, [1.] * 2, [10.] * 2]])

        quantized_mean_values_db = np.array([[[1., 0.9], [0., 0.], [3., 0.7]],
                                             [[0., 0.], [5., 0.5], [4., 0.6]],
                                             [[0., 0.], [7., 0.3], [0., 0.]]])

        self.quantized_pixel_coords_from_cov_db = \
            np.concatenate((self.simple_uniform_pixel_coords * quantized_mean_values_db[..., 0:1],
                            quantized_mean_values_db[..., 1:]), -1)
        self.quantized_cov_values_db = quantized_cov_values_db
        self.validity_mask = self.quantized_counter > 0

        # Omni Pixel Quantization #
        # ------------------------#

        self.simple_projected_omni_pixel_coords = np.reshape(np.concatenate(
            (pixel_coords[..., 0:2], depth_values, intensity_values), -1), (-1, 4))
        self.quantized_omni_pixel_coords_from_cov = np.concatenate((self.simple_uniform_pixel_coords[..., 0:2],
                                                                    quantized_mean_vals_from_cov), -1)
        self.quantized_omni_pixel_coords_from_cov_db = np.concatenate((self.simple_uniform_pixel_coords[..., 0:2],
                                                                       quantized_mean_values_db), -1)

        # Image to Mesh #
        # --------------#

        self.pre_mesh_image = np.array([[[1], [2], [3]],
                                        [[4], [5], [6]],
                                        [[7], [8], [9]]])
        self.mesh_image = np.array([[[[[1], [4], [2]], [[5], [4], [2]]], [[[2], [5], [3]], [[6], [5], [3]]]],
                                    [[[[4], [7], [5]], [[8], [7], [5]]], [[[5], [8], [6]], [[9], [8], [6]]]]])

        # Triangle Rasterization #
        # -----------------------#

        self.mesh_triangle = np.array([[0., 0., 1.],
                                       [0., 2., 1.],
                                       [2., 0., 1.]])
        self.rasterized_image = np.array([[[True], [False], [False]],
                                          [[True], [True], [False]],
                                          [[True], [True], [True]]])

        # Image Smoothing #
        # ----------------#

        self.mean_img = np.array([[[[1., 0.5], [0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.], [0., 0.]]]])

        self.var_img = np.array([[[[0.5, 1.], [1000., 1000.], [1000., 1000.]],
                                  [[1000., 1000.], [1000., 1000.], [1000., 1000.]],
                                  [[1000., 1000.], [1000., 1000.], [1000., 1000.]]]])

        self.kernel_size = 3
        self.kernel_scale = np.array([0., 0.])

        self.smoothed_img_from_weights = np.array([[[[0.9960162, 0.49603155]]]])

        self.smoothed_img_from_var = np.array([[[[0.9532882, 0.4553732]]]])
        self.smoothed_var_from_var = np.array([[[[4.289797, 8.196717]]]])

        # Omni Image Padding #
        # -------------------#

        self.omni_image = np.array([[[[0.], [1.], [2.], [3.]],
                                     [[4.], [5.], [6.], [7.]],
                                     [[8.], [9.], [10.], [11.]],
                                     [[12.], [13.], [14.], [15.]]]])

        self.padded_omni_image = np.array([[[[1.], [2.], [3.], [0.], [1.], [2.]],
                                            [[3.], [0.], [1.], [2.], [3.], [0.]],
                                            [[7.], [4.], [5.], [6.], [7.], [4.]],
                                            [[11.], [8.], [9.], [10.], [11.], [8.]],
                                            [[15.], [12.], [13.], [14.], [15.], [12.]],
                                            [[13.], [14.], [15.], [12.], [13.], [14.]]]])

        # Triangle Mesh from Image #
        # -------------------------#

        self.coord_img = np.array([[[[0.5, 1.1, 2.3], [0.7, 1.2, 1.5], [0.9, 1.9, 2.0]],
                                    [[0.6, 0.7, 0.3], [0.2, 1.3, 1.2], [0.7, 0.7, 0.4]],
                                    [[1.4, 1.7, 0.1], [1.3, 1.6, 1.4], [0.8, 1.1, 1.3]],
                                    [[1.0, 0.9, 0.1], [1.2, 1.7, 1.3], [0.6, 1.7, 1.1]]]])
        self.coord_validity_img = np.array([[[[True], [False], [True]],
                                             [[True], [True], [True]],
                                             [[True], [False], [True]],
                                             [[True], [True], [True]]]])

        self.tri_mesh_4x3_indices = np.array([[[0.,  1.,  3.],
                                               [1.,  2.,  4.],
                                               [3.,  4.,  6.],
                                               [4.,  5.,  7.],
                                               [6.,  7.,  9.],
                                               [7.,  8., 10.],
                                               [4.,  1.,  3.],
                                               [5.,  2.,  4.],
                                               [7.,  4.,  6.],
                                               [8.,  5.,  7.],
                                               [10.,  7.,  9.],
                                               [11.,  8., 10.]]])

        self.tri_mesh_4x3_valid_indices = np.array([[3.,  4.,  6.],
                                                    [5.,  2.,  4.],
                                                    [11.,  8., 10.]])

        self.tri_mesh_4x3_vertices = np.reshape(self.coord_img, [1, -1, 3])


td = RenderingTestData()


def test_quantize_pixel_coords(dev_str, call):
    assert np.allclose(call(ivy_ren.quantize_pixel_coords,
                            np.reshape(td.pixel_coords, (1, 2, -1, 3)),
                            np.zeros_like(td.pixel_coords[..., -1:]),
                            td.image_dims, batch_shape=[td.batch_size, 2])[0][..., 0:3],
                       td.pixel_coords, atol=1e-3)


def test_quantize_pixel_coordinates_with_var(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet does not support sum for scatter nd, only non-deterministic replacement for duplicates
        pytest.skip()
    mean, var, counter = call(ivy_ren.quantize_pixel_coords,
                              td.coords_to_scatter,
                              np.zeros_like(td.simple_uniform_pixel_coords[..., -2:]),
                              [3, 3],
                              pixel_coords_var=td.vars_to_scatter,
                              prior_var=np.ones_like(td.simple_uniform_pixel_coords[..., -2:]) * 10,
                              var_threshold=np.array([[0., 1e4]]*2))
    assert np.allclose(counter, td.quantized_counter, atol=1e-6)
    assert np.allclose(mean, td.quantized_pixel_coords_from_cov, atol=1e-3)
    assert np.allclose(var, td.quantized_cov_values, atol=1e-3)

    assert call(ivy_ren.quantize_pixel_coords,
                np.ones((1, 9, 3)), np.ones((1, 3, 3, 1)), [3, 3],
                pixel_coords_var=np.random.uniform(size=(1, 9, 1)),
                prior_var=np.ones((1, 3, 3, 1)),
                var_threshold=np.array([[0., 0.5]]))
    assert call(ivy_ren.quantize_pixel_coords,
                np.expand_dims(td.coords_to_scatter, 0),
                np.expand_dims(np.zeros_like(td.simple_uniform_pixel_coords[..., -2:]), 0),
                [3, 3],
                pixel_coords_var=np.expand_dims(td.vars_to_scatter, 0),
                prior_var=np.expand_dims(np.ones_like(td.simple_uniform_pixel_coords[..., -2:]), 0) * 10,
                var_threshold=np.array([[0.]*2]*2))


def test_quantize_pixel_coords_with_var_db(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet does not support min for scatter nd, only non-deterministic replacement for duplicates
        pytest.skip()
    mean, var, validity_mask = call(ivy_ren.quantize_pixel_coords,
                                    td.coords_to_scatter,
                                    np.zeros_like(td.simple_uniform_pixel_coords[..., -2:]),
                                    [3, 3],
                                    with_db=True,
                                    pixel_coords_var=td.vars_to_scatter,
                                    prior_var=np.ones_like(td.simple_uniform_pixel_coords[..., -2:]) * 10,
                                    var_threshold=np.array([[0., 1e4]]*2))
    assert np.allclose(validity_mask, td.validity_mask, atol=1e-6)
    assert np.allclose(mean, td.quantized_pixel_coords_from_cov_db, atol=1e-3)
    assert np.allclose(var, td.quantized_cov_values_db, atol=1e-3)

    assert call(ivy_ren.quantize_pixel_coords,
                np.ones((1, 9, 3)), np.ones((1, 3, 3, 1)), [3, 3],
                with_db=True,
                pixel_coords_var=np.random.uniform(size=(1, 9, 1)),
                prior_var=np.ones((1, 3, 3, 1)),
                var_threshold=np.array([[0., 0.5]]))
    assert call(ivy_ren.quantize_pixel_coords,
                np.expand_dims(td.coords_to_scatter, 0),
                np.expand_dims(np.zeros_like(td.simple_uniform_pixel_coords[..., -2:]), 0),
                [3, 3],
                with_db=True,
                pixel_coords_var=np.expand_dims(td.vars_to_scatter, 0),
                prior_var=np.expand_dims(np.ones_like(td.simple_uniform_pixel_coords[..., -2:]), 0) * 10,
                var_threshold=np.array([[0., 0.]]*2))


def test_quantize_omni_pixel_coords(dev_str, call):
    assert np.allclose(call(ivy_ren.quantize_pixel_coords,
                            np.reshape(td.pixel_coords / td.pixel_coords[..., -1:], (1, 2, -1, 3)),
                            np.zeros_like(td.pixel_coords[..., -1:]),
                            td.image_dims, batch_shape=[td.batch_size, 2])[0][..., 0:3],
                       td.pixel_coords / td.pixel_coords[..., -1:], atol=1e-3)


def test_quantize_omni_pixel_coords_with_var(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet does not support sum for scatter nd, only non-deterministic replacement for duplicates
        pytest.skip()
    mean, var, counter = call(ivy_ren.quantize_pixel_coords,
                              td.simple_projected_omni_pixel_coords,
                              np.zeros_like(td.simple_uniform_pixel_coords[..., -2:]),
                              [3, 3], mode='omni',
                              pixel_coords_var=td.vars_to_scatter,
                              prior_var=np.ones_like(td.simple_uniform_pixel_coords[..., -2:]) * 10,
                              var_threshold=np.array([[0., 1e4]]*2))
    assert np.allclose(counter, td.quantized_counter, atol=1e-6)
    assert np.allclose(mean, td.quantized_omni_pixel_coords_from_cov, atol=1e-3)
    assert np.allclose(var, td.quantized_cov_values, atol=1e-3)

    assert call(ivy_ren.quantize_pixel_coords,
                np.ones((1, 9, 3)), np.ones((1, 3, 3, 1)), [3, 3], mode='omni',
                pixel_coords_var=np.random.uniform(size=(1, 9, 1)),
                prior_var=np.ones((1, 3, 3, 1)),
                var_threshold=np.array([[0., 0.5]]))
    assert call(ivy_ren.quantize_pixel_coords,
                np.ones((1, 9, 3)), np.ones((1, 3, 3, 1)), [3, 3], mode='omni',
                pixel_coords_var=np.ones((1, 9, 1)),
                prior_var=np.ones((1, 3, 3, 1)),
                var_threshold=np.array([[0., 0.]]))


def test_quantize_omni_pixel_coords_with_var_db(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet does not support min for scatter nd, only non-deterministic replacement for duplicates
        pytest.skip()
    mean, var, validity_mask = call(ivy_ren.quantize_pixel_coords,
                                    td.simple_projected_omni_pixel_coords,
                                    np.zeros_like(td.simple_uniform_pixel_coords[..., -2:]),
                                    [3, 3], mode='omni', with_db=True,
                                    pixel_coords_var=td.vars_to_scatter,
                                    prior_var=np.ones_like(td.simple_uniform_pixel_coords[..., -2:]) * 10,
                                    var_threshold=np.array([[0., 1e4]]*2))
    assert np.allclose(validity_mask, td.validity_mask, atol=1e-6)
    assert np.allclose(mean, td.quantized_omni_pixel_coords_from_cov_db, atol=1e-3)
    assert np.allclose(var, td.quantized_cov_values_db, atol=1e-3)

    assert call(ivy_ren.quantize_pixel_coords,
                np.ones((1, 9, 3)), np.ones((1, 3, 3, 1)), [3, 3], mode='omni', with_db=True,
                pixel_coords_var=np.random.uniform(size=(1, 9, 1)),
                prior_var=np.ones((1, 3, 3, 1)),
                var_threshold=np.array([[0., 0.5]]))


def test_rasterize_triangles(dev_str, call):
    if call in [helpers.tf_graph_call]:
        # the need to dynamically infer array shapes for scatter makes this only valid in eager mode currently
        pytest.skip()
    assert np.allclose(call(ivy_ren.rasterize_triangles, td.mesh_triangle, [3, 3]), td.rasterized_image, atol=1e-3)
    assert np.allclose(call(ivy_ren.rasterize_triangles, np.expand_dims(td.mesh_triangle, 0), [3, 3],
                            batch_shape=[1])[0], td.rasterized_image, atol=1e-3)


def test_weighted_image_smooth(dev_str, call):
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support depthwise 2d convolutions
        pytest.skip()
    mean_ret, _ = call(ivy_ren.weighted_image_smooth, td.mean_img, 1/td.var_img, td.kernel_size)
    assert np.allclose(mean_ret, td.smoothed_img_from_weights, atol=1e-6)


def test_smooth_image_fom_var_image(dev_str, call):
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support depthwise 2d convolutions
        pytest.skip()
    mean_ret, var_ret = call(ivy_ren.smooth_image_fom_var_image, td.mean_img, td.var_img, td.kernel_size,
                             td.kernel_scale)
    assert np.allclose(mean_ret, td.smoothed_img_from_var, atol=1e-6)
    assert np.allclose(var_ret, td.smoothed_var_from_var, atol=1e-6)


def test_pad_omni_image(dev_str, call):
    assert np.allclose(call(ivy_ren.pad_omni_image, td.omni_image, 1), td.padded_omni_image, atol=1e-3)


def test_create_trimesh_indices_for_image(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet matmul only support N-D*N-D array (N >= 3)
        pytest.skip()
    assert np.allclose(call(ivy_ren.create_trimesh_indices_for_image, [1], [4, 3]),
                       td.tri_mesh_4x3_indices, atol=1e-3)


def test_coord_image_to_trimesh(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet matmul only support N-D*N-D array (N >= 3)
        pytest.skip()
    coord_img = ivy.array(td.coord_img.tolist())
    vertices, trimesh_indices = call(ivy_ren.coord_image_to_trimesh, coord_img,
                                     batch_shape=[1], image_dims=[4, 3], dev_str='cpu')
    assert np.allclose(vertices, td.tri_mesh_4x3_vertices, atol=1e-3)
    assert np.allclose(trimesh_indices, td.tri_mesh_4x3_indices, atol=1e-3)
    vertices, trimesh_indices = call(ivy_ren.coord_image_to_trimesh, td.coord_img, td.coord_validity_img,
                                     batch_shape=[1], image_dims=[4, 3], dev_str='cpu')
    assert np.allclose(vertices, td.tri_mesh_4x3_vertices, atol=1e-3)
    assert np.allclose(trimesh_indices, td.tri_mesh_4x3_valid_indices, atol=1e-3)
