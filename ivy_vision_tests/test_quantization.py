# global
import ivy
import ivy_tests.helpers as helpers
import numpy as np
import pytest

from ivy_vision import quantization as ivy_quant
# local
from ivy_vision_tests.data import TestData


class QuantizationTestData(TestData):

    def __init__(self):
        super().__init__()

        # Pixel Quantization #
        # -------------------#

        self.uniform_pixel_coords = np.array([[[0., 0.], [1., 0.], [2., 0.]],
                                              [[0., 1.], [1., 1.], [2., 1.]],
                                              [[0., 2.], [1., 2.], [2., 2.]]])

        self.pixel_coord_means_to_scatter = np.reshape(np.array([[[0., 0.], [0., 0.], [2., 0.]],
                                                                 [[2., 1.], [1., 1.], [2., 1.]],
                                                                 [[1., 2.], [1., 2.], [1., 2.]]]), (-1, 2))
        self.pixel_coord_vars_to_scatter = np.ones_like(self.pixel_coord_means_to_scatter) * 1e-3

        depth_values = np.array([[[1.], [2.], [3.]],
                                 [[4.], [5.], [6.]],
                                 [[7.], [8.], [9.]]])

        intensity_values = np.array([[[0.9], [0.8], [0.7]],
                                     [[0.6], [0.5], [0.4]],
                                     [[0.3], [0.2], [0.1]]])

        self.feat_means_to_scatter = np.reshape(np.concatenate((depth_values, intensity_values), -1), (-1, 2))
        self.feat_vars_to_scatter = np.reshape(np.array([[[2.] * 2, [4.] * 2, [1.] * 2],
                                                         [[0.5] * 2, [3.] * 2, [3.] * 2],
                                                         [[1.] * 2, [1.] * 2, [2.] * 2]]), (-1, 2))
        self.prior_feat_mean = np.zeros((3, 3, 2))
        self.prior_feat_var = np.ones((3, 3, 2)) * 1e12
        self.prior_pixel_coord_var = np.ones((3, 3, 2)) * 1e-3

        # duplicate averaging from default variance

        self.quantized_pixel_coords = np.array([[[0., 0., 1.5, 0.84999996],
                                                 [1., 0., 0., 0.],
                                                 [2., 0., 3., 0.7]],
                                                [[0., 1., 0., 0.],
                                                 [1., 1., 4.9999995, 0.5],
                                                 [2., 1., 4.9999995, 0.5]],
                                                [[0., 2., 0., 0.],
                                                 [1., 2., 8., 0.20000002],
                                                 [2., 2., 0., 0.]]])
        self.quantized_cov_values = np.array([[[1e-3, 1e-3], [1e12, 1e12], [1e-3, 1e-3]],
                                              [[1e12, 1e12], [1e-3, 1e-3], [1e-3, 1e-3]],
                                              [[1e12, 1e12], [1e-3, 1e-3], [1e12, 1e12]]])

        # duplicate fusing from defined variance

        quantized_cov_values = np.array([[[1 / (1 / 2 + 1 / 4)] * 2, [1e12] * 2, [1.] * 2],
                                         [[1e12] * 2, [3.] * 2, [1 / (1 / 0.5 + 1 / 3)] * 2],
                                         [[1e12] * 2, [1 / (1 / 1 + 1 / 1 + 1 / 2)] * 2, [1e12] * 2]])

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
            np.concatenate((self.uniform_pixel_coords, quantized_mean_vals_from_cov), -1)
        self.quantized_cov_values_from_cov = np.where(self.quantized_counter > 0,
                                                      quantized_cov_values * self.quantized_counter,
                                                      quantized_cov_values)

        # duplicate minimizing

        quantized_cov_values_db = np.array([[[2.] * 2, [1e12] * 2, [1.] * 2],
                                            [[1e12] * 2, [3.] * 2, [0.5] * 2],
                                            [[1e12] * 2, [1.] * 2, [1e12] * 2]])

        quantized_mean_values_db = np.array([[[1., 0.9], [0., 0.], [3., 0.7]],
                                             [[0., 0.], [5., 0.5], [4., 0.6]],
                                             [[0., 0.], [7., 0.3], [0., 0.]]])

        self.quantized_pixel_coords_from_cov_db = \
            np.concatenate((self.uniform_pixel_coords, quantized_mean_values_db), -1)
        self.quantized_cov_values_db = quantized_cov_values_db
        self.validity_mask = self.quantized_counter > 0

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

        self.tri_mesh_4x3_indices = np.array([[[0., 1., 3.],
                                               [1., 2., 4.],
                                               [3., 4., 6.],
                                               [4., 5., 7.],
                                               [6., 7., 9.],
                                               [7., 8., 10.],
                                               [4., 1., 3.],
                                               [5., 2., 4.],
                                               [7., 4., 6.],
                                               [8., 5., 7.],
                                               [10., 7., 9.],
                                               [11., 8., 10.]]])

        self.tri_mesh_4x3_valid_indices = np.array([[3., 4., 6.],
                                                    [5., 2., 4.],
                                                    [11., 8., 10.]])

        self.tri_mesh_4x3_vertices = np.reshape(self.coord_img, [1, -1, 3])


td = QuantizationTestData()


def test_quantize_pixel_coords(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet does not support sum for scatter nd, only non-deterministic replacement for duplicates
        pytest.skip()
    pixel_coords_cont = ivy.Container({'mean': ivy.array(td.pixel_coord_means_to_scatter, 'float32')})
    feat_cont = ivy.Container({'mean': ivy.array(td.feat_means_to_scatter, 'float32')})
    prior_cont = ivy.Container({'feat': {'mean': ivy.array(td.prior_feat_mean, 'float32')}})

    mean, var, counter = call(ivy_quant.quantize_pixel_coords,
                              pixel_coords_cont, feat_cont, prior_cont, [3, 3], np.array([[0., 1e4]] * 4))
    assert np.allclose(counter, td.quantized_counter, atol=1e-6)
    assert np.allclose(mean, td.quantized_pixel_coords, atol=1e-3)
    assert np.allclose(var[..., 2:], td.quantized_cov_values, atol=1e-3)


def test_quantize_pixel_coordinates_with_var(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet does not support sum for scatter nd, only non-deterministic replacement for duplicates
        pytest.skip()
    pixel_coords_cont = ivy.Container({'mean': ivy.array(td.pixel_coord_means_to_scatter, 'float32'),
                                       'var': ivy.array(td.pixel_coord_vars_to_scatter, 'float32')})
    feat_cont = ivy.Container({'mean': ivy.array(td.feat_means_to_scatter, 'float32'),
                               'var': ivy.array(td.feat_vars_to_scatter, 'float32')})
    prior_cont = ivy.Container({'feat': {'mean': ivy.array(td.prior_feat_mean, 'float32'),
                                         'var': ivy.array(td.prior_feat_var, 'float32')},
                                'pixel_coords': {'var': ivy.array(td.prior_pixel_coord_var, 'float32')}})

    mean, var, counter = call(ivy_quant.quantize_pixel_coords,
                              pixel_coords_cont, feat_cont, prior_cont, [3, 3], np.array([[0., 1e4]] * 4))
    assert np.allclose(counter, td.quantized_counter, atol=1e-6)
    assert np.allclose(mean, td.quantized_pixel_coords_from_cov, atol=1e-3)
    assert np.allclose(var[..., 2:], td.quantized_cov_values_from_cov, atol=1e-3)


def test_quantize_pixel_coords_with_var_db(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet does not support min for scatter nd, only non-deterministic replacement for duplicates
        pytest.skip()
    mean, var, validity_mask = call(ivy_quant.quantize_pixel_coords,
                                    td.coords_to_scatter,
                                    np.zeros_like(td.uniform_pixel_coords[..., -2:]),
                                    [3, 3],
                                    with_db=True,
                                    pixel_coords_var=td.vars_to_scatter,
                                    prior_var=np.ones_like(td.uniform_pixel_coords[..., -2:]) * 10,
                                    var_threshold=np.array([[0., 1e4]] * 2))
    assert np.allclose(validity_mask, td.validity_mask, atol=1e-6)
    assert np.allclose(mean, td.quantized_pixel_coords_from_cov_db, atol=1e-3)
    assert np.allclose(var, td.quantized_cov_values_db, atol=1e-3)

    assert call(ivy_quant.quantize_pixel_coords,
                np.ones((1, 9, 3)), np.ones((1, 3, 3, 1)), [3, 3],
                with_db=True,
                pixel_coords_var=np.random.uniform(size=(1, 9, 1)),
                prior_var=np.ones((1, 3, 3, 1)),
                var_threshold=np.array([[0., 0.5]]))
    assert call(ivy_quant.quantize_pixel_coords,
                np.expand_dims(td.coords_to_scatter, 0),
                np.expand_dims(np.zeros_like(td.uniform_pixel_coords[..., -2:]), 0),
                [3, 3],
                with_db=True,
                pixel_coords_var=np.expand_dims(td.vars_to_scatter, 0),
                prior_var=np.expand_dims(np.ones_like(td.uniform_pixel_coords[..., -2:]), 0) * 10,
                var_threshold=np.array([[0., 0.]] * 2))


def test_rasterize_triangles(dev_str, call):
    if call in [helpers.tf_graph_call]:
        # the need to dynamically infer array shapes for scatter makes this only valid in eager mode currently
        pytest.skip()
    assert np.allclose(call(ivy_quant.rasterize_triangles, td.mesh_triangle, [3, 3]), td.rasterized_image, atol=1e-3)
    assert np.allclose(call(ivy_quant.rasterize_triangles, np.expand_dims(td.mesh_triangle, 0), [3, 3],
                            batch_shape=[1])[0], td.rasterized_image, atol=1e-3)


def test_weighted_image_smooth(dev_str, call):
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support depthwise 2d convolutions
        pytest.skip()
    mean_ret, _ = call(ivy_quant.weighted_image_smooth, td.mean_img, 1 / td.var_img, td.kernel_size)
    assert np.allclose(mean_ret, td.smoothed_img_from_weights, atol=1e-6)


def test_smooth_image_fom_var_image(dev_str, call):
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support depthwise 2d convolutions
        pytest.skip()
    mean_ret, var_ret = call(ivy_quant.smooth_image_fom_var_image, td.mean_img, td.var_img, td.kernel_size,
                             td.kernel_scale)
    assert np.allclose(mean_ret, td.smoothed_img_from_var, atol=1e-6)
    assert np.allclose(var_ret, td.smoothed_var_from_var, atol=1e-6)


def test_pad_omni_image(dev_str, call):
    assert np.allclose(call(ivy_quant.pad_omni_image, td.omni_image, 1), td.padded_omni_image, atol=1e-3)


def test_create_trimesh_indices_for_image(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet matmul only support N-D*N-D array (N >= 3)
        pytest.skip()
    assert np.allclose(call(ivy_quant.create_trimesh_indices_for_image, [1], [4, 3]),
                       td.tri_mesh_4x3_indices, atol=1e-3)


def test_coord_image_to_trimesh(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet matmul only support N-D*N-D array (N >= 3)
        pytest.skip()
    coord_img = ivy.array(td.coord_img.tolist())
    vertices, trimesh_indices = call(ivy_quant.coord_image_to_trimesh, coord_img,
                                     batch_shape=[1], image_dims=[4, 3], dev_str='cpu')
    assert np.allclose(vertices, td.tri_mesh_4x3_vertices, atol=1e-3)
    assert np.allclose(trimesh_indices, td.tri_mesh_4x3_indices, atol=1e-3)
    vertices, trimesh_indices = call(ivy_quant.coord_image_to_trimesh, td.coord_img, td.coord_validity_img,
                                     batch_shape=[1], image_dims=[4, 3], dev_str='cpu')
    assert np.allclose(vertices, td.tri_mesh_4x3_vertices, atol=1e-3)
    assert np.allclose(trimesh_indices, td.tri_mesh_4x3_valid_indices, atol=1e-3)
