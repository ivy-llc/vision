# global
import ivy
import pytest
import numpy as np
from ivy_tests import helpers

# local
from ivy_vision import implicit as ivy_imp
from ivy_vision_tests.data import TestData


class ImplicitTestData(TestData):

    def __init__(self):
        super().__init__()

        # set framework
        ivy.set_framework('numpy')

        # sampled pixel coords
        self.samples_per_dim = [9, 12]

        # stratified sampling
        self.start_vals = np.array([0, 1, 2], np.float32)
        self.end_vals = np.array([10, 21, 7], np.float32)

        # render rays via quadrature rule
        self.radial_depths = np.array([[0, 1, 2, 3],
                                       [1, 1.5, 2, 2.5],
                                       [2, 4, 5, 6]], np.float32)
        self.inter_sample_distances = self.radial_depths[..., 1:] - self.radial_depths[..., :-1]
        self.features = np.array([[[0., 0.5, 0.7], [1., 0.2, 0.3], [0.1, 0.2, 0.3]],
                                  [[0.3, 0.2, 0.1], [0.5, 0.7, 0.3], [0.1, 0.9, 0.7]],
                                  [[0.6, 0.4, 0.2], [0.9, 0.3, 0.6], [0.4, 0.8, 0.5]]], np.float32)
        self.densities = np.array([[0.3, 2.4, 0.4], [0.2, 0.7, 0.4], [0.1, 0.3, 1.2]], np.float32)

        self.occ_probs = np.array([[0.2591818, 0.9092821, 0.32967997],
                                   [0.09516257, 0.29531187, 0.18126929],
                                   [0.18126929, 0.2591818, 0.6988058]], np.float32)
        self.ray_term_probs = np.array([[0.2591818, 0.6736127, 0.0221563],
                                        [0.09516257, 0.26720923, 0.11558241],
                                        [0.18126929, 0.21220009, 0.4238471]], np.float32)
        self.term_prob_feature_rendering = np.array([[0.67582834, 0.2687447, 0.39015797],
                                                     [0.17371163, 0.31010312, 0.1705867],
                                                     [0.46928048, 0.47524542, 0.37549746]], np.float32)
        self.term_prob_var_rendering = np.array([[0.18848504, 0.03828975, 0.00651178],
                                                 [0.01524993, 0.12262806, 0.12084134],
                                                 [0.02295334, 0.03439996, 0.01307792]], np.float32)

        # positional encoding
        self.x = np.array([[0., 2.], [1., 3.], [2., 4.], [3., 5.]])
        self.embedding = np.array(
            [[0., 2., 0., 0.90929743, 1., -0.41614684, 0., -0.7568025, 1., -0.65364362],
             [1., 3., 0.84147098, 0.14112001, 0.54030231, -0.9899925, 0.90929743, -0.2794155, -0.41614684, 0.96017029],
             [2., 4., 0.90929743, -0.7568025, -0.41614684, -0.65364362, -0.7568025, 0.98935825, -0.65364362, -0.14550003],
             [3., 5., 0.14112001, -0.95892427, -0.9899925, 0.28366219, -0.2794155, -0.54402111, 0.96017029, -0.83907153]])

        # render implicit features and depth
        self.implicit_fn = lambda pts, feat, timestamps, with_grads=True, v=None:\
            (ivy.array(self.features), ivy.array(self.densities))
        self.rays_o = np.array([0, 0, 0], np.float32)
        self.rays_d = np.array([[1, 2, 3], [-1, -2, -3], [1, -2, 1]], np.float32)
        self.near = np.array([0.5, 0.7, 0.9], np.float32)
        self.far = np.array([6, 7, 8], np.float32)
        self.samples_per_ray = 3
        self.timestamps = np.array([0], np.float32)
        self.inter_feat_fn = lambda x: x ** 2

        # render implicit features and depth from net inputs
        batch_shape = list(self.rays_o.shape[:-1])
        num_batch_dims = len(batch_shape)
        ray_batch_shape = list(self.rays_d.shape[num_batch_dims:-1])
        num_ray_batch_dims = len(ray_batch_shape)
        self.z_vals = ivy.expand_dims(ivy_imp.stratified_sample(self.near, self.far, 3), -1)
        rays_d = ivy.expand_dims(self.rays_d, -2)
        rays_o = ivy.broadcast_to(ivy.reshape(self.rays_o, batch_shape + [1] * (num_ray_batch_dims + 1) + [3]),
                                  rays_d.shape)
        self.query_points = rays_o + rays_d * self.z_vals

        # unset framework
        ivy.unset_framework()


td = ImplicitTestData()


def test_downsampled_image_dims_from_desired_num_pixels(dev_str, call):
    new_img_dims, num_pixels = call(ivy_imp.downsampled_image_dims_from_desired_num_pixels, [32, 32], 256)
    assert new_img_dims == [16, 16]
    assert num_pixels == 256
    new_img_dims, num_pixels = call(ivy_imp.downsampled_image_dims_from_desired_num_pixels, [14, 18], 125)
    assert new_img_dims == [10, 13]
    assert num_pixels == 130
    new_img_dims, num_pixels = call(ivy_imp.downsampled_image_dims_from_desired_num_pixels, [14, 18], 125, maximum=True)
    assert new_img_dims == [9, 12]
    assert num_pixels == 108


def test_create_sampled_pixel_coords_image(dev_str, call):
    if call is helpers.mx_call:
        # MXNet does not support clipping based on min or max specified as arrays
        pytest.skip()
    sampled_img = call(ivy_imp.create_sampled_pixel_coords_image, td.image_dims, td.samples_per_dim,
                       (td.batch_size, td.num_cameras), normalized=False, randomize=False)
    assert np.min(sampled_img).item() == 26
    assert np.max(sampled_img).item() == 613
    sampled_img = call(ivy_imp.create_sampled_pixel_coords_image, td.image_dims, td.samples_per_dim,
                       (td.batch_size, td.num_cameras), normalized=False, randomize=True)
    assert np.min(sampled_img).item() >= 0
    assert np.max(sampled_img).item() < max(td.image_dims)
    sampled_img = call(ivy_imp.create_sampled_pixel_coords_image, td.image_dims, td.samples_per_dim,
                       (td.batch_size, td.num_cameras), normalized=True, randomize=False)
    assert np.allclose(np.min(sampled_img).item(), 0.040625)
    assert np.allclose(np.max(sampled_img).item(), 0.9578125)
    sampled_img = call(ivy_imp.create_sampled_pixel_coords_image, td.image_dims, td.samples_per_dim,
                       (td.batch_size, td.num_cameras), normalized=True, randomize=True)
    assert np.min(sampled_img).item() >= 0
    assert np.max(sampled_img).item() < 1


def test_sample_images(dev_str, call):
    if call is helpers.mx_call:
        # MXNet does not support splitting based on section sizes,only number of sections as integer input is supported.
        pytest.skip()
    img0, img1 = call(ivy_imp.sample_images, [td.pixel_coords_to_scatter]*2, 32,
                      (td.batch_size, td.num_cameras), td.image_dims)
    assert list(img0.shape) == [td.batch_size, td.num_cameras, 35, 3]
    assert list(img1.shape) == [td.batch_size, td.num_cameras, 35, 3]


def test_sinusoid_positional_encoding(dev_str, call):
    embed_length = 2
    embedding = call(ivy_imp.sinusoid_positional_encoding, td.x, embed_length)
    assert embedding.shape[-1] == td.x.shape[-1] + td.x.shape[-1] * 2 * embed_length
    assert np.allclose(embedding, td.embedding)


def test_sampled_volume_density_to_occupancy_probability(dev_str, call):
    occ_prob = call(ivy_imp.sampled_volume_density_to_occupancy_probability, td.densities, td.inter_sample_distances)
    assert occ_prob.shape == td.densities.shape
    assert np.allclose(occ_prob, td.occ_probs)


def test_ray_termination_probabilities(dev_str, call):
    ray_term_probs = call(ivy_imp.ray_termination_probabilities, td.densities, td.inter_sample_distances)
    assert ray_term_probs.shape == td.densities.shape
    assert np.allclose(ray_term_probs, td.ray_term_probs)


def test_stratified_sample(dev_str, call):
    num = 10
    res = call(ivy_imp.stratified_sample, td.start_vals, td.end_vals, num)
    assert res.shape == (3, num)
    for i in range(3):
        for j in range(num - 1):
            assert res[i][j] < res[i][j + 1]


def test_render_rays_via_termination_probabilities(dev_str, call):
    rendering, var = call(ivy_imp.render_rays_via_termination_probabilities, td.ray_term_probs, td.features,
                          render_variance=True)
    assert rendering.shape == td.radial_depths.shape[:-1] + (3,)
    assert var.shape == td.radial_depths.shape[:-1] + (3,)
    assert np.allclose(rendering, td.term_prob_feature_rendering)
    assert np.allclose(var, td.term_prob_var_rendering)


@pytest.mark.parametrize(
    "with_features", [True, False])
@pytest.mark.parametrize(
    "with_timestamps", [True, False])
def test_render_implicit_features_and_depth(dev_str, call, with_features, with_timestamps):
    if call is helpers.mx_call:
        # MXNet does not support splitting with remainder
        pytest.skip()
    rgb, depth = call(ivy_imp.render_implicit_features_and_depth, td.implicit_fn, td.rays_o, td.rays_d, td.near,
                      td.far, td.samples_per_ray, td.timestamps if with_timestamps else None,
                      inter_feat_fn=td.inter_feat_fn if with_features else None)
    assert rgb.shape == (3, 3)
    assert depth.shape == (3, 1)
