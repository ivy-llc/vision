# global
import ivy
import pytest
import numpy as np
from ivy_tests.test_ivy import helpers

# local
from ivy_vision import implicit as ivy_imp
from ivy_vision_tests.data import TestData


class ImplicitTestData(TestData):

    def __init__(self):
        super().__init__()

        # set framework
        ivy.set_backend('numpy')

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
        self.z_vals = ivy.expand_dims(ivy_imp.stratified_sample(self.near, self.far, 3), axis=-1)
        rays_d = ivy.expand_dims(self.rays_d, axis=-2)
        rays_o = ivy.broadcast_to(ivy.reshape(self.rays_o, batch_shape + [1] * (num_ray_batch_dims + 1) + [3]),
                                  rays_d.shape)
        self.query_points = rays_o + rays_d * self.z_vals

        # unset framework
        ivy.previous_backend()


td = ImplicitTestData()


def test_downsampled_image_dims_from_desired_num_pixels(dev_str, fw):
    ivy.set_backend(fw)
    new_img_dims, num_pixels = ivy_imp.downsampled_image_dims_from_desired_num_pixels([32, 32], 256)
    assert new_img_dims == [16, 16]
    assert num_pixels == 256
    new_img_dims, num_pixels = ivy_imp.downsampled_image_dims_from_desired_num_pixels([14, 18], 125)
    assert new_img_dims == [10, 13]
    assert num_pixels == 130
    new_img_dims, num_pixels = ivy_imp.downsampled_image_dims_from_desired_num_pixels([14, 18], 125, maximum=True)
    assert new_img_dims == [9, 12]
    assert num_pixels == 108
    ivy.previous_backend()


def test_create_sampled_pixel_coords_image(dev_str, fw):
    if fw == 'mxnet':
        # MXNet does not support clipping based on min or max specified as arrays
        pytest.skip()
    ivy.set_backend(fw)
    sampled_img = ivy_imp.create_sampled_pixel_coords_image(td.image_dims, td.samples_per_dim,
                       (td.batch_size, td.num_cameras), normalized=False, randomize=False)
    assert ivy.min(sampled_img).to_scalar() == 26
    assert ivy.max(sampled_img).to_scalar() == 613
    sampled_img = ivy_imp.create_sampled_pixel_coords_image(td.image_dims, td.samples_per_dim,
                       (td.batch_size, td.num_cameras), normalized=False, randomize=True)
    assert ivy.min(sampled_img).to_scalar() >= 0
    assert ivy.max(sampled_img).to_scalar() < max(td.image_dims)
    sampled_img = ivy_imp.create_sampled_pixel_coords_image(td.image_dims, td.samples_per_dim,
                       (td.batch_size, td.num_cameras), normalized=True, randomize=False)
    assert np.allclose(ivy.min(sampled_img).to_scalar(), 0.040625)
    assert np.allclose(ivy.max(sampled_img).to_scalar(), 0.9578125)
    sampled_img = ivy_imp.create_sampled_pixel_coords_image(td.image_dims, td.samples_per_dim,
                       (td.batch_size, td.num_cameras), normalized=True, randomize=True)
    assert ivy.min(sampled_img).to_scalar() >= 0
    assert ivy.max(sampled_img).to_scalar() < 1
    ivy.previous_backend()


def test_sample_images(dev_str, fw):
    if fw == 'mxnet':
        # MXNet does not support splitting based on section sizes,only number of sections as integer input is supported.
        pytest.skip()
    ivy.set_backend(fw)
    img0, img1 = ivy_imp.sample_images([ivy.array(td.pixel_coords_to_scatter)]*2, 32,
                      (td.batch_size, td.num_cameras), td.image_dims)
    assert list(img0.shape) == [td.batch_size, td.num_cameras, 35, 3]
    assert list(img1.shape) == [td.batch_size, td.num_cameras, 35, 3]
    ivy.previous_backend()


def test_sampled_volume_density_to_occupancy_probability(dev_str, fw):
    ivy.set_backend(fw)
    occ_prob = ivy_imp.sampled_volume_density_to_occupancy_probability(ivy.array(td.densities), ivy.array(td.inter_sample_distances))
    assert occ_prob.shape == td.densities.shape
    assert np.allclose(occ_prob, td.occ_probs)
    ivy.previous_backend()


def test_ray_termination_probabilities(dev_str, fw):
    ivy.set_backend(fw)
    ray_term_probs = ivy_imp.ray_termination_probabilities(ivy.array(td.densities), ivy.array(td.inter_sample_distances))
    assert ray_term_probs.shape == td.densities.shape
    assert np.allclose(ray_term_probs, td.ray_term_probs)
    ivy.previous_backend()


def test_stratified_sample(dev_str, fw):
    ivy.set_backend(fw)
    num = 10
    res = ivy_imp.stratified_sample(ivy.array(td.start_vals), ivy.array(td.end_vals), num)
    assert res.shape == (3, num)
    for i in range(3):
        for j in range(num - 1):
            assert res[i][j] < res[i][j + 1]
    ivy.previous_backend()


def test_render_rays_via_termination_probabilities(dev_str, fw):
    ivy.set_backend(fw)
    rendering, var = ivy_imp.render_rays_via_termination_probabilities(ivy.array(td.ray_term_probs), ivy.array(td.features),
                          render_variance=True)
    assert rendering.shape == td.radial_depths.shape[:-1] + (3,)
    assert var.shape == td.radial_depths.shape[:-1] + (3,)
    assert np.allclose(rendering, td.term_prob_feature_rendering)
    assert np.allclose(var, td.term_prob_var_rendering)
    ivy.previous_backend()


@pytest.mark.parametrize(
    "with_features", [True, False])
@pytest.mark.parametrize(
    "with_timestamps", [True, False])
def test_render_implicit_features_and_depth(dev_str, fw, with_features, with_timestamps):
    if fw == 'mxnet':
        # MXNet does not support splitting with remainder
        pytest.skip()
    ivy.set_backend(fw)
    rgb, depth = ivy_imp.render_implicit_features_and_depth(td.implicit_fn, ivy.array(td.rays_o), ivy.array(td.rays_d), ivy.array(td.near),
                      ivy.array(td.far), td.samples_per_ray, ivy.array(td.timestamps) if with_timestamps else None,
                      inter_feat_fn=td.inter_feat_fn if with_features else None)
    assert rgb.shape == (3, 3)
    assert depth.shape == (3, 1)
    ivy.previous_backend()
