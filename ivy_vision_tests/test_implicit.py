# global
import numpy as np

# local
from ivy_vision import implicit as ivy_imp
from ivy_vision_tests.data import TestData


class ImplicitTestData(TestData):

    def __init__(self):
        super().__init__()

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
        # ToDo: verify the below renderings should indeed be unique, if not, find the bug.
        self.quad_feature_rendering = np.array([[0.06259394, 0.11119541, 0.1571928],
                                                [0.12944466, 0.23419851, 0.13134202],
                                                [0.2815921, 0.20865306, 0.18783332]], np.float32)
        self.term_prob_feature_rendering = np.array([[0.1132895, 0.43268824, 0.17733827],
                                                     [0.14346276, 0.5127491, 0.33358333],
                                                     [0.3136631, 0.51936793, 0.28570426]], np.float32)
        self.term_prob_var_rendering = np.array([[0.08141835, 0.05083185, 0.00626953],
                                                 [0.02940753, 0.0765848, 0.0730124],
                                                 [0.11721418, 0.02952592, 0.02767936]], np.float32)


td = ImplicitTestData()


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
            assert res[i][j] < res[i][j+1]


def test_render_rays_via_termination_probabilities(dev_str, call):
    rendering, var = call(ivy_imp.render_rays_via_termination_probabilities, td.radial_depths, td.features,
                          td.densities, render_variance=True)
    assert rendering.shape == td.radial_depths.shape[:-1] + (3,)
    assert var.shape == td.radial_depths.shape[:-1] + (3,)
    assert np.allclose(rendering, td.term_prob_feature_rendering)
    assert np.allclose(var, td.term_prob_var_rendering)


def test_render_rays_via_quadrature_rule(dev_str, call):
    res = call(ivy_imp.render_rays_via_quadrature_rule, td.radial_depths, td.features, td.densities)
    assert res.shape == td.radial_depths.shape[:-1] + (3,)
    assert np.allclose(res, td.quad_feature_rendering)
