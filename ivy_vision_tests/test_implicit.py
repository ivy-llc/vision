# global
import numpy as np
import ivy_tests.helpers as helpers

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
        self.features = np.array([[[0., 0.5, 0.7], [1., 0.2, 0.3], [0.1, 0.2, 0.3]],
                                  [[0.3, 0.2, 0.1], [0.5, 0.7, 0.3], [0.1, 0.9, 0.7]],
                                  [[0.6, 0.4, 0.2], [0.9, 0.3, 0.6], [0.4, 0.8, 0.5]]], np.float32)
        self.densities = np.array([[0.3, 2.4, 0.4], [0.2, 0.7, 0.4], [0.1, 0.3, 1.2]], np.float32)

        self.quadrature_rendering = np.array([[0.04095065, 0.15215576, 0.08457951],
                                              [0.07130891, 0.20325199, 0.24568483],
                                              [0.14261782, 0.19076426, 0.24895583]], np.float32)


td = ImplicitTestData()


def test_stratified_sample(dev_str, call):
    num = 10
    res = call(ivy_imp.stratified_sample, td.start_vals, td.end_vals, num)
    assert res.shape == (3, num)
    for i in range(3):
        for j in range(num - 1):
            assert res[i][j] < res[i][j+1]


def test_render_rays_via_quadrature_rule(dev_str, call):
    res = call(ivy_imp.render_rays_via_quadrature_rule, td.radial_depths, td.features, td.densities)
    assert res.shape == td.radial_depths.shape[:-1] + (3,)
    assert np.allclose(res, td.quadrature_rendering)
