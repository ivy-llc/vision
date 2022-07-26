# global
import pytest
import numpy as np
import ivy_tests.test_ivy.helpers as helpers

# local
from ivy_vision_tests.data import TestData
from ivy_vision import smoothing as ivy_smooth


class SmoothingTestData(TestData):

    def __init__(self):
        super().__init__()

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


td = SmoothingTestData()


def test_weighted_image_smooth(device, call):
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support depthwise 2d convolutions
        pytest.skip()
    mean_ret, _ = call(ivy_smooth.weighted_image_smooth, td.mean_img, 1 / td.var_img, td.kernel_size)
    assert np.allclose(mean_ret, td.smoothed_img_from_weights, atol=1e-6)


def test_smooth_image_fom_var_image(device, call):
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support depthwise 2d convolutions
        pytest.skip()
    mean_ret, var_ret = call(ivy_smooth.smooth_image_fom_var_image, td.mean_img, td.var_img, td.kernel_size,
                             td.kernel_scale)
    assert np.allclose(mean_ret, td.smoothed_img_from_var, atol=1e-6)
    assert np.allclose(var_ret, td.smoothed_var_from_var, atol=1e-6)
