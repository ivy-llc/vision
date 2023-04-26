# global
import ivy
import numpy as np
import pytest

# local
from ivy_vision_tests.data import TestData
from ivy_vision import quantization as ivy_quant


class QuantizationTestData(TestData):
    def __init__(self):
        super().__init__()

        self.uniform_pixel_coords = np.array(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
                [[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]],
            ]
        )

        self.pixel_coords_to_scatter = np.reshape(
            np.array(
                [
                    [[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]],
                    [[2.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
                    [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
                ]
            ),
            (-1, 2),
        )
        self.pixel_coord_vars_to_scatter = (
            np.ones_like(self.pixel_coords_to_scatter) * 1e-3
        )

        depth_values = np.array(
            [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]], [[7.0], [8.0], [9.0]]]
        )

        intensity_values = np.array(
            [[[0.9], [0.8], [0.7]], [[0.6], [0.5], [0.4]], [[0.3], [0.2], [0.1]]]
        )

        self.feats_to_scatter = np.reshape(
            np.concatenate((depth_values, intensity_values), -1), (-1, 2)
        )
        self.feat_vars_to_scatter = np.reshape(
            np.array(
                [
                    [[2.0] * 2, [4.0] * 2, [1.0] * 2],
                    [[0.5] * 2, [3.0] * 2, [3.0] * 2],
                    [[1.0] * 2, [1.0] * 2, [2.0] * 2],
                ]
            ),
            (-1, 2),
        )
        self.feat_prior = np.zeros((3, 3, 2))
        self.feat_prior_var = np.ones((3, 3, 2)) * 1e12
        self.pixel_coord_prior_var = np.ones((3, 3, 2)) * 1e-3

        # duplicate averaging from default variance

        self.quantized_pixel_coords = np.array(
            [
                [
                    [0.0, 0.0, 1.5, 0.84999996],
                    [1.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 3.0, 0.7],
                ],
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 4.9999995, 0.5],
                    [2.0, 1.0, 4.9999995, 0.5],
                ],
                [
                    [0.0, 2.0, 0.0, 0.0],
                    [1.0, 2.0, 8.0, 0.20000002],
                    [2.0, 2.0, 0.0, 0.0],
                ],
            ]
        )
        self.quantized_cov_values = np.array(
            [
                [[1e-3, 1e-3], [1e12, 1e12], [1e-3, 1e-3]],
                [[1e12, 1e12], [1e-3, 1e-3], [1e-3, 1e-3]],
                [[1e12, 1e12], [1e-3, 1e-3], [1e12, 1e12]],
            ]
        )

        # duplicate fusing from defined variance

        quantized_cov_values = np.array(
            [
                [[1 / (1 / 2 + 1 / 4)] * 2, [1e12] * 2, [1.0] * 2],
                [[1e12] * 2, [3.0] * 2, [1 / (1 / 0.5 + 1 / 3)] * 2],
                [[1e12] * 2, [1 / (1 / 1 + 1 / 1 + 1 / 2)] * 2, [1e12] * 2],
            ]
        )

        quantized_sum_mean_x_recip_cov_values = np.array(
            [
                [
                    [((1 / 2) * 1 + (1 / 4) * 2), ((1 / 2) * 0.9 + (1 / 4) * 0.8)],
                    [0.0, 0.0],
                    [(1 / 1) * 3, (1 / 1) * 0.7],
                ],
                [
                    [0.0, 0],
                    [(1 / 3) * 5, (1 / 3) * 0.5],
                    [((1 / 0.5) * 4 + (1 / 3) * 6), ((1 / 0.5) * 0.6 + (1 / 3) * 0.4)],
                ],
                [
                    [0.0, 0.0],
                    [
                        ((1 / 1) * 7 + (1 / 1) * 8 + (1 / 2) * 9),
                        ((1 / 1) * 0.3 + (1 / 1) * 0.2 + (1 / 2) * 0.1),
                    ],
                    [0.0, 0.0],
                ],
            ]
        )

        self.quantized_counter = np.array(
            [[[2.0], [0.0], [1.0]], [[0.0], [1.0], [2.0]], [[0.0], [3.0], [0.0]]]
        )

        quantized_mean_vals_from_cov = (
            quantized_cov_values * quantized_sum_mean_x_recip_cov_values
        )
        self.quantized_pixel_coords_from_cov = np.concatenate(
            (self.uniform_pixel_coords, quantized_mean_vals_from_cov), -1
        )
        self.quantized_cov_values_from_cov = np.where(
            self.quantized_counter > 0,
            quantized_cov_values * self.quantized_counter,
            quantized_cov_values,
        )

        # duplicate minimizing

        quantized_cov_values_db = np.array(
            [
                [[2.0] * 2, [1e12] * 2, [1.0] * 2],
                [[1e12] * 2, [3.0] * 2, [0.5] * 2],
                [[1e12] * 2, [1.0] * 2, [1e12] * 2],
            ]
        )

        quantized_mean_values_db = np.array(
            [
                [[1.0, 0.9], [0.0, 0.0], [3.0, 0.7]],
                [[0.0, 0.0], [5.0, 0.5], [4.0, 0.6]],
                [[0.0, 0.0], [7.0, 0.3], [0.0, 0.0]],
            ]
        )

        self.quantized_pixel_coords_from_cov_db = np.concatenate(
            (self.uniform_pixel_coords, quantized_mean_values_db), -1
        )
        self.quantized_cov_values_db = quantized_cov_values_db
        self.validity_mask = self.quantized_counter > 0


td = QuantizationTestData()


def test_quantize_pixel_coords(dev_str, fw):
    if fw == "mxnet":
        # mxnet does not support sum for scatter nd, only non-deterministic replacement for duplicates
        pytest.skip()
    mean, var, counter = ivy_quant.quantize_to_image(
        ivy.array(td.pixel_coords_to_scatter),
        [3, 3],
        ivy.array(td.feats_to_scatter),
        ivy.array(td.feat_prior),
        var_threshold=ivy.array([[0.0, 1e4]] * 4),
    )
    assert np.allclose(counter, td.quantized_counter, atol=1e-6)
    assert np.allclose(mean, td.quantized_pixel_coords, atol=1e-3)
    assert np.allclose(var[..., 2:], td.quantized_cov_values, atol=1e-3)


def test_quantize_pixel_coordinates_with_var(dev_str, fw):
    if fw == "mxnet":
        # mxnet does not support sum for scatter nd, only non-deterministic replacement for duplicates
        pytest.skip()
    mean, var, counter = ivy_quant.quantize_to_image(
        ivy.array(td.pixel_coords_to_scatter),
        [3, 3],
        ivy.array(td.feats_to_scatter),
        ivy.array(td.feat_prior),
        pixel_coords_var=ivy.array(td.pixel_coord_vars_to_scatter),
        feat_var=ivy.array(td.feat_vars_to_scatter),
        pixel_coords_prior_var=ivy.array(td.pixel_coord_prior_var),
        feat_prior_var=ivy.array(td.feat_prior_var),
        var_threshold=ivy.array([[0.0, 1e4]] * 4),
    )
    assert np.allclose(counter, td.quantized_counter, atol=1e-6)
    assert np.allclose(mean, td.quantized_pixel_coords_from_cov, atol=1e-3)
    assert np.allclose(var[..., 2:], td.quantized_cov_values_from_cov, atol=1e-3)


def test_quantize_pixel_coords_with_var_db(dev_str, fw):
    if fw == "mxnet":
        # mxnet does not support min for scatter nd, only non-deterministic replacement for duplicates
        pytest.skip()
    mean, var, counter = ivy_quant.quantize_to_image(
        ivy.array(td.pixel_coords_to_scatter),
        [3, 3],
        ivy.array(td.feats_to_scatter),
        ivy.array(td.feat_prior),
        with_db=True,
        pixel_coords_var=ivy.array(td.pixel_coord_vars_to_scatter),
        feat_var=ivy.array(td.feat_vars_to_scatter),
        pixel_coords_prior_var=ivy.array(td.pixel_coord_prior_var),
        feat_prior_var=ivy.array(td.feat_prior_var),
        var_threshold=ivy.array([[0.0, 1e4]] * 4),
    )
    assert np.allclose(counter == -1, td.validity_mask, atol=1e-6)
    assert np.allclose(mean, td.quantized_pixel_coords_from_cov_db, atol=1e-3)
    assert np.allclose(var[..., 2:], td.quantized_cov_values_db, atol=1e-3)
