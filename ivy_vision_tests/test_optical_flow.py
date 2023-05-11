# global
import ivy
import pytest
import numpy as np
import ivy_tests.test_ivy.helpers as helpers

try:
    pass
except ImportError:
    pass

# local
from ivy_vision_tests.data import TestData
from ivy_vision import optical_flow as ivy_flow


class OpticalFlowTestData(TestData):
    def __init__(self):
        super().__init__()

        # fundamental matrix
        e2 = np.matmul(
            self.full_mats[:, 1:2],
            np.concatenate(
                (self.C_hats[:, 0:1], np.ones((self.batch_size, 1, 1, 1))), 2
            ),
        )[..., -1]
        e2_skew_matrices = ivy.vector_to_skew_symmetric_matrix(e2)
        self.fund_mats = np.matmul(
            e2_skew_matrices,
            np.matmul(self.full_mats[:, 1:2], self.pinv_full_mats[:, 0:1]),
        )

        # closest mutual point
        self.tvg_world_rays = np.concatenate(
            (self.world_rays[:, 0:1], self.proj_world_rays[:, 1:2]), 1
        )

        # triangulation
        self.tvg_pixel_coords = np.concatenate(
            (self.pixel_coords_to_scatter[:, 0:1], self.proj_pixel_coords[:, 1:2]), 1
        )

        # pixel cost volume
        self.cv_image1 = np.reshape(np.arange(9, dtype=np.float32), (1, 1, 3, 3, 1))
        self.cv_image2 = np.reshape(
            np.flip(np.arange(9, dtype=np.float32), 0), (1, 1, 3, 3, 1)
        )
        self.cv = np.array(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
                            [0.0, 0.0, 0.0, 14.0, 12.0, 0.0, 8.0, 6.0, 0.0],
                        ],
                        [
                            [0.0, 24.0, 21.0, 0.0, 15.0, 12.0, 0.0, 6.0, 3.0],
                            [32.0, 28.0, 24.0, 20.0, 16.0, 12.0, 8.0, 4.0, 0.0],
                            [35.0, 30.0, 0.0, 20.0, 15.0, 0.0, 5.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 30.0, 24.0, 0.0, 12.0, 6.0, 0.0, 0.0, 0.0],
                            [35.0, 28.0, 21.0, 14.0, 7.0, 0.0, 0.0, 0.0, 0.0],
                            [32.0, 24.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                    ]
                ]
            ],
            dtype=np.float32,
        )


td = OpticalFlowTestData()


def test_depth_from_flow_and_cam_poses(dev_str, fw):
    assert np.allclose(ivy_flow.depth_from_flow_and_cam_mats(ivy.array(td.optical_flow), ivy.array(td.full_mats)),
                       td.depth_maps[:, 0:1], atol=1e-6)
    assert np.allclose(
        ivy_flow.depth_from_flow_and_cam_mats(ivy.array(td.optical_flow[0]), ivy.array(td.full_mats[0])),
        td.depth_maps[0, 0:1], atol=1e-6)


def test_flow_from_depth_and_cam_poses(dev_str, fw):
    assert np.allclose(ivy_flow.flow_from_depth_and_cam_mats(ivy.array(td.pixel_coords_to_scatter[:, 0:1]),
                            ivy.array(td.cam2cam_full_mats[:, 0:1])), td.optical_flow, atol=1e-3)
    assert np.allclose(ivy_flow.flow_from_depth_and_cam_mats(ivy.array(td.pixel_coords_to_scatter[0, 0:1]),
                            ivy.array(td.cam2cam_full_mats[0, 0:1])), td.optical_flow[0], atol=1e-3)


def test_project_flow_to_epipolar_line(dev_str, fw):
    assert np.allclose(
        ivy_flow.project_flow_to_epipolar_line(ivy.array(td.optical_flow), ivy.array(td.fund_mats[0])), td.optical_flow, atol=1e-3)
    assert np.allclose(
        ivy_flow.project_flow_to_epipolar_line(ivy.array(td.optical_flow[0]), ivy.array(td.fund_mats[0, 0])),
        td.optical_flow[0], atol=1e-3)


def test_pixel_cost_volume(dev_str, fw):
    assert np.allclose(ivy_flow.pixel_cost_volume(ivy.array(td.cv_image1), ivy.array(td.cv_image2), 1), td.cv, atol=1e-3)
    assert np.allclose(ivy_flow.pixel_cost_volume(ivy.array(td.cv_image1[0]), ivy.array(td.cv_image2[0]), 1), td.cv[0], atol=1e-3)


def test_velocity_from_flow_cam_coords_and_cam_mats(dev_str, fw):
    assert ivy_flow.velocity_from_flow_cam_coords_and_cam_mats(ivy.array(td.optical_flow), ivy.array(td.cam_coords[:, 0]), td.cam_coords[:, 1], td.cam2cam_ext_mats[:, 1], td.delta_t)


def test_project_cam_coords_with_object_transformations(dev_str, fw):

    # test data
    np.random.seed(0)
    cam_coords_t = np.array(
        [
            [
                [[0.0, 1.0, 2.0, 1.0], [1.0, 2.0, 3.0, 1.0]],
                [[2.0, 3.0, 4.0, 1.0], [3.0, 4.0, 5.0, 1.0]],
            ]
        ],
        np.float32,
    )

    id_image = np.array([[[[0.0], [1.0]], [[2.0], [3.0]]]], np.float32)

    obj_ids = np.array([[[0.0], [1.0], [2.0], [3.0]]], np.float32)

    obj_trans = np.random.uniform(0, 1, 48).astype(np.float32).reshape((1, 4, 3, 4))

    cam2cam_mat = np.random.uniform(0, 1, 12).astype(np.float32).reshape((1, 3, 4))

    true_reprojection = np.array(
        [
            [
                [
                    [2.4655993, 2.4128416, 2.4957864, 1.0],
                    [2.7194753, 4.8899407, 4.741903, 1.0],
                ],
                [
                    [3.6743941, 4.1201386, 3.3103971, 1.0],
                    [9.704583, 6.375033, 5.863691, 1.0],
                ],
            ]
        ],
        dtype=np.float32,
    )

    # testing
    assert np.allclose(ivy_flow.project_cam_coords_with_object_transformations(ivy.array(cam_coords_t), ivy.array(id_image),
                            obj_ids, obj_trans, cam2cam_mat)[0], true_reprojection, atol=1e-6)


def test_velocity_from_cam_coords_id_image_and_object_trans(dev_str, fw):

    # test data
    np.random.seed(0)
    cam_coords_t = np.array(
        [
            [
                [[0.0, 1.0, 2.0, 1.0], [1.0, 2.0, 3.0, 1.0]],
                [[2.0, 3.0, 4.0, 1.0], [3.0, 4.0, 5.0, 1.0]],
            ]
        ],
        np.float32,
    )

    id_image = np.array([[[[0.0], [1.0]], [[2.0], [3.0]]]], np.float32)

    obj_ids = np.array([[[0.0], [1.0], [2.0], [3.0]]], np.float32)

    obj_trans = np.random.uniform(0, 1, 48).astype(np.float32).reshape((1, 4, 3, 4))

    delta_t = np.array([[0.05]], np.float32)

    true_vel = np.array(
        [
            [
                [
                    [-49.311985, -28.25683, -9.915729],
                    [-34.389503, -57.798813, -34.838055],
                ],
                [
                    [-33.48788, -22.402773, 13.792057],
                    [-134.09166, -47.500656, -17.273817],
                ],
            ]
        ],
        np.float32,
    )

    # testing
    assert np.allclose(ivy_flow.velocity_from_cam_coords_id_image_and_object_trans(ivy.array(cam_coords_t), ivy.array(id_image),
                            obj_ids, obj_trans, delta_t), true_vel, atol=1e-6)


def test_flow_from_cam_coords_id_image_and_object_trans(dev_str, fw):

    # test data
    np.random.seed(0)
    cam_coords_1 = np.array(
        [
            [
                [[0.0, 1.0, 2.0, 1.0], [1.0, 2.0, 3.0, 1.0]],
                [[2.0, 3.0, 4.0, 1.0], [3.0, 4.0, 5.0, 1.0]],
            ]
        ],
        np.float32,
    )

    id_image = np.array([[[[0.0], [1.0]], [[2.0], [3.0]]]], np.float32)

    obj_ids = np.array([[[0.0], [1.0], [2.0], [3.0]]], np.float32)

    obj_trans = np.random.uniform(0, 1, 48).astype(np.float32).reshape((1, 4, 3, 4))

    calib_mat = np.random.uniform(0, 1, 9).astype(np.float32).reshape((1, 3, 3))

    cam1to2_ext_mat = np.random.uniform(0, 1, 12).astype(np.float32).reshape((1, 3, 4))

    true_flow = np.array([[[[0.18979263, 0.655679],
                            [0.09357154, 0.3690083]],
                           [[0.12552917, 0.46584404],
                            [0.10946929, 0.34653413]]]], dtype=float)

    # testing
    assert np.allclose(ivy_flow.flow_from_cam_coords_id_image_and_object_trans(ivy.array(cam_coords_1), ivy.array(id_image),
                            obj_ids, obj_trans, calib_mat, cam1to2_ext_mat), true_flow, atol=1e-6)
