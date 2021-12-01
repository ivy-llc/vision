# global
import ivy
import numpy as np
import ivy.numpy as ivy_np
try:
    import tensorflow as tf
except ImportError:
    pass

# local
from ivy_vision_tests.data import TestData
from ivy_vision import two_view_geometry as ivy_tvg


class TwoViewGeometryTestData(TestData):

    def __init__(self):
        super().__init__()

        # fundamental matrix
        e2 = np.matmul(self.full_mats[:, 1:2], np.concatenate((self.C_hats[:, 0:1],
                                                               np.ones((self.batch_size, 1, 1, 1))), 2))[..., -1]
        e2_skew_matrices = ivy.linalg.vector_to_skew_symmetric_matrix(e2, f=ivy_np)
        self.fund_mats = np.matmul(e2_skew_matrices, np.matmul(self.full_mats[:, 1:2], self.pinv_full_mats[:, 0:1]))

        # closest mutual point
        self.tvg_world_rays = np.concatenate((self.world_rays[:, 0:1], self.proj_world_rays[:, 1:2]), 1)

        # triangulation
        self.tvg_pixel_coords = np.concatenate((self.pixel_coords_to_scatter[:, 0:1], self.proj_pixel_coords[:, 1:2]), 1)

        # pixel cost volume
        self.cv_image1 = np.reshape(np.arange(9, dtype=np.float32), (1, 1, 3, 3, 1))
        self.cv_image2 = np.reshape(np.flip(np.arange(9, dtype=np.float32), 0), (1, 1, 3, 3, 1))
        self.cv = np.array([[[[[0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                               [0.,  0.,  0.,  8.,  7.,  6.,  5.,  4.,  3.],
                               [0.,  0.,  0., 14., 12.,  0.,  8.,  6.,  0.]],
                              [[0., 24., 21.,  0., 15., 12.,  0.,  6.,  3.],
                               [32., 28., 24., 20., 16., 12.,  8.,  4.,  0.],
                               [35., 30.,  0., 20., 15.,  0.,  5.,  0.,  0.]],
                              [[0., 30., 24.,  0., 12.,  6.,  0.,  0.,  0.],
                               [35., 28., 21., 14.,  7.,  0.,  0.,  0.,  0.],
                               [32., 24.,  0.,  8.,  0.,  0.,  0.,  0.,  0.]]]]], dtype=np.float32)


td = TwoViewGeometryTestData()


def test_pixel_to_pixel_coords(dev_str, call):
    assert np.allclose(call(ivy_tvg.ds_pixel_to_ds_pixel_coords, td.pixel_coords_to_scatter[:, 0:1], td.cam2cam_full_mats[:, 0:1]),
                       td.proj_pixel_coords[:, 1:2], atol=1e-6)
    assert np.allclose(call(ivy_tvg.ds_pixel_to_ds_pixel_coords, td.pixel_coords_to_scatter[:, 0], td.cam2cam_full_mats[:, 0]),
                       td.proj_pixel_coords[:, 1], atol=1e-6)


def test_angular_pixel_to_angular_pixel_coords(dev_str, call):
    assert np.allclose(call(ivy_tvg.angular_pixel_to_angular_pixel_coords, td.angular_pixel_coords[:, 0:1],
                            td.cam2cam_ext_mats[:, 0:1], td.pixels_per_degree),
                       td.proj_angular_pixel_coords[:, 1:2], atol=1e-3)
    assert np.allclose(call(ivy_tvg.angular_pixel_to_angular_pixel_coords, td.angular_pixel_coords[:, 0],
                            td.cam2cam_ext_mats[:, 0], td.pixels_per_degree),
                       td.proj_angular_pixel_coords[:, 1], atol=1e-3)


def test_cam_to_cam_coords(dev_str, call):
    assert np.allclose(call(ivy_tvg.cam_to_cam_coords, td.cam_coords[:, 0:1], td.cam2cam_ext_mats[:, 0:1]),
                       td.proj_cam_coords[:, 1:2], atol=1e-6)
    assert np.allclose(call(ivy_tvg.cam_to_cam_coords, td.cam_coords[:, 0], td.cam2cam_ext_mats[:, 0]),
                       td.proj_cam_coords[:, 1], atol=1e-6)


def test_sphere_to_sphere_coords(dev_str, call):
    assert np.allclose(call(ivy_tvg.sphere_to_sphere_coords, td.sphere_coords[:, 0:1], td.cam2cam_ext_mats[:, 0:1]),
                       td.proj_sphere_coords[:, 1:2], atol=1e-5)
    assert np.allclose(call(ivy_tvg.sphere_to_sphere_coords, td.sphere_coords[:, 0], td.cam2cam_ext_mats[:, 0]),
                       td.proj_sphere_coords[:, 1], atol=1e-5)


def test_get_fundamental_matrix(dev_str, call):
    assert np.allclose(call(ivy_tvg.get_fundamental_matrix, td.full_mats[:, 0:1], td.full_mats[:, 1:2]),
                       td.fund_mats, atol=1e-5)
    assert np.allclose(call(ivy_tvg.get_fundamental_matrix, td.full_mats[:, 0], td.full_mats[:, 1]),
                       td.fund_mats[:, 0], atol=1e-5)


def test_closest_mutual_points_along_two_skew_rays(dev_str, call):
    closest_mutual_points = call(ivy_tvg.closest_mutual_points_along_two_skew_rays, td.C_hats, td.tvg_world_rays)
    assert np.allclose(closest_mutual_points[:, 0], td.world_coords[:, 0], atol=1e-6)
    assert np.allclose(closest_mutual_points[:, 1], td.world_coords[:, 0], atol=1e-6)
    assert np.allclose(closest_mutual_points[:, 0:1], td.world_coords[:, 0:1], atol=1e-6)
    assert np.allclose(closest_mutual_points[:, 1:2], td.world_coords[:, 0:1], atol=1e-6)


def test_triangulate_depth_by_closest_mutual_points(dev_str, call):
    assert np.allclose(call(ivy_tvg.triangulate_depth, td.tvg_pixel_coords, td.full_mats, td.inv_full_mats,
                            td.C_hats), td.tvg_pixel_coords[:, 0], atol=1e-4)
    assert np.allclose(call(ivy_tvg.triangulate_depth, td.tvg_pixel_coords[0], td.full_mats[0], td.inv_full_mats[0],
                            td.C_hats[0]), td.tvg_pixel_coords[0, 0], atol=1e-4)


def test_triangulate_depth_by_homogeneous_dlt(dev_str, call):
    assert np.allclose(call(ivy_tvg.triangulate_depth, td.tvg_pixel_coords, td.full_mats, method='dlt'),
                       td.tvg_pixel_coords[:, 0], atol=1e-3)
    assert np.allclose(call(ivy_tvg.triangulate_depth, td.tvg_pixel_coords[0], td.full_mats[0], method='dlt'),
                           td.tvg_pixel_coords[0, 0], atol=1e-3)
