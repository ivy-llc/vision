# global
import ivy
import numpy as np

try:
    pass
except ImportError:
    pass

# local
from ivy_vision_tests.data import TestData
from ivy_vision import two_view_geometry as ivy_tvg


class TwoViewGeometryTestData(TestData):
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


td = TwoViewGeometryTestData()


def test_pixel_to_pixel_coords(dev_str, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_tvg.ds_pixel_to_ds_pixel_coords(
            ivy.array(td.pixel_coords_to_scatter[:, 0:1]),
            ivy.array(td.cam2cam_full_mats[:, 0:1]),
        ),
        td.proj_pixel_coords[:, 1:2],
        atol=1e-6,
    )
    assert np.allclose(
        ivy_tvg.ds_pixel_to_ds_pixel_coords(
            ivy.array(td.pixel_coords_to_scatter[:, 0]),
            ivy.array(td.cam2cam_full_mats[:, 0]),
        ),
        td.proj_pixel_coords[:, 1],
        atol=1e-6,
    )
    ivy.unset_backend()


def test_angular_pixel_to_angular_pixel_coords(dev_str, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_tvg.angular_pixel_to_angular_pixel_coords(
            ivy.array(td.angular_pixel_coords[:, 0:1]),
            ivy.array(td.cam2cam_ext_mats[:, 0:1]),
            td.pixels_per_degree,
        ),
        td.proj_angular_pixel_coords[:, 1:2],
        atol=1e-3,
    )
    assert np.allclose(
        ivy_tvg.angular_pixel_to_angular_pixel_coords(
            ivy.array(td.angular_pixel_coords[:, 0]),
            ivy.array(td.cam2cam_ext_mats[:, 0]),
            td.pixels_per_degree,
        ),
        td.proj_angular_pixel_coords[:, 1],
        atol=1e-3,
    )
    ivy.unset_backend()


def test_cam_to_cam_coords(dev_str, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_tvg.cam_to_cam_coords(
            ivy.array(td.cam_coords[:, 0:1]), ivy.array(td.cam2cam_ext_mats[:, 0:1])
        ),
        td.proj_cam_coords[:, 1:2],
        atol=1e-6,
    )
    assert np.allclose(
        ivy_tvg.cam_to_cam_coords(
            ivy.array(td.cam_coords[:, 0]), ivy.array(td.cam2cam_ext_mats[:, 0])
        ),
        td.proj_cam_coords[:, 1],
        atol=1e-6,
    )
    ivy.unset_backend()


def test_sphere_to_sphere_coords(dev_str, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_tvg.sphere_to_sphere_coords(
            ivy.array(td.sphere_coords[:, 0:1]), ivy.array(td.cam2cam_ext_mats[:, 0:1])
        ),
        td.proj_sphere_coords[:, 1:2],
        atol=1e-5,
    )
    assert np.allclose(
        ivy_tvg.sphere_to_sphere_coords(
            ivy.array(td.sphere_coords[:, 0]), ivy.array(td.cam2cam_ext_mats[:, 0])
        ),
        td.proj_sphere_coords[:, 1],
        atol=1e-5,
    )
    ivy.unset_backend()


def test_get_fundamental_matrix(dev_str, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_tvg.get_fundamental_matrix(
            ivy.array(td.full_mats[:, 0:1]), ivy.array(td.full_mats[:, 1:2])
        ),
        td.fund_mats,
        atol=1e-5,
    )
    assert np.allclose(
        ivy_tvg.get_fundamental_matrix(
            ivy.array(td.full_mats[:, 0]), ivy.array(td.full_mats[:, 1])
        ),
        td.fund_mats[:, 0],
        atol=1e-5,
    )
    ivy.unset_backend()


def test_closest_mutual_points_along_two_skew_rays(dev_str, fw):
    ivy.set_backend(fw)
    closest_mutual_points = ivy_tvg.closest_mutual_points_along_two_skew_rays(
        ivy.array(td.C_hats), ivy.array(td.tvg_world_rays)
    )
    assert np.allclose(closest_mutual_points[:, 0], td.world_coords[:, 0], atol=1e-6)
    assert np.allclose(closest_mutual_points[:, 1], td.world_coords[:, 0], atol=1e-6)
    assert np.allclose(
        closest_mutual_points[:, 0:1], td.world_coords[:, 0:1], atol=1e-6
    )
    assert np.allclose(
        closest_mutual_points[:, 1:2], td.world_coords[:, 0:1], atol=1e-6
    )
    ivy.unset_backend()


def test_triangulate_depth_by_closest_mutual_points(dev_str, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_tvg.triangulate_depth(
            ivy.array(td.tvg_pixel_coords),
            ivy.array(td.full_mats),
            ivy.array(td.inv_full_mats),
            ivy.array(td.C_hats),
        ),
        td.tvg_pixel_coords[:, 0],
        atol=1e-4,
    )
    assert np.allclose(
        ivy_tvg.triangulate_depth(
            ivy.array(td.tvg_pixel_coords[0]),
            ivy.array(td.full_mats[0]),
            ivy.array(td.inv_full_mats[0]),
            ivy.array(td.C_hats[0]),
        ),
        td.tvg_pixel_coords[0, 0],
        atol=1e-4,
    )
    ivy.unset_backend()


def test_triangulate_depth_by_homogeneous_dlt(dev_str, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_tvg.triangulate_depth(
            ivy.array(td.tvg_pixel_coords), ivy.array(td.full_mats), method="dlt"
        ),
        td.tvg_pixel_coords[:, 0],
        atol=1e-3,
    )
    assert np.allclose(
        ivy_tvg.triangulate_depth(
            ivy.array(td.tvg_pixel_coords[0]), ivy.array(td.full_mats[0]), method="dlt"
        ),
        td.tvg_pixel_coords[0, 0],
        atol=1e-3,
    )
    ivy.unset_backend()
