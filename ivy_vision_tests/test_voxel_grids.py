# global
import numpy as np
import ivy.numpy as ivy_np

# local
import ivy_vision_tests.helpers as helpers
import ivy_vision.voxel_grids as ivy_vg
from ivy_vision_tests.data import TestData


class VoxelGridsTestData(TestData):

    def __init__(self):
        super().__init__()

        # un-batched
        self.simple_world_coords = np.array([[[0., 1.5, 3., 1.],
                                              [1.9, 0., 0.9, 1.]],
                                             [[0.7, 3., 0., 1.],
                                              [3., 0.8, 1.9, 1.]]])
        self.simple_world_coords_flat = np.reshape(self.simple_world_coords, (4, 4))
        self.simple_world_features_flat = np.ones((4, 1))
        self.simple_voxel_grid_dims = (3, 3, 3)
        self.simple_voxel_grid =\
            np.array([[[0, 0, 0], [0, 0, 1], [1, 0, 0]],
                      [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 1, 0], [0, 0, 0], [0, 0, 0]]])
        self.simple_voxel_grid_m1_4_bounded =\
            np.array([[[0, 0, 0], [0, 0, 1], [0, 0, 0]],
                      [[0, 1, 0], [0, 0, 0], [1, 0, 0]],
                      [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.simple_voxel_grid_0_4_bounded = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                                       [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                                       [[0, 1, 0], [0, 0, 0], [0, 0, 0]]])

        # batched
        self.simple_world_coords_batched =\
            np.array([[[[0.5, 1.5, 2.6, 1.],
                        [1.9, 0.7, 0.9, 1.]],
                       [[0.7, 2.9, 0.7, 1.],
                        [2.4, 0.8, 1.9, 1.]]],

                      [[[1., 1., 1., 1],
                        [1., 1., 2., 1]],
                       [[1., 2., 1., 1],
                        [1., 2., 2., 1]]]])
        self.simple_world_coords_batched_flat = np.reshape(self.simple_world_coords_batched, (2, 4, 4))

        self.simple_voxel_grid_batched = np.array([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                                   [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                                   [[1, 0, 1], [0, 0, 0], [0, 0, 0]]],

                                                   [[[1, 0, 1], [0, 0, 0], [1, 0, 1]],
                                                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]])

        # world coords
        self.world_coords_flat = np.reshape(self.world_coords, (1, 2, 480*640, 4))


td = VoxelGridsTestData()


def test_world_coords_to_bounding_voxel_grid():
    for lib, call in helpers.calls:
        if call in [helpers.tf_graph_call, helpers.mx_graph_call]:
            # the need to dynamically infer array shapes for scatter makes this only valid in eager mode currently
            continue
        assert np.allclose(np.sum(
            call(ivy_vg.coords_to_voxel_grid, td.simple_world_coords_flat, np.array([3, 3, 3]))[0], -1) > 0,
                           td.simple_voxel_grid, atol=1e-6)
        assert np.allclose(np.sum(
            call(ivy_vg.coords_to_voxel_grid, td.simple_world_coords_flat, (1, 1, 1), 'RES')[0], -1) > 0,
                           td.simple_voxel_grid, atol=1e-6)
        if call is not helpers.mx_call:
            # MXNet cannot slice arrays with more than 6 dimensions
            assert np.allclose(np.sum(
                call(ivy_vg.coords_to_voxel_grid, td.world_coords_flat, (32, 32, 32), 'DIMS')[0], -1) > 0,
                               np.sum(ivy_vg.coords_to_voxel_grid(
                                   td.world_coords_flat, (32, 32, 32), 'DIMS', f=ivy_np)[0], -1) > 0, atol=1e-6)
            assert np.allclose(np.sum(
                call(ivy_vg.coords_to_voxel_grid, td.world_coords_flat, (0.1, 0.1, 0.1), 'RES')[0], -1) > 0,
                               np.sum(ivy_vg.coords_to_voxel_grid(
                                td.world_coords_flat, (0.1, 0.1, 0.1), 'RES', f=ivy_np)[0], -1) > 0, atol=1e-6)
        # with coord bounds
        assert np.allclose(np.sum(
            call(ivy_vg.coords_to_voxel_grid, td.simple_world_coords_flat, (3, 3, 3),
                 coord_bounds=[-1]*3 + [4]*3)[0], -1) > 0, td.simple_voxel_grid_m1_4_bounded, atol=1e-6)
        assert np.allclose(np.sum(
            call(ivy_vg.coords_to_voxel_grid, td.simple_world_coords_flat, (3, 3, 3),
                 coord_bounds=[0]*3 + [4]*3)[0], -1) > 0, td.simple_voxel_grid_0_4_bounded, atol=1e-6)
        assert np.allclose(np.sum(
            call(ivy_vg.coords_to_voxel_grid, td.simple_world_coords_batched_flat, (3, 3, 3),
                 coord_bounds=[0.5]*3 + [2.5]*3)[0], -1) > 0, td.simple_voxel_grid_batched, atol=1e-6)

        # with features
        assert np.allclose(
            call(ivy_vg.coords_to_voxel_grid, td.simple_world_coords_flat, (3, 3, 3),
                 features=td.simple_world_features_flat)[0][..., 3], td.simple_voxel_grid, atol=1e-6)
