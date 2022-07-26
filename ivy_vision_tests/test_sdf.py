# global
import numpy as np

# local
from ivy_vision import sdf as ivy_sdf
from ivy_vision_tests.data import TestData


class SDFTestData(TestData):

    def __init__(self):
        super().__init__()

        # sphere
        self.sphere_positions = np.array([[[0., 0., 0.], [0., 1., 2]]])
        self.sphere_radii = np.array([[[1.], [0.5]]])
        self.sphere_query_positions = np.array([[[0., 0., 0.], [0., 1., 0.], [0., 1., 2.], [0., 1.5, 2.]]])
        self.sphere_sdf_vals = np.array([[[-1.], [0.], [-0.5], [0.]]])

        # cuboid
        self.cuboid_ext_mats = np.concatenate((np.expand_dims(np.expand_dims(np.identity(4)[0:3], 0), 0),
                                               np.linalg.inv(np.array([[[[0, 1, 0, 1],
                                                                         [0, 0, 1, 2],
                                                                         [1, 0, 0, 3],
                                                                         [0, 0, 0, 1]]]]))[:, :, 0:3, :]), 1)
        self.cuboid_dims = np.array([[[1., 1., 1.], [0.5, 1.0, 0.75]]])
        self.cuboid_query_positions = np.array([[[0., 0., 0.], [0., 0.5, 0.], [1., 2., 3.], [1., 2, 3.25]]])
        self.cuboid_sdf_vals = np.array([[[-0.5], [0.], [-0.25], [0.]]])


td = SDFTestData()


def test_sphere_signed_distance(device, call):
    assert np.allclose(call(ivy_sdf.sphere_signed_distances, td.sphere_positions, td.sphere_radii,
                            td.sphere_query_positions), td.sphere_sdf_vals, atol=1e-6)
    assert np.allclose(call(ivy_sdf.sphere_signed_distances, td.sphere_positions[0], td.sphere_radii[0],
                            td.sphere_query_positions[0]), td.sphere_sdf_vals[0], atol=1e-6)


def test_cuboid_signed_distance(device, call):
    assert np.allclose(call(ivy_sdf.cuboid_signed_distances, td.cuboid_ext_mats, td.cuboid_dims,
                            td.cuboid_query_positions), td.cuboid_sdf_vals, atol=1e-6)
    assert np.allclose(call(ivy_sdf.cuboid_signed_distances, td.cuboid_ext_mats[0], td.cuboid_dims[0],
                            td.cuboid_query_positions[0]), td.cuboid_sdf_vals[0], atol=1e-6)
