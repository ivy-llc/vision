# global
import ivy
import numpy as np

# local
from ivy_vision import sdf as ivy_sdf
from ivy_vision_tests.data import TestData


class SDFTestData(TestData):
    def __init__(self):
        super().__init__()

        # sphere
        self.sphere_positions = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 2]]])
        self.sphere_radii = np.array([[[1.0], [0.5]]])
        self.sphere_query_positions = np.array(
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 2.0], [0.0, 1.5, 2.0]]]
        )
        self.sphere_sdf_vals = np.array([[[-1.0], [0.0], [-0.5], [0.0]]])

        # cuboid
        self.cuboid_ext_mats = np.concatenate(
            (
                np.expand_dims(np.expand_dims(np.identity(4)[0:3], 0), 0),
                np.linalg.inv(
                    np.array(
                        [[[[0, 1, 0, 1], [0, 0, 1, 2], [1, 0, 0, 3], [0, 0, 0, 1]]]]
                    )
                )[:, :, 0:3, :],
            ),
            1,
        )
        self.cuboid_dims = np.array([[[1.0, 1.0, 1.0], [0.5, 1.0, 0.75]]])
        self.cuboid_query_positions = np.array(
            [[[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.0, 2.0, 3.0], [1.0, 2, 3.25]]]
        )
        self.cuboid_sdf_vals = np.array([[[-0.5], [0.0], [-0.25], [0.0]]])


td = SDFTestData()


def test_sphere_signed_distance(dev_str, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_sdf.sphere_signed_distances(
            ivy.array(td.sphere_positions),
            ivy.array(td.sphere_radii),
            ivy.array(td.sphere_query_positions),
        ),
        td.sphere_sdf_vals,
        atol=1e-6,
    )
    assert np.allclose(
        ivy_sdf.sphere_signed_distances(
            ivy.array(td.sphere_positions[0]),
            ivy.array(td.sphere_radii[0]),
            ivy.array(td.sphere_query_positions[0]),
        ),
        td.sphere_sdf_vals[0],
        atol=1e-6,
    )
    ivy.previous_backend()


def test_cuboid_signed_distance(dev_str, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_sdf.cuboid_signed_distances(
            ivy.array(td.cuboid_ext_mats),
            ivy.array(td.cuboid_dims),
            ivy.array(td.cuboid_query_positions),
        ),
        td.cuboid_sdf_vals,
        atol=1e-6,
    )
    assert np.allclose(
        ivy_sdf.cuboid_signed_distances(
            ivy.array(td.cuboid_ext_mats[0]),
            ivy.array(td.cuboid_dims[0]),
            ivy.array(td.cuboid_query_positions[0]),
        ),
        td.cuboid_sdf_vals[0],
        atol=1e-6,
    )
    ivy.previous_backend()
