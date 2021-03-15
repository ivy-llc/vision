# global
import numpy as np

# local
from ivy_vision_tests.data import TestData
import ivy_vision.projective_geometry as ivy_pg


class ProjectiveGeometryTestData(TestData):

    def __init__(self):
        super().__init__()

        # solve homogeneous DLT
        self.A = np.array([[[[1., 2.],
                             [3., 4.]]],

                           [[[5., 6.],
                             [7., 8.]]]])

        _, _, VT = np.linalg.svd(self.A)
        self.X = VT[..., -1, :]


td = ProjectiveGeometryTestData()


def test_transform(dev_str, call):
    assert np.allclose(call(ivy_pg.transform, td.world_coords, td.ext_mats), td.cam_coords[:, :, :, :, 0:3], atol=1e-6)
    assert np.allclose(call(ivy_pg.transform, td.world_coords[0], td.ext_mats[0]),
                       td.cam_coords[0, :, :, :, 0:3], atol=1e-6)


def test_projection_matrix_pseudo_inverse(dev_str, call):
    assert np.allclose(call(ivy_pg.projection_matrix_pseudo_inverse, td.ext_mats), td.pinv_ext_mats, atol=1e-6)
    assert np.allclose(call(ivy_pg.projection_matrix_pseudo_inverse, td.ext_mats[0]), td.pinv_ext_mats[0], atol=1e-6)


def test_projection_matrix_inverse(dev_str, call):
    assert np.allclose(call(ivy_pg.projection_matrix_inverse, td.ext_mats), td.inv_ext_mats, atol=1e-6)
    assert np.allclose(call(ivy_pg.projection_matrix_inverse, td.ext_mats[0]), td.inv_ext_mats[0], atol=1e-6)


def test_solve_homogeneous_dlt(dev_str, call):
    assert np.allclose(call(ivy_pg.solve_homogeneous_dlt, td.A), td.X, atol=1e-6)
    assert np.allclose(call(ivy_pg.solve_homogeneous_dlt, td.A[0]), td.X[0], atol=1e-6)
