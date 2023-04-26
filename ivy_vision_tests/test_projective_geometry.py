# global
import ivy
import numpy as np

# local
from ivy_vision_tests.data import TestData
import ivy_vision.projective_geometry as ivy_pg


class ProjectiveGeometryTestData(TestData):
    def __init__(self):
        super().__init__()

        # solve homogeneous DLT
        self.A = np.array([[[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]]])

        _, _, VT = np.linalg.svd(self.A)
        self.X = VT[..., -1, :]


td = ProjectiveGeometryTestData()


def test_transform(dev_str, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_pg.transform(ivy.array(td.world_coords), ivy.array(td.ext_mats)),
        td.cam_coords[:, :, :, :, 0:3],
        atol=1e-6,
    )
    assert np.allclose(
        ivy_pg.transform(ivy.array(td.world_coords[0]), ivy.array(td.ext_mats[0])),
        td.cam_coords[0, :, :, :, 0:3],
        atol=1e-6,
    )
    ivy.unset_backend()


def test_projection_matrix_pseudo_inverse(dev_str, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_pg.projection_matrix_pseudo_inverse(ivy.array(td.ext_mats)),
        td.pinv_ext_mats,
        atol=1e-6,
    )
    assert np.allclose(
        ivy_pg.projection_matrix_pseudo_inverse(ivy.array(td.ext_mats[0])),
        td.pinv_ext_mats[0],
        atol=1e-6,
    )
    ivy.unset_backend()


def test_projection_matrix_inverse(dev_str, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_pg.projection_matrix_inverse(ivy.array(td.ext_mats)),
        td.inv_ext_mats,
        atol=1e-6,
    )
    assert np.allclose(
        ivy_pg.projection_matrix_inverse(ivy.array(td.ext_mats[0])),
        td.inv_ext_mats[0],
        atol=1e-6,
    )
    ivy.unset_backend()


def test_solve_homogeneous_dlt(dev_str, fw):
    ivy.set_backend(fw)
    assert np.allclose(ivy_pg.solve_homogeneous_dlt(ivy.array(td.A)), td.X, atol=1e-6)
    assert np.allclose(
        ivy_pg.solve_homogeneous_dlt(ivy.array(td.A[0])), td.X[0], atol=1e-6
    )
    ivy.unset_backend()
