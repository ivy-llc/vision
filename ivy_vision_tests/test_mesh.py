# global
import ivy
import numpy as np
import pytest

# local
from ivy_vision_tests.data import TestData
from ivy_vision import mesh as ivy_mesh


class MeshTestData(TestData):
    def __init__(self):
        super().__init__()

        # Image to Mesh #
        # --------------#

        self.pre_mesh_image = np.array(
            [[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]
        )
        self.mesh_image = np.array(
            [
                [
                    [[[1], [4], [2]], [[5], [4], [2]]],
                    [[[2], [5], [3]], [[6], [5], [3]]],
                ],
                [
                    [[[4], [7], [5]], [[8], [7], [5]]],
                    [[[5], [8], [6]], [[9], [8], [6]]],
                ],
            ]
        )

        # Triangle Rasterization #
        # -----------------------#

        self.mesh_triangle = np.array(
            [[0.0, 0.0, 1.0], [0.0, 2.0, 1.0], [2.0, 0.0, 1.0]]
        )
        self.rasterized_image = np.array(
            [
                [[True], [False], [False]],
                [[True], [True], [False]],
                [[True], [True], [True]],
            ]
        )

        # Triangle Mesh from Image #
        # -------------------------#

        self.coord_img = np.array(
            [
                [
                    [[0.5, 1.1, 2.3], [0.7, 1.2, 1.5], [0.9, 1.9, 2.0]],
                    [[0.6, 0.7, 0.3], [0.2, 1.3, 1.2], [0.7, 0.7, 0.4]],
                    [[1.4, 1.7, 0.1], [1.3, 1.6, 1.4], [0.8, 1.1, 1.3]],
                    [[1.0, 0.9, 0.1], [1.2, 1.7, 1.3], [0.6, 1.7, 1.1]],
                ]
            ]
        )
        self.coord_validity_img = np.array(
            [
                [
                    [[True], [False], [True]],
                    [[True], [True], [True]],
                    [[True], [False], [True]],
                    [[True], [True], [True]],
                ]
            ]
        )

        self.tri_mesh_4x3_indices = np.array(
            [
                [
                    [0.0, 1.0, 3.0],
                    [1.0, 2.0, 4.0],
                    [3.0, 4.0, 6.0],
                    [4.0, 5.0, 7.0],
                    [6.0, 7.0, 9.0],
                    [7.0, 8.0, 10.0],
                    [4.0, 1.0, 3.0],
                    [5.0, 2.0, 4.0],
                    [7.0, 4.0, 6.0],
                    [8.0, 5.0, 7.0],
                    [10.0, 7.0, 9.0],
                    [11.0, 8.0, 10.0],
                ]
            ]
        )

        self.tri_mesh_4x3_valid_indices = np.array(
            [[3.0, 4.0, 6.0], [5.0, 2.0, 4.0], [11.0, 8.0, 10.0]]
        )

        self.tri_mesh_4x3_vertices = np.reshape(self.coord_img, [1, -1, 3])


td = MeshTestData()


def test_rasterize_triangles(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_mesh.rasterize_triangles(ivy.array(td.mesh_triangle), [3, 3]),
        td.rasterized_image,
        atol=1e-3,
    )
    assert np.allclose(
        ivy_mesh.rasterize_triangles(
            ivy.expand_dims(ivy.array(td.mesh_triangle), axis=0),
            [3, 3],
            batch_shape=[1],
        )[0],
        td.rasterized_image,
        atol=1e-3,
    )
    ivy.previous_backend()


def test_create_trimesh_indices_for_image(device, fw):
    if fw == "mxnet":
        # mxnet matmul only support N-D*N-D array (N >= 3)
        pytest.skip()
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_mesh.create_trimesh_indices_for_image([1], [4, 3]),
        td.tri_mesh_4x3_indices,
        atol=1e-3,
    )
    ivy.previous_backend()


def test_coord_image_to_trimesh(device, fw):
    if fw == "mxnet":
        # mxnet matmul only support N-D*N-D array (N >= 3)
        pytest.skip()
    ivy.set_backend(fw)
    coord_img = ivy.array(td.coord_img.tolist())
    vertices, trimesh_indices = ivy_mesh.coord_image_to_trimesh(
        coord_img, batch_shape=[1], image_dims=[4, 3], device=device
    )
    assert np.allclose(vertices, td.tri_mesh_4x3_vertices, atol=1e-3)
    assert np.allclose(trimesh_indices, td.tri_mesh_4x3_indices, atol=1e-3)
    vertices, trimesh_indices = ivy_mesh.coord_image_to_trimesh(
        ivy.array(td.coord_img),
        ivy.array(td.coord_validity_img),
        batch_shape=[1],
        image_dims=[4, 3],
        device=device,
    )
    assert np.allclose(vertices, td.tri_mesh_4x3_vertices, atol=1e-3)
    assert np.allclose(trimesh_indices, td.tri_mesh_4x3_valid_indices, atol=1e-3)
    ivy.previous_backend()
