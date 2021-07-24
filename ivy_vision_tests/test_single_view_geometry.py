# global
import ivy
import numpy as np

# local
from ivy_vision_tests.data import TestData
from ivy_vision import single_view_geometry as ivy_svg


class SingleViewGeometryTestData(TestData):

    def __init__(self):
        super().__init__()

        # bilinear sampling
        self.simple_image = np.tile(np.arange(9).astype(np.float).reshape((1, 1, 3, 3, 1)),
                                    (self.batch_size, 1, 1, 1, 1))

        self.warp = np.tile(np.array([[[[[0.0, 0.0], [0.5, 0.5], [2.0, 0.0]],
                                        [[1.5, 0.5], [1.0, 1.0], [0.5, 1.5]],
                                        [[0.0, 2.0], [1.5, 1.5], [2.0, 2.0]]]]]), (self.batch_size, 1, 1, 1, 1))

        self.warped_simple_image = np.tile(np.array([[[[[0.], [2.], [2.]],
                                                       [[3.], [4.], [5.]],
                                                       [[6.], [6.], [8.]]]]]), (self.batch_size, 1, 1, 1, 1))


td = SingleViewGeometryTestData()


def test_create_uniform_pixel_coords_image(dev_str, call):
    assert np.array_equal(
        call(ivy_svg.create_uniform_pixel_coords_image, td.image_dims, (td.batch_size, td.num_cameras)),
        td.uniform_pixel_coords)
    assert np.array_equal(call(ivy_svg.create_uniform_pixel_coords_image, td.image_dims, (td.num_cameras,)),
                          td.uniform_pixel_coords[0])
    call(ivy_svg.create_uniform_pixel_coords_image, td.image_dims, (td.num_cameras,), True)


def test_persp_angles_to_focal_lengths(dev_str, call):
    assert np.allclose(call(ivy_svg.persp_angles_to_focal_lengths, td.persp_angles, td.image_dims, dev_str='cpu'),
                       td.focal_lengths, atol=1e-6)
    assert np.allclose(call(ivy_svg.persp_angles_to_focal_lengths, td.persp_angles[0], td.image_dims, dev_str='cpu'),
                       td.focal_lengths[0], atol=1e-6)


def test_focal_lengths_to_persp_angles(dev_str, call):
    assert np.allclose(call(ivy_svg.focal_lengths_to_persp_angles, td.focal_lengths, td.image_dims, dev_str='cpu'),
                       td.persp_angles, atol=1e-6)
    assert np.allclose(call(ivy_svg.focal_lengths_to_persp_angles, td.focal_lengths[0], td.image_dims, dev_str='cpu'),
                       td.persp_angles[0], atol=1e-6)


def test_focal_lengths_and_pp_offsets_to_calib_mats(dev_str, call):
    assert np.allclose(call(ivy_svg.focal_lengths_and_pp_offsets_to_calib_mat, td.focal_lengths,
                            td.pp_offsets, dev_str='cpu'), td.calib_mats, atol=1e-6)
    assert np.allclose(call(ivy_svg.focal_lengths_and_pp_offsets_to_calib_mat, td.focal_lengths[0],
                            td.pp_offsets[0], dev_str='cpu'), td.calib_mats[0], atol=1e-6)


def test_rot_mats_and_cam_centers_to_ext_mats(dev_str, call):
    assert np.allclose(call(ivy_svg.rot_mat_and_cam_center_to_ext_mat, td.Rs, td.C_hats), td.ext_mats, atol=1e-6)
    assert np.allclose(call(ivy_svg.rot_mat_and_cam_center_to_ext_mat, td.Rs[0], td.C_hats[0]),
                       td.ext_mats[0], atol=1e-6)


def test_depth_to_ds_pixel_coords(dev_str, call):
    assert (np.allclose(call(ivy_svg.depth_to_ds_pixel_coords, td.depth_maps, td.uniform_pixel_coords),
                        td.pixel_coords_to_scatter, atol=1e-4))
    assert (np.allclose(call(ivy_svg.depth_to_ds_pixel_coords, td.depth_maps), td.pixel_coords_to_scatter, atol=1e-4))
    assert (np.allclose(call(ivy_svg.depth_to_ds_pixel_coords, td.depth_maps[0], td.uniform_pixel_coords[0]),
                        td.pixel_coords_to_scatter[0], atol=1e-4))


def test_depth_to_radial_depth(dev_str, call):
    assert (np.allclose(call(ivy_svg.depth_to_radial_depth, td.depth_maps, td.inv_calib_mats),
                        td.radial_depth_maps, atol=1e-4))
    assert (np.allclose(call(ivy_svg.depth_to_radial_depth, td.depth_maps, td.inv_calib_mats),
                        td.radial_depth_maps, atol=1e-4))
    assert (np.allclose(call(ivy_svg.depth_to_radial_depth, td.depth_maps[0],
                             td.inv_calib_mats[0]), td.radial_depth_maps[0], atol=1e-4))


def test_ds_pixel_coords_to_radial_depth(dev_str, call):
    assert (np.allclose(call(ivy_svg.ds_pixel_coords_to_radial_depth, td.pixel_coords_to_scatter, td.inv_calib_mats),
                        td.radial_depth_maps, atol=1e-4))
    assert (np.allclose(call(ivy_svg.ds_pixel_coords_to_radial_depth, td.pixel_coords_to_scatter, td.inv_calib_mats),
                        td.radial_depth_maps, atol=1e-4))
    assert (np.allclose(call(ivy_svg.ds_pixel_coords_to_radial_depth, td.pixel_coords_to_scatter[0],
                             td.inv_calib_mats[0]), td.radial_depth_maps[0], atol=1e-4))


def test_cam_to_ds_pixel_coords(dev_str, call):
    assert (
        np.allclose(call(ivy_svg.cam_to_ds_pixel_coords, td.cam_coords, td.calib_mats), td.pixel_coords_to_scatter, atol=1e-4))
    assert (np.allclose(call(ivy_svg.cam_to_ds_pixel_coords, td.cam_coords[0], td.calib_mats[0]),
                        td.pixel_coords_to_scatter[0], atol=1e-4))


def test_ds_pixel_to_cam_coords(dev_str, call):
    assert np.allclose(call(ivy_svg.ds_pixel_to_cam_coords, td.pixel_coords_to_scatter, td.inv_calib_mats, dev_str='cpu'),
                       td.cam_coords, atol=1e-6)
    assert np.allclose(call(ivy_svg.ds_pixel_to_cam_coords, td.pixel_coords_to_scatter[0], td.inv_calib_mats[0], dev_str='cpu'),
                       td.cam_coords[0], atol=1e-6)


def test_world_to_cam_coords(dev_str, call):
    assert np.allclose(call(ivy_svg.world_to_cam_coords, td.world_coords, td.ext_mats, dev_str='cpu'),
                       td.cam_coords, atol=1e-6)
    assert np.allclose(call(ivy_svg.world_to_cam_coords, td.world_coords[0], td.ext_mats[0], dev_str='cpu'),
                       td.cam_coords[0], atol=1e-6)


def test_cam_to_world_coords(dev_str, call):
    assert np.allclose(call(ivy_svg.cam_to_world_coords, td.cam_coords, td.inv_ext_mats, dev_str='cpu'),
                       td.world_coords, atol=1e-6)
    assert np.allclose(call(ivy_svg.cam_to_world_coords, td.cam_coords[0], td.inv_ext_mats[0], dev_str='cpu'),
                       td.world_coords[0], atol=1e-6)


def test_world_to_ds_pixel_coords(dev_str, call):
    assert np.allclose(call(ivy_svg.world_to_ds_pixel_coords, td.world_coords, td.full_mats), td.pixel_coords_to_scatter,
                       atol=1e-4)
    assert np.allclose(call(ivy_svg.world_to_ds_pixel_coords, td.world_coords[0], td.full_mats[0]),
                       td.pixel_coords_to_scatter[0], atol=1e-4)


def test_ds_pixel_to_world_coords(dev_str, call):
    # with 2D image dimensions
    assert np.allclose(call(ivy_svg.ds_pixel_to_world_coords, td.pixel_coords_to_scatter, td.inv_full_mats),
                       td.world_coords, atol=1e-6)
    assert np.allclose(call(ivy_svg.ds_pixel_to_world_coords, td.pixel_coords_to_scatter[0], td.inv_full_mats[0]),
                       td.world_coords[0], atol=1e-6)
    # with flat image dimensions
    batch_shape = list(td.inv_full_mats.shape[:-2])
    assert np.allclose(call(ivy_svg.ds_pixel_to_world_coords,
                            np.reshape(td.pixel_coords_to_scatter, batch_shape + [-1, 3]), td.inv_full_mats),
                       np.reshape(td.world_coords, batch_shape + [-1, 4]), atol=1e-6)


def test_ds_pixel_coords_to_world_rays(dev_str, call):
    assert np.allclose(
        call(ivy_svg.ds_pixel_coords_to_world_ray_vectors, td.pixel_coords_to_scatter, td.inv_full_mats),
        td.world_rays, atol=1e-6)
    assert np.allclose(
        call(ivy_svg.ds_pixel_coords_to_world_ray_vectors, td.pixel_coords_to_scatter[0], td.inv_full_mats[0]),
        td.world_rays[0], atol=1e-6)


def test_sphere_coords_to_world_ray_vectors(dev_str, call):
    assert np.allclose(
        call(ivy_svg.sphere_coords_to_world_ray_vectors, td.sphere_coords, td.inv_Rs),
        td.world_rays, atol=1e-6)
    assert np.allclose(
        call(ivy_svg.sphere_coords_to_world_ray_vectors, td.sphere_coords[0], td.inv_Rs[0]),
        td.world_rays[0], atol=1e-6)


def test_bilinearly_interpolate_image(dev_str, call):
    assert np.allclose(call(ivy_svg.bilinearly_interpolate_image, td.world_coords,
                            td.uniform_pixel_coords[:, :, :, :, 0:2]), td.world_coords, atol=1e-5)
    assert np.allclose(call(ivy_svg.bilinearly_interpolate_image, td.world_coords[0],
                            td.uniform_pixel_coords[0, :, :, :, 0:2]), td.world_coords[0], atol=1e-5)
    assert np.allclose(call(ivy_svg.bilinearly_interpolate_image, td.simple_image, td.warp),
                       td.warped_simple_image, atol=1e-5)


def test_inv_ext_mat_to_camera_center(dev_str, call):
    assert np.allclose(call(ivy_svg.inv_ext_mat_to_camera_center, td.inv_ext_mats), td.C_hats, atol=1e-6)
    assert np.allclose(call(ivy_svg.inv_ext_mat_to_camera_center, td.inv_ext_mats[0]), td.C_hats[0], atol=1e-6)


def test_calib_and_ext_to_full_mat(dev_str, call):
    assert np.allclose(call(ivy_svg.calib_and_ext_to_full_mat, td.calib_mats, td.ext_mats), td.full_mats, atol=1e-6)
    assert np.allclose(call(ivy_svg.calib_and_ext_to_full_mat, td.calib_mats[0], td.ext_mats[0]), td.full_mats[0],
                       atol=1e-6)


def test_cam_to_sphere_coords(dev_str, call):
    assert np.allclose(call(ivy_svg.cam_to_sphere_coords, td.cam_coords), td.sphere_coords, atol=1e-4)
    assert np.allclose(call(ivy_svg.cam_to_sphere_coords, td.cam_coords[0]), td.sphere_coords[0], atol=1e-4)


def test_ds_pixel_to_sphere_coords(dev_str, call):
    assert np.allclose(call(ivy_svg.ds_pixel_to_sphere_coords, td.pixel_coords_to_scatter, td.inv_calib_mats),
                       td.sphere_coords, atol=1e-4)
    assert np.allclose(call(ivy_svg.ds_pixel_to_sphere_coords, td.pixel_coords_to_scatter[0], td.inv_calib_mats[0]),
                       td.sphere_coords[0], atol=1e-4)


def test_angular_pixel_to_sphere_coords(dev_str, call):
    assert np.allclose(call(ivy_svg.angular_pixel_to_sphere_coords, td.angular_pixel_coords,
                            td.pixels_per_degree), td.sphere_coords, atol=1e-3)
    assert np.allclose(call(ivy_svg.angular_pixel_to_sphere_coords, td.angular_pixel_coords[0],
                            td.pixels_per_degree), td.sphere_coords[0], atol=1e-3)


def test_sphere_to_cam_coords(dev_str, call):
    assert np.allclose(call(ivy_svg.sphere_to_cam_coords, td.sphere_coords, dev_str='cpu'), td.cam_coords, atol=1e-3)
    assert np.allclose(call(ivy_svg.sphere_to_cam_coords, td.sphere_coords[0], dev_str='cpu'),
                       td.cam_coords[0], atol=1e-3)


def test_sphere_to_ds_pixel_coords(dev_str, call):
    assert np.allclose(call(ivy_svg.sphere_to_ds_pixel_coords, td.sphere_coords, td.calib_mats),
                       td.pixel_coords_to_scatter, atol=1e-3)
    assert np.allclose(call(ivy_svg.sphere_to_ds_pixel_coords, td.sphere_coords[0], td.calib_mats[0]),
                       td.pixel_coords_to_scatter[0], atol=1e-3)


def test_sphere_to_angular_pixel_coords(dev_str, call):
    assert np.allclose(call(ivy_svg.sphere_to_angular_pixel_coords, td.sphere_coords,
                            td.pixels_per_degree), td.angular_pixel_coords, atol=1e-3)
    assert np.allclose(call(ivy_svg.sphere_to_angular_pixel_coords, td.sphere_coords[0],
                            td.pixels_per_degree), td.angular_pixel_coords[0], atol=1e-3)
