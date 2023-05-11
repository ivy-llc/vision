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
        self.simple_image = np.tile(np.arange(9).astype(float).reshape((1, 1, 3, 3, 1)),
                                    (self.batch_size, 1, 1, 1, 1))

        self.warp = np.tile(np.array([[[[[0.0, 0.0], [0.5, 0.5], [2.0, 0.0]],
                                        [[1.5, 0.5], [1.0, 1.0], [0.5, 1.5]],
                                        [[0.0, 2.0], [1.5, 1.5], [2.0, 2.0]]]]]), (self.batch_size, 1, 1, 1, 1))

        self.warped_simple_image = np.tile(np.array([[[[[0.], [2.], [2.]],
                                                       [[3.], [4.], [5.]],
                                                       [[6.], [6.], [8.]]]]]), (self.batch_size, 1, 1, 1, 1))


td = SingleViewGeometryTestData()


def test_create_uniform_pixel_coords_image(dev_str, fw):
    assert np.array_equal(
        ivy_svg.create_uniform_pixel_coords_image(td.image_dims, (td.batch_size, td.num_cameras)),
        td.uniform_pixel_coords)
    assert np.array_equal(ivy_svg.create_uniform_pixel_coords_image(td.image_dims, (td.num_cameras,)),
                          td.uniform_pixel_coords[0])
    ivy_svg.create_uniform_pixel_coords_image(td.image_dims, (td.num_cameras,), True)


def test_persp_angles_to_focal_lengths(dev_str, fw):
    assert np.allclose(ivy_svg.persp_angles_to_focal_lengths(td.persp_angles, td.image_dims, dev_str=dev_str),
                       td.focal_lengths, atol=1e-6)
    assert np.allclose(ivy_svg.persp_angles_to_focal_lengths(td.persp_angles[0], td.image_dims, dev_str=dev_str),
                       td.focal_lengths[0], atol=1e-6)


def test_focal_lengths_to_persp_angles(dev_str, fw):
    assert np.allclose(ivy_svg.focal_lengths_to_persp_angles(ivy.array(td.focal_lengths), td.image_dims, dev_str=dev_str),
                       td.persp_angles, atol=1e-6)
    assert np.allclose(ivy_svg.focal_lengths_to_persp_angles(ivy.array(td.focal_lengths[0]), td.image_dims, dev_str=dev_str),
                       td.persp_angles[0], atol=1e-6)


def test_focal_lengths_and_pp_offsets_to_calib_mats(dev_str, fw):
    assert np.allclose(ivy_svg.focal_lengths_and_pp_offsets_to_calib_mat(ivy.array(td.focal_lengths),
                            ivy.array(td.pp_offsets), dev_str=dev_str), td.calib_mats, atol=1e-6)
    assert np.allclose(ivy_svg.focal_lengths_and_pp_offsets_to_calib_mat(ivy.array(td.focal_lengths[0]),
                            ivy.array(td.pp_offsets[0]), dev_str=dev_str), td.calib_mats[0], atol=1e-6)


def test_rot_mats_and_cam_centers_to_ext_mats(dev_str, fw):
    assert np.allclose(ivy_svg.rot_mat_and_cam_center_to_ext_mat(ivy.array(td.Rs), ivy.array(td.C_hats)), td.ext_mats, atol=1e-6)
    assert np.allclose(ivy_svg.rot_mat_and_cam_center_to_ext_mat(ivy.array(td.Rs[0]), ivy.array(td.C_hats[0])),
                       td.ext_mats[0], atol=1e-6)


def test_depth_to_ds_pixel_coords(dev_str, fw):
    assert (np.allclose(ivy_svg.depth_to_ds_pixel_coords(ivy.array(td.depth_maps), ivy.array(td.uniform_pixel_coords)),
                        td.pixel_coords_to_scatter, atol=1e-4))
    assert (np.allclose(ivy_svg.depth_to_ds_pixel_coords(ivy.array(td.depth_maps)), td.pixel_coords_to_scatter, atol=1e-4))
    assert (np.allclose(ivy_svg.depth_to_ds_pixel_coords(ivy.array(td.depth_maps[0]), ivy.array(td.uniform_pixel_coords[0])),
                        td.pixel_coords_to_scatter[0], atol=1e-4))


def test_depth_to_radial_depth(dev_str, fw):
    assert (np.allclose(ivy_svg.depth_to_radial_depth(ivy.array(td.depth_maps), ivy.array(td.inv_calib_mats)),
                        td.radial_depth_maps, atol=1e-4))
    assert (np.allclose(ivy_svg.depth_to_radial_depth(ivy.array(td.depth_maps), ivy.array(td.inv_calib_mats)),
                        td.radial_depth_maps, atol=1e-4))
    assert (np.allclose(ivy_svg.depth_to_radial_depth(ivy.array(td.depth_maps[0]),
                             ivy.array(td.inv_calib_mats[0])), td.radial_depth_maps[0], atol=1e-4))


def test_ds_pixel_coords_to_radial_depth(dev_str, fw):
    assert (np.allclose(ivy_svg.ds_pixel_coords_to_radial_depth(ivy.array(td.pixel_coords_to_scatter), ivy.array(td.inv_calib_mats)),
                        td.radial_depth_maps, atol=1e-4))
    assert (np.allclose(ivy_svg.ds_pixel_coords_to_radial_depth(ivy.array(td.pixel_coords_to_scatter), ivy.array(td.inv_calib_mats)),
                        td.radial_depth_maps, atol=1e-4))
    assert (np.allclose(ivy_svg.ds_pixel_coords_to_radial_depth(ivy.array(td.pixel_coords_to_scatter[0]),
                             ivy.array(td.inv_calib_mats[0])), td.radial_depth_maps[0], atol=1e-4))


def test_cam_to_ds_pixel_coords(dev_str, fw):
    assert (
        np.allclose(ivy_svg.cam_to_ds_pixel_coords(ivy.array(td.cam_coords), ivy.array(td.calib_mats)), td.pixel_coords_to_scatter, atol=1e-4))
    assert (np.allclose(ivy_svg.cam_to_ds_pixel_coords(ivy.array(td.cam_coords[0]), ivy.array(td.calib_mats[0])),
                        td.pixel_coords_to_scatter[0], atol=1e-4))


def test_cam_coords_to_depth(dev_str, fw):
    assert (
        np.allclose(ivy_svg.cam_coords_to_depth(ivy.array(td.cam_coords), ivy.array(td.calib_mats)), td.depth_maps, atol=1e-4))
    assert (np.allclose(ivy_svg.cam_coords_to_depth(ivy.array(td.cam_coords[0]), ivy.array(td.calib_mats[0])),
                        td.depth_maps[0], atol=1e-4))


def test_ds_pixel_to_cam_coords(dev_str, fw):
    assert np.allclose(ivy_svg.ds_pixel_to_cam_coords(ivy.array(td.pixel_coords_to_scatter), ivy.array(td.inv_calib_mats), dev_str=dev_str),
                       td.cam_coords, atol=1e-6)
    assert np.allclose(ivy_svg.ds_pixel_to_cam_coords(ivy.array(td.pixel_coords_to_scatter[0]), ivy.array(td.inv_calib_mats[0]), dev_str=dev_str),
                       td.cam_coords[0], atol=1e-6)


def test_depth_to_cam_coords(dev_str, fw):
    assert np.allclose(ivy_svg.depth_to_cam_coords(ivy.array(td.depth_maps), ivy.array(td.inv_calib_mats), dev_str=dev_str),
                       td.cam_coords, atol=1e-6)
    assert np.allclose(ivy_svg.depth_to_cam_coords(ivy.array(td.depth_maps[0]), ivy.array(td.inv_calib_mats[0]), dev_str=dev_str),
                       td.cam_coords[0], atol=1e-6)


def test_world_to_cam_coords(dev_str, fw):
    assert np.allclose(ivy_svg.world_to_cam_coords(ivy.array(td.world_coords), ivy.array(td.ext_mats), dev_str=dev_str),
                       td.cam_coords, atol=1e-6)
    assert np.allclose(ivy_svg.world_to_cam_coords(ivy.array(td.world_coords[0]), ivy.array(td.ext_mats[0]), dev_str=dev_str),
                       td.cam_coords[0], atol=1e-6)


def test_cam_to_world_coords(dev_str, fw):
    assert np.allclose(ivy_svg.cam_to_world_coords(ivy.array(td.cam_coords), ivy.array(td.inv_ext_mats), dev_str=dev_str),
                       td.world_coords, atol=1e-6)
    assert np.allclose(ivy_svg.cam_to_world_coords(ivy.array(td.cam_coords[0]), ivy.array(td.inv_ext_mats[0]), dev_str=dev_str),
                       td.world_coords[0], atol=1e-6)


def test_world_to_ds_pixel_coords(dev_str, fw):
    assert np.allclose(ivy_svg.world_to_ds_pixel_coords(ivy.array(td.world_coords), ivy.array(td.full_mats)), td.pixel_coords_to_scatter,
                       atol=1e-4)
    assert np.allclose(ivy_svg.world_to_ds_pixel_coords(ivy.array(td.world_coords[0]), ivy.array(td.full_mats[0])),
                       td.pixel_coords_to_scatter[0], atol=1e-4)


def test_world_coords_to_depth(dev_str, fw):
    assert np.allclose(ivy_svg.world_coords_to_depth(ivy.array(td.world_coords), ivy.array(td.full_mats)), td.depth_maps,
                       atol=1e-4)
    assert np.allclose(ivy_svg.world_coords_to_depth(ivy.array(td.world_coords[0]), ivy.array(td.full_mats[0])),
                       td.depth_maps[0], atol=1e-4)


def test_ds_pixel_to_world_coords(dev_str, fw):
    # with 2D image dimensions
    assert np.allclose(ivy_svg.ds_pixel_to_world_coords(ivy.array(td.pixel_coords_to_scatter), ivy.array(td.inv_full_mats)),
                       td.world_coords, atol=1e-6)
    assert np.allclose(ivy_svg.ds_pixel_to_world_coords(ivy.array(td.pixel_coords_to_scatter[0]), ivy.array(td.inv_full_mats[0])),
                       td.world_coords[0], atol=1e-6)
    # with flat image dimensions
    batch_shape = list(td.inv_full_mats.shape[:-2])
    assert np.allclose(ivy_svg.ds_pixel_to_world_coords(ivy.reshape(ivy.array(td.pixel_coords_to_scatter), batch_shape + [-1, 3]), ivy.array(td.inv_full_mats)),
                       ivy.reshape(ivy.array(td.world_coords), batch_shape + [-1, 4]), atol=1e-6)


def test_depth_to_world_coords(dev_str, fw):
    assert np.allclose(ivy_svg.depth_to_world_coords(ivy.array(td.depth_maps), ivy.array(td.inv_full_mats)),
                       td.world_coords, atol=1e-6)
    assert np.allclose(ivy_svg.depth_to_world_coords(ivy.array(td.depth_maps[0]), ivy.array(td.inv_full_mats[0])),
                       td.world_coords[0], atol=1e-6)


def test_pixel_coords_to_world_rays(dev_str, fw):
    assert np.allclose(
        ivy_svg.pixel_coords_to_world_ray_vectors(ivy.array(td.inv_full_mats), ivy.array(td.pixel_coords_to_scatter)),
        td.world_rays, atol=1e-6)
    assert np.allclose(
        ivy_svg.pixel_coords_to_world_ray_vectors(ivy.array(td.inv_full_mats[0]), ivy.array(td.pixel_coords_to_scatter[0])),
        td.world_rays[0], atol=1e-6)


def test_sphere_coords_to_world_ray_vectors(dev_str, fw):
    assert np.allclose(
        ivy_svg.sphere_coords_to_world_ray_vectors(ivy.array(td.sphere_coords.data), ivy.array(td.inv_Rs)),
        td.world_rays, atol=1e-6)
    assert np.allclose(
        ivy_svg.sphere_coords_to_world_ray_vectors(ivy.array(td.sphere_coords[0]), ivy.array(td.inv_Rs[0])),
        td.world_rays[0], atol=1e-6)


def test_bilinearly_interpolate_image(dev_str, fw):
    assert np.allclose(ivy_svg.bilinearly_interpolate_image(td.world_coords,
                            td.uniform_pixel_coords[:, :, :, :, 0:2]), td.world_coords, atol=1e-5)
    assert np.allclose(ivy_svg.bilinearly_interpolate_image(td.world_coords[0],
                            td.uniform_pixel_coords[0, :, :, :, 0:2]), td.world_coords[0], atol=1e-5)
    assert np.allclose(ivy_svg.bilinearly_interpolate_image(td.simple_image, td.warp),
                       td.warped_simple_image, atol=1e-5)


def test_inv_ext_mat_to_camera_center(dev_str, fw):
    assert np.allclose(ivy_svg.inv_ext_mat_to_camera_center(td.inv_ext_mats), td.C_hats, atol=1e-6)
    assert np.allclose(ivy_svg.inv_ext_mat_to_camera_center(td.inv_ext_mats[0]), td.C_hats[0], atol=1e-6)


def test_calib_and_ext_to_full_mat(dev_str, fw):
    assert np.allclose(ivy_svg.calib_and_ext_to_full_mat(ivy.array(td.calib_mats), ivy.array(td.ext_mats)), td.full_mats, atol=1e-6)
    assert np.allclose(ivy_svg.calib_and_ext_to_full_mat(ivy.array(td.calib_mats[0]), ivy.array(td.ext_mats[0])), td.full_mats[0],
                       atol=1e-6)


def test_cam_to_sphere_coords(dev_str, fw):
    assert np.allclose(ivy_svg.cam_to_sphere_coords(ivy.array(td.cam_coords)), td.sphere_coords, atol=1e-4)
    assert np.allclose(ivy_svg.cam_to_sphere_coords(ivy.array(td.cam_coords[0])), td.sphere_coords[0], atol=1e-4)


def test_ds_pixel_to_sphere_coords(dev_str, fw):
    assert np.allclose(ivy_svg.ds_pixel_to_sphere_coords(ivy.array(td.pixel_coords_to_scatter), ivy.array(td.inv_calib_mats)),
                       td.sphere_coords, atol=1e-4)
    assert np.allclose(ivy_svg.ds_pixel_to_sphere_coords(ivy.array(td.pixel_coords_to_scatter[0]), ivy.array(td.inv_calib_mats[0])),
                       td.sphere_coords[0], atol=1e-4)


def test_angular_pixel_to_sphere_coords(dev_str, fw):
    assert np.allclose(ivy_svg.angular_pixel_to_sphere_coords(ivy.array(td.angular_pixel_coords),
                            td.pixels_per_degree), td.sphere_coords, atol=1e-3)
    assert np.allclose(ivy_svg.angular_pixel_to_sphere_coords(ivy.array(td.angular_pixel_coords[0]),
                            td.pixels_per_degree), td.sphere_coords[0], atol=1e-3)


def test_sphere_to_cam_coords(dev_str, fw):
    assert np.allclose(ivy_svg.sphere_to_cam_coords(ivy.array(td.sphere_coords.data), dev_str=dev_str), td.cam_coords, atol=1e-3)
    assert np.allclose(ivy_svg.sphere_to_cam_coords(ivy.array(td.sphere_coords[0]), dev_str=dev_str), td.cam_coords[0], atol=1e-3)


def test_sphere_to_ds_pixel_coords(dev_str, fw):
    assert np.allclose(ivy_svg.sphere_to_ds_pixel_coords(ivy.array(td.sphere_coords.data), ivy.array(td.calib_mats)),
                       td.pixel_coords_to_scatter, atol=1e-3)
    assert np.allclose(ivy_svg.sphere_to_ds_pixel_coords(ivy.array(td.sphere_coords[0]), ivy.array(td.calib_mats[0])),
                       td.pixel_coords_to_scatter[0], atol=1e-3)


def test_sphere_to_angular_pixel_coords(dev_str, fw):
    assert np.allclose(ivy_svg.sphere_to_angular_pixel_coords(ivy.array(td.sphere_coords.data), td.pixels_per_degree), td.angular_pixel_coords, atol=1e-3)
    assert np.allclose(ivy_svg.sphere_to_angular_pixel_coords(ivy.array(td.sphere_coords[0]), td.pixels_per_degree), td.angular_pixel_coords[0], atol=1e-3)
