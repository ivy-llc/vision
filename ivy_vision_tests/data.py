# global
import os
import cv2
import ivy_mech
import ivy.numpy
import numpy as np
import xml.etree.ElementTree as ETree

MIN_DENOMINATOR = 1e-12


def str_list_to_list(str_list):
    return [float(item) for item in str_list[1:-1].split(',')]


class TestData:

    def __init__(self):
        self.batch_size = 1
        self.image_dims = [480, 640]
        self.num_cameras = 2

        # load camera data

        calib_mats_list = list()
        vrep_mats_list = list()
        depth_maps_list = list()

        state_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'camera_data.xml')
        with open(state_filepath) as xml_file:
            data_string = xml_file.read()
            root = ETree.fromstring(data_string)

            for i in range(self.num_cameras):
                camera_data = root.find('camera' + str(i + 1))

                calib_mat_element = camera_data.find('row_major_calib_mat')
                calib_mat = np.array(str_list_to_list(calib_mat_element.text)).reshape(3, 3)
                calib_mats_list.append(calib_mat)

                vrep_mat_element = camera_data.find('row_major_inv_ext_mat')
                vrep_mat = np.array(str_list_to_list(vrep_mat_element.text)).reshape(3, 4)
                vrep_mats_list.append(vrep_mat)

                depth_image = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                                      'depth_image_' + str(i + 1) + '.png'), -1)
                depth_buffer_bytes = depth_image.reshape(-1).tobytes()
                depth_buffer_flat = np.frombuffer(depth_buffer_bytes, np.float32)
                depth_map = depth_buffer_flat.reshape((self.image_dims[0], self.image_dims[1]))
                depth_maps_list.append(depth_map)

        # intrinsic mats

        self.calib_mats = np.tile(np.concatenate([np.expand_dims(item, 0) for item in calib_mats_list], 0),
                                  (self.batch_size, 1, 1, 1))
        self.inv_calib_mats = np.linalg.inv(self.calib_mats)

        # intrinsic data

        self.focal_lengths = np.concatenate((self.calib_mats[:, :, 0, 0:1], self.calib_mats[:, :, 1, 1:2]), -1)
        self.persp_angles = 2 * np.arctan(np.flip(np.array(self.image_dims), -1) / (2 * -self.focal_lengths))
        self.pp_offsets = np.concatenate((self.calib_mats[:, :, 0, 2:3], self.calib_mats[:, :, 1, 2:3]), -1)

        # camera centres

        self.C_hats = np.tile(np.concatenate([np.expand_dims(item[:, -1:], 0) for item in vrep_mats_list], 0),
                              (self.batch_size, 1, 1, 1))

        # camera rotation matrix wrt world frame

        self.inv_Rs = np.tile(np.concatenate([np.expand_dims(item[:, :-1], 0) for item in vrep_mats_list], 0),
                              (self.batch_size, 1, 1, 1))
        self.Rs = np.linalg.inv(self.inv_Rs)

        # extrinsic mats
        self.R_C_hats = np.matmul(self.Rs, self.C_hats)
        self.ext_mats = np.concatenate((self.Rs, -self.R_C_hats), -1)
        self.ext_mats_homo = np.concatenate((self.ext_mats, np.tile(np.array([0, 0, 0, 1]),
                                                                    (self.batch_size, self.num_cameras, 1, 1))), 2)

        # inv extrinsic mats
        self.inv_ext_mats_homo = np.linalg.inv(self.ext_mats_homo)
        self.inv_ext_mats = self.inv_ext_mats_homo[:, :, 0:3]

        self.pinv_ext_mats = np.linalg.pinv(self.ext_mats)

        # full mats

        self.full_mats = np.matmul(self.calib_mats, self.ext_mats)
        self.full_mats_homo = np.concatenate((self.full_mats, np.tile(np.array([0, 0, 0, 1]),
                                                                      (self.batch_size, self.num_cameras, 1, 1))), 2)
        self.inv_full_mats_homo = np.linalg.inv(self.full_mats_homo)
        self.inv_full_mats = self.inv_full_mats_homo[:, :, 0:3]

        self.pinv_full_mats = np.linalg.pinv(self.full_mats)

        # cam2cam ext mats

        self.cam2cam_ext_mats_homo = np.matmul(np.flip(self.ext_mats_homo, 1), self.inv_ext_mats_homo)
        self.cam2cam_ext_mats = self.cam2cam_ext_mats_homo[:, :, 0:3]

        # cam2cam full mats

        self.cam2cam_full_mats_homo = np.matmul(np.flip(self.full_mats_homo, 1), self.inv_full_mats_homo)
        self.cam2cam_full_mats = self.cam2cam_full_mats_homo[:, :, 0:3]

        # uniform pixel coords
        pixel_x_coords = np.reshape(np.tile(np.arange(self.image_dims[1]), [self.image_dims[0]]),
                                    (self.image_dims[0], self.image_dims[1], 1)).astype(np.float32)
        pixel_y_coords_ = np.reshape(np.tile(np.arange(self.image_dims[0]), [self.image_dims[1]]),
                                     (self.image_dims[1], self.image_dims[0], 1)).astype(np.float32)
        pixel_y_coords = np.transpose(pixel_y_coords_, (1, 0, 2))
        ones = np.ones_like(pixel_x_coords)
        uniform_pixel_coords = np.tile(np.expand_dims(np.concatenate((pixel_x_coords, pixel_y_coords, ones), -1), 0),
                                       (self.batch_size, 1, 1, 1))
        self.uniform_pixel_coords = np.tile(np.expand_dims(uniform_pixel_coords, 1), (1, 2, 1, 1, 1))

        # depth maps
        self.depth_maps = np.tile(np.concatenate([item.reshape((1, 1, self.image_dims[0], self.image_dims[1], 1))
                                                  for item in depth_maps_list], 1), (self.batch_size, 1, 1, 1, 1))

        # pixel coords
        self.pixel_coords = self.uniform_pixel_coords * self.depth_maps
        self.pixel_coords_normed = self.pixel_coords / self.pixel_coords[:, :, :, :, -1:]

        # cam coords
        coords_reshaped = np.reshape(np.transpose(self.pixel_coords, (0, 1, 4, 2, 3)),
                                     (self.batch_size, self.num_cameras, 3, -1))
        transformed_coords_vector = np.matmul(self.inv_calib_mats, coords_reshaped)
        transformed_coords_vector_transposed = np.transpose(transformed_coords_vector, (0, 1, 3, 2))
        self.cam_coords_not_homo = np.reshape(transformed_coords_vector_transposed,
                                              (self.batch_size, self.num_cameras, self.image_dims[0],
                                               self.image_dims[1], 3))
        self.cam_coords = np.concatenate((self.cam_coords_not_homo, np.ones(
            (self.batch_size, self.num_cameras, self.image_dims[0], self.image_dims[1], 1))), -1)

        # sphere coords
        with ivy.numpy.use:
            self.sphere_coords = \
                np.reshape(ivy_mech.cartesian_to_polar_coords(
                    np.reshape(self.cam_coords_not_homo, (-1, 3))),
                    (self.batch_size, self.num_cameras, self.image_dims[0], self.image_dims[1], 3))

        # angular_pixel_coords
        self.sphere_img_dims = [90, 180]
        self.pixels_per_degree = 1

        sphere_angle_coords = self.sphere_coords[..., 0:2]
        sphere_radius_vals = self.sphere_coords[..., -1:]
        sphere_angle_coords_in_degs = sphere_angle_coords * 180 / np.pi
        sphere_x_coords = (sphere_angle_coords_in_degs[..., 0:1] + 180) * self.pixels_per_degree
        sphere_y_coords = sphere_angle_coords_in_degs[..., 1:2] * self.pixels_per_degree
        self.angular_pixel_coords = np.concatenate((sphere_x_coords, sphere_y_coords, sphere_radius_vals), -1)

        # world coords
        coords_reshaped = np.reshape(np.transpose(self.cam_coords, (0, 1, 4, 2, 3)),
                                     (self.batch_size, self.num_cameras, 4, -1))
        transformed_coords_vector = np.matmul(self.inv_ext_mats, coords_reshaped)
        transformed_coords_vector_transposed = np.transpose(transformed_coords_vector, (0, 1, 3, 2))
        self.world_coords_not_homo = np.reshape(transformed_coords_vector_transposed, (
            self.batch_size, self.num_cameras, self.image_dims[0], self.image_dims[1], 3))
        self.world_coords = np.concatenate((self.world_coords_not_homo, np.ones(
            (self.batch_size, self.num_cameras, self.image_dims[0], self.image_dims[1], 1))), -1)

        # world rays
        vectors = self.world_coords[:, :, :, :, 0:3] - np.reshape(self.C_hats,
                                                                  (self.batch_size, self.num_cameras, 1, 1, 3))
        self.world_rays = vectors / (np.sqrt(np.sum(np.square(vectors), -1, keepdims=True)) + MIN_DENOMINATOR)

        # projected world rays
        vectors = np.flip(self.world_coords[:, :, :, :, 0:3], 1) - np.reshape(self.C_hats, (
            self.batch_size, self.num_cameras, 1, 1, 3))
        self.proj_world_rays = vectors / (np.sqrt(np.sum(np.square(vectors), -1, keepdims=True)) + MIN_DENOMINATOR)

        # projected cam coords
        coords_reshaped = np.reshape(np.transpose(np.flip(self.world_coords, 1), (0, 1, 4, 2, 3)),
                                     (self.batch_size, self.num_cameras, 4, -1))
        transformed_coords_vector = np.matmul(self.ext_mats, coords_reshaped)
        transformed_coords_vector_transposed = np.transpose(transformed_coords_vector, (0, 1, 3, 2))
        proj_cam_coords_not_homo = np.reshape(transformed_coords_vector_transposed, (
            self.batch_size, self.num_cameras, self.image_dims[0], self.image_dims[1], 3))
        self.proj_cam_coords = np.concatenate((proj_cam_coords_not_homo, np.ones(
            (self.batch_size, self.num_cameras, self.image_dims[0], self.image_dims[1], 1))), -1)

        # projected sphere coords
        with ivy.numpy.use:
            self.proj_sphere_coords = \
                np.reshape(ivy_mech.cartesian_to_polar_coords(
                    np.reshape(self.proj_cam_coords[..., 0:3], (-1, 3))),
                    (self.batch_size, self.num_cameras, self.image_dims[0], self.image_dims[1], 3))

        # projected pixel coords
        self.proj_cam_coords_not_homo = self.proj_cam_coords[:, :, :, :, 0:3]
        coords_reshaped = np.reshape(np.transpose(self.proj_cam_coords_not_homo, (0, 1, 4, 2, 3)),
                                     (self.batch_size, self.num_cameras, 3, -1))
        transformed_coords_vector = np.matmul(self.calib_mats, coords_reshaped)
        transformed_coords_vector_transposed = np.transpose(transformed_coords_vector, (0, 1, 3, 2))
        self.proj_pixel_coords = np.reshape(transformed_coords_vector_transposed, (
            self.batch_size, self.num_cameras, self.image_dims[0], self.image_dims[1], 3))
        self.proj_pixel_coords_normed = self.proj_pixel_coords / self.proj_pixel_coords[:, :, :, :, -1:]

        # projected angular pixel coords
        sphere_radius_vals = self.proj_sphere_coords[..., -1:]
        sphere_angle_coords = self.proj_sphere_coords[..., 0:2]
        sphere_angle_coords_in_degs = sphere_angle_coords * 180 / np.pi
        sphere_x_coords = (sphere_angle_coords_in_degs[..., 0:1] + 180) * self.pixels_per_degree
        sphere_y_coords = sphere_angle_coords_in_degs[..., 1:2] * self.pixels_per_degree
        self.proj_angular_pixel_coords =\
            np.concatenate((sphere_x_coords, sphere_y_coords, sphere_radius_vals), -1)

        # pixel correspondences
        self.pixel_correspondences = np.concatenate((self.pixel_coords[:, 0:1], self.proj_pixel_coords_normed[:, 0:1]),
                                                    1)

        # optical flow
        self.optical_flow = self.proj_pixel_coords_normed[:, 1, :, :, 0:2] - \
                            self.pixel_coords_normed[:, 0, :, :, 0:2]
        self.reverse_optical_flow = self.proj_pixel_coords_normed[:, 0, :, :, 0:2] - \
                                    self.pixel_coords_normed[:, 1, :, :, 0:2]

        # velocity from flow
        self.delta_t = np.ones((1, 1)) * 0.05
