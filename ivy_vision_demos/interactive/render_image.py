# global
import os
import ivy
import argparse
import ivy_mech
import ivy_vision
import numpy as np
from ivy_demo_utils.ivy_scene.scene_utils import SimCam, BaseSimulator


class DummyCam:

    def __init__(self, calib_mat, inv_calib_mat, ext_mat, inv_ext_mat, name):
        self.calib_mat = calib_mat
        self.inv_calib_mat = inv_calib_mat
        self._ext_mat = ext_mat
        self._inv_ext_mat = inv_ext_mat
        self._name = name

    def get_ext_mat(self):
        return self._ext_mat

    def get_inv_ext_mat(self):
        return self._inv_ext_mat

    def cap(self):
        this_dir = os.path.dirname(os.path.realpath(__file__))
        return ivy.array(np.load(os.path.join(this_dir, 'ri_no_sim', '{}_depth.npy'.format(self._name)))),\
               ivy.array(np.load(os.path.join(this_dir, 'ri_no_sim', '{}_rgb.npy'.format(self._name))))


class Simulator(BaseSimulator):

    def __init__(self, interactive, try_use_sim):
        super().__init__(interactive, try_use_sim)

        # initialize scene
        if self.with_pyrep:
            self._spherical_vision_sensor.remove()
            for i in range(0, 3):
                [item.set_color([0.2, 0.2, 0.8]) for item in self._vision_sensor_rays[i]]
            for i in range(4, 6):
                self._vision_sensors[i].remove()
                self._vision_sensor_bodies[i].remove()
                [item.remove() for item in self._vision_sensor_rays[i]]
            self._plant.remove()
            self._dining_chair_0.remove()
            self._dining_chair_1.remove()
            self._dining_table.set_position(np.array([0.3, 0, 0.55]))
            self._dining_table.set_orientation(np.array([np.pi / 2, 0., 0.]))
            self._swivel_chair.set_position(np.array([0.33, 0.98, 0.46]))
            self._swivel_chair.set_orientation(np.array([0., 0., np.pi]))
            self._vision_sensor_0.set_perspective_angle(60)
            self._vision_sensor_0.set_resolution([1024, 1024])
            self._vision_sensor_body_0.set_position(np.array([1.35, -0.05, 1.95]))
            self._vision_sensor_body_0.set_orientation(np.array([-145.81*np.pi/180, -27.763*np.pi/180, 136.5*np.pi/180]))
            self._vision_sensor_1.set_perspective_angle(60)
            self._vision_sensor_1.set_resolution([1024, 1024])
            self._vision_sensor_body_1.set_position(np.array([1.65, -2.075, 0.875]))
            self._vision_sensor_body_1.set_orientation(np.array([-91.181*np.pi/180, -30.478*np.pi/180, -171.39*np.pi/180]))
            self._vision_sensor_2.set_perspective_angle(60)
            self._vision_sensor_2.set_resolution([1024, 1024])
            self._vision_sensor_body_2.set_position(np.array([-0.94, -1.71, 0.994]))
            self._vision_sensor_body_2.set_orientation(np.array([-116.22*np.pi/180, 39.028*np.pi/180, -138.88*np.pi/180]))

            self._vision_sensor_3.set_perspective_angle(60)
            self._vision_sensor_3.set_resolution([512, 512])
            self._vision_sensor_body_3.set_position(np.array([0.35, -2.05, 0.68]))
            self._vision_sensor_body_3.set_orientation(np.array([-90.248*np.pi/180, -1.2555*np.pi/180, -179.88*np.pi/180]))

            self._default_camera.set_position(np.array([2.4732, -3.2641, 2.9269]))
            self._default_camera.set_orientation(np.array([i*np.pi/180 for i in [-134.8, -33.52, 151.26]]))

            # public objects
            self.cams = [SimCam(cam) for cam in self._vision_sensors[0:3]]
            self.target_cam = SimCam(self._vision_sensor_3)

            # wait for user input
            self._user_prompt(
                '\nInitialized scene with 3 acquiring projective cameras (blue rays) and 1 target projective camera (green rays) '
                'facing the overturned table.\n\n'
                'The simulator visualizer can be translated and rotated by clicking either the left mouse button or the wheel, '
                'and then dragging the mouse.\n'
                'Scrolling the mouse wheel zooms the view in and out.\n\n'
                'You can click on the object "vision_senor_3" in the left hand panel, '
                'then select the box icon with four arrows in the top panel of the simulator, '
                'and then drag the target camera around dynamically.\n'
                'Starting to drag and then holding ctrl allows you to also drag the cameras up and down.\n'
                'Clicking the top icon with a box and two rotating arrows similarly allows rotation of the camera.\n\n'
                'This demo enables you to capture images from the cameras 10 times, '
                'and render the associated images in the target frame.\n\n'
                '\nPress enter to use method ivy_vision.render_pixel_coords and show the first renderings for the target frame, '
                'produced by projecting the depth and color values from the 3 acquiring frames.'
                '\nRenderings are shown both with and without the use of a depth buffer in the ivy_vision method.\n'
                'If the image window appears blank at first, maximize it to show the renderings.')

        else:

            cam_names = ['vs{}'.format(i) for i in range(3)] + ['tvs']
            pp_offsets = ivy.array([item/2 - 0.5 for item in [1024, 1024]])
            persp_angles = ivy.array([60 * np.pi/180]*2)
            intrinsics = ivy_vision.persp_angles_and_pp_offsets_to_intrinsics_object(
                persp_angles, pp_offsets, [1024, 1024])
            calib_mat = intrinsics.calib_mats
            inv_calib_mat = intrinsics.inv_calib_mats

            pp_offsets = ivy.array([item/2 - 0.5 for item in [512, 512]])
            persp_angles = ivy.array([60 * np.pi/180]*2)
            target_intrinsics = ivy_vision.persp_angles_and_pp_offsets_to_intrinsics_object(
                persp_angles, pp_offsets, [512, 512])
            target_calib_mat = target_intrinsics.calib_mats
            target_inv_calib_mat = target_intrinsics.inv_calib_mats

            cam_positions = [ivy.array([1.35, -0.05, 1.95]),
                             ivy.array([1.65, -2.075, 0.875]),
                             ivy.array([-0.94, -1.71, 0.994]),
                             ivy.array([0.35, -2.05, 0.68])]

            cam_quaternions = [ivy.array([-0.40934521,  0.83571182,  0.35003018, -0.10724328]),
                               ivy.array([-0.13167774,  0.7011009,  0.65917628, -0.23791856]),
                               ivy.array([0.44628197, 0.68734518, 0.56583211, 0.09068139]),
                               ivy.array([-0.00698829,  0.70860066,  0.70552395, -0.00850271])]

            cam_quat_poses = [ivy.concat((pos, eul), axis=-1) for pos, eul in zip(cam_positions, cam_quaternions)]
            cam_inv_ext_mats = [ivy_mech.quaternion_pose_to_mat_pose(qp) for qp in cam_quat_poses]
            cam_ext_mats = [ivy.inv(ivy_mech.make_transformation_homogeneous(iem))[..., 0:3, :]
                            for iem in cam_inv_ext_mats]
            self.cams = [DummyCam(calib_mat, inv_calib_mat, em, iem, n)
                         for em, iem, n in zip(cam_ext_mats, cam_inv_ext_mats[:-1], cam_names[:-1])]
            self.target_cam = DummyCam(target_calib_mat, target_inv_calib_mat, cam_ext_mats[-1], cam_inv_ext_mats[-1],
                                       cam_names[-1])

            # message
            print('\nInitialized dummy scene with 3 acquiring projective cameras and 1 target projective camera '
                  'facing the overturned table.'
                  '\nClose the visualization window to use method ivy_vision.render_pixel_coords and show renderings for the target frame, '
                  'produced by projecting the depth and color values from the 3 acquiring frames.'
                  '\nRenderings are shown both with and without the use of a depth buffer in the ivy_vision method.\n')

            # plot scene before rotation
            if interactive:
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg
                plt.imshow(mpimg.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                     'ri_no_sim', 'before_capture.png')))
                plt.show()


def main(interactive=True, try_use_sim=True, f=None, fw=None):
    fw = ivy.choose_random_backend() if fw is None else fw
    ivy.set_backend(fw)
    f = ivy.get_backend(backend=fw) if f is None else f

    with_mxnet = ivy.current_backend_str() == 'mxnet'
    if with_mxnet:
        print('\nMXnet does not support "sum" or "min" reductions for scatter_nd,\n'
              'instead it only supports non-deterministic replacement for duplicates.\n'
              'Depth buffer rendering (requies min) or fusion buffer (requies sum) are therefore unsupported.\n'
              'The rendering in this demo with MXNet backend exhibits non-deterministic jagged edges as a result.')
    sim = Simulator(interactive, try_use_sim)
    import matplotlib.pyplot as plt
    xyzs = list()
    rgbs = list()
    iterations = 10 if sim.with_pyrep else 1
    for _ in range(iterations):
        for cam in sim.cams:
            depth, rgb = cam.cap()
            xyz = sim.depth_to_xyz(depth, cam.get_inv_ext_mat(), cam.inv_calib_mat, [1024]*2)
            xyzs.append(xyz)
            rgbs.append(rgb)
        xyz = ivy.reshape(ivy.concat(xyzs, axis=1), (-1, 3))
        rgb = ivy.reshape(ivy.concat(rgbs, axis=1), (-1, 3))
        cam_coords = ivy_vision.world_to_cam_coords(ivy_mech.make_coordinates_homogeneous(ivy.expand_dims(xyz, axis=1)),
                                                    sim.target_cam.get_ext_mat())
        ds_pix_coords = ivy_vision.cam_to_ds_pixel_coords(cam_coords, sim.target_cam.calib_mat)
        depth = ds_pix_coords[..., -1]
        pix_coords = ds_pix_coords[..., 0, 0:2] / depth
        final_image_dims = [512]*2
        feat = ivy.concat((depth, rgb), axis=-1)
        rendered_img_no_db, _, _ = ivy_vision.quantize_to_image(
            pix_coords, final_image_dims, feat, ivy.zeros(final_image_dims + [4]), with_db=False)
        with_db = not with_mxnet
        rendered_img_with_db, _, _ = ivy_vision.quantize_to_image(
            pix_coords, final_image_dims, feat, ivy.zeros(final_image_dims + [4]), with_db=with_db)

        import cv2
        a_img = cv2.resize(ivy.to_numpy(rgbs[0]), (256, 256))
        a_img[0:50, 0:50] = np.zeros_like(a_img[0:50, 0:50])
        a_img[5:45, 5:45] = np.ones_like(a_img[5:45, 5:45])
        cv2.putText(a_img, 'a', (13, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.2, tuple([0] * 3), 2)

        b_img = cv2.resize(ivy.to_numpy(rgbs[1]), (256, 256))
        b_img[0:50, 0:50] = np.zeros_like(b_img[0:50, 0:50])
        b_img[5:45, 5:45] = np.ones_like(b_img[5:45, 5:45])
        cv2.putText(b_img, 'b', (13, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.2, tuple([0] * 3), 2)

        c_img = cv2.resize(ivy.to_numpy(rgbs[2]), (256, 256))
        c_img[0:50, 0:50] = np.zeros_like(c_img[0:50, 0:50])
        c_img[5:45, 5:45] = np.ones_like(c_img[5:45, 5:45])
        cv2.putText(c_img, 'c', (13, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.2, tuple([0] * 3), 2)

        target_img = cv2.resize(ivy.to_numpy(sim.target_cam.cap()[1]), (256, 256))
        target_img[0:50, 0:140] = np.zeros_like(target_img[0:50, 0:140])
        target_img[5:45, 5:135] = np.ones_like(target_img[5:45, 5:135])
        cv2.putText(target_img, 'target', (13, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.2, tuple([0] * 3), 2)

        msg = 'non-deterministic' if with_mxnet else 'no depth buffer'
        width = 360 if with_mxnet else 320
        no_db_img = np.copy(ivy.to_numpy(rendered_img_no_db[..., 3:]))
        no_db_img[0:50, 0:width+5] = np.zeros_like(no_db_img[0:50, 0:width+5])
        no_db_img[5:45, 5:width] = np.ones_like(no_db_img[5:45, 5:width])
        cv2.putText(no_db_img, msg, (13, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.2, tuple([0] * 3), 2)

        with_db_img = np.copy(ivy.to_numpy(rendered_img_with_db[..., 3:]))
        with_db_img[0:50, 0:350] = np.zeros_like(with_db_img[0:50, 0:350])
        with_db_img[5:45, 5:345] = np.ones_like(with_db_img[5:45, 5:345])
        cv2.putText(with_db_img, 'with depth buffer', (13, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.2, tuple([0] * 3), 2)

        raw_imgs = np.concatenate((np.concatenate((a_img, b_img), 1),
                                   np.concatenate((c_img, target_img), 1)), 0)
        to_concat = (raw_imgs, no_db_img) if with_mxnet else (raw_imgs, no_db_img, with_db_img)
        final_img = np.concatenate(to_concat, 1)

        if interactive:
            print('\nClose the image window when you are ready.\n')
            plt.imshow(final_img)
            plt.show()
        xyzs.clear()
        rgbs.clear()
    sim.close()
    ivy.unset_backend()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--non_interactive', action='store_true',
                        help='whether to run the demo in non-interactive mode.')
    parser.add_argument('--no_sim', action='store_true',
                        help='whether to run the demo without attempt to use the PyRep simulator.')
    parser.add_argument('--backend', type=str, default=None,
                        help='which backend to use. Chooses a random backend if unspecified.')
    parsed_args = parser.parse_args()
    fw = parsed_args.backend
    f = None if fw is None else ivy.get_backend(backend=fw)
    main(not parsed_args.non_interactive, not parsed_args.no_sim, f, fw)
