# global
import os
import ivy
import argparse
import ivy.jax
import ivy_mech
import ivy_vision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ivy_demo_utils.open3d_utils import Visualizer
from ivy_demo_utils.ivy_scene.scene_utils import SimCam, BaseSimulator
from ivy_demo_utils.framework_utils import choose_random_framework, get_framework_from_str


class DummyCam:

    def __init__(self, inv_calib_mat, inv_ext_mat, name, f):
        self.inv_calib_mat = inv_calib_mat
        self._inv_ext_mat = inv_ext_mat
        self._name = name
        self._f = f

    def get_inv_ext_mat(self):
        return self._inv_ext_mat

    def cap(self):
        this_dir = os.path.dirname(os.path.realpath(__file__))
        return self._f.array(np.load(
            os.path.join(this_dir, 'ctvg_no_sim', '{}_depth.npy'.format(self._name))).astype(np.float32)),\
               self._f.array(np.load(
                   os.path.join(this_dir, 'ctvg_no_sim', '{}_rgb.npy'.format(self._name))).astype(np.float32))


class Simulator(BaseSimulator):

    def __init__(self, interactive, try_use_sim, f):
        super().__init__(interactive, try_use_sim, f)

        # initialize scene
        if self.with_pyrep:
            for i in range(6):
                [item.remove() for item in self._vision_sensor_rays[i]]
            self._spherical_vision_sensor.remove()
            self._default_camera.set_position(np.array([-2.3518, 4.3953, 2.8949]))
            self._default_camera.set_orientation(np.array([i*np.pi/180 for i in [112.90, 27.329, -10.978]]))
            inv_ext_mat = self._f.reshape(self._f.array(self._default_vision_sensor.get_matrix(), 'float32'), (3, 4))
            self.default_camera_ext_mat_homo = self._f.inv(ivy_mech.make_transformation_homogeneous(inv_ext_mat))

            # public objects
            self.cams = [SimCam(cam, f) for cam in self._vision_sensors]

            # wait for user input
            self._user_prompt('\nInitialized scene with 6 projective cameras in the centre.\n\n'
                              'You can click on the dummy object "vision_senors" in the left hand panel, '
                              'then select the box icon with four arrows in the top panel of the simulator, '
                              'and then drag the cameras around dynamically.\n'
                              'Starting to drag and then holding ctrl allows you to also drag the cameras up and down. \n\n'
                              'This demo enables you to capture images from the cameras 10 times, '
                              'and render the associated 10 voxel grids in an open3D visualizer.\n\n'
                              'Both visualizers can be translated and rotated by clicking either the left mouse button or the wheel, '
                              'and then dragging the mouse.\n'
                              'Scrolling the mouse wheel zooms the view in and out.\n\n'
                              'Press enter in the terminal to use method ivy_vision.coords_to_bounding_voxel_grid and show the first voxel grid '
                              'reconstruction of the scene, produced using the 6 depth and color images from the projective cameras.\n')
        else:

            cam_names = ['vs{}'.format(i) for i in range(6)]
            pp_offsets = f.array([item/2 - 0.5 for item in [128, 128]])
            persp_angles = f.array([90 * np.pi/180]*2)
            intrinsics = ivy_vision.persp_angles_and_pp_offsets_to_intrinsics_object(
                persp_angles, pp_offsets, [128, 128])
            inv_calib_mat = intrinsics.inv_calib_mats
            cam_positions = [f.array([0., 0., 1.5]) for _ in range(6)]
            cam_quaternions = [f.array([-0.5,  0.5,  0.5, -0.5]), f.array([0.707, 0, 0,  0.707]), f.array([1., 0., 0., 0.]),
                               f.array([0.5, 0.5, 0.5, 0.5]), f.array([0, 0.707, 0.707, 0]), f.array([0., 0., 0., 1.])]
            cam_quat_poses = [f.concatenate((pos, eul), -1) for pos, eul in zip(cam_positions, cam_quaternions)]
            cam_inv_ext_mats = [ivy_mech.quaternion_pose_to_mat_pose(qp) for qp in cam_quat_poses]
            self.cams = [DummyCam(inv_calib_mat, iem, n, f) for iem, n in zip(cam_inv_ext_mats, cam_names)]
            self.default_camera_ext_mat_homo = f.array(
                [[-0.872, -0.489,  0., 0.099],
                 [-0.169,  0.301, -0.938, 0.994],
                 [0.459, -0.818, -0.346, 5.677],
                 [0., 0., 0., 1.]])

            # message
            print('\nInitialized dummy scene with 6 projective cameras in the centre.'
                  '\nClose the visualization window to use method ivy_vision.coords_to_bounding_voxel_grid and show a voxel grid '
                  'reconstruction of the scene, produced using the 6 depth and color images from the projective cameras.\n')

            # plot scene before rotation
            if interactive:
                plt.imshow(mpimg.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                     'ctvg_no_sim', 'before_capture.png')))
                plt.show()


def main(interactive=True, try_use_sim=True, f=None):
    f = choose_random_framework() if f is None else f
    sim = Simulator(interactive, try_use_sim, f)
    vis = Visualizer(f.to_numpy(sim.default_camera_ext_mat_homo))
    xyzs = list()
    rgbs = list()
    iterations = 10 if sim.with_pyrep else 1
    for _ in range(iterations):
        for cam in sim.cams:
            depth, rgb = cam.cap()
            xyz = sim.depth_to_xyz(depth, cam.get_inv_ext_mat(), cam.inv_calib_mat, [128]*2)
            xyzs.append(xyz)
            rgbs.append(rgb)
        xyz = ivy.reshape(ivy.concatenate(xyzs, 1), (-1, 3))
        rgb = ivy.reshape(ivy.concatenate(rgbs, 1), (-1, 3))
        voxels = ivy_vision.coords_to_voxel_grid(xyz, [100] * 3, features=rgb)
        vis.show_voxel_grid(voxels, interactive,
                            cuboid_inv_ext_mats=[ivy_mech.make_transformation_homogeneous(
                                f.to_numpy(cam.get_inv_ext_mat())) for cam in sim.cams],
                            cuboid_dims=[np.array([0.045, 0.045, 0.112]) for _ in sim.cams])
        xyzs.clear()
        rgbs.clear()
    sim.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--non_interactive', action='store_true',
                        help='whether to run the demo in non-interactive mode.')
    parser.add_argument('--no_sim', action='store_true',
                        help='whether to run the demo without attempt to use the PyRep simulator.')
    parser.add_argument('--framework', type=str, default=None,
                        help='which framework to use. Chooses a random framework if unspecified.')
    parsed_args = parser.parse_args()
    framework = None if parsed_args.framework is None else get_framework_from_str(parsed_args.framework)
    main(not parsed_args.non_interactive, not parsed_args.no_sim, framework)
