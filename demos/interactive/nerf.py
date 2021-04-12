# global
import os
import ivy
import shutil
import argparse
import ivy_vision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from ivy_demo_utils.framework_utils import choose_random_framework, get_framework_from_str


class Model(ivy.Module):

    def __init__(self, num_layers, layer_dim, embedding_length, dev_str='cpu'):
        self._num_layers = num_layers
        self._layer_dim = layer_dim
        self._embedding_length = embedding_length
        embedding_size = 3 + 3 * 2 * embedding_length
        self._fc_layers = [ivy.Linear(embedding_size, layer_dim)]
        self._fc_layers += [ivy.Linear(layer_dim + (embedding_size if i % 4 == 0 and i > 0 else 0), layer_dim)
                            for i in range(num_layers-2)]
        self._fc_layers.append(ivy.Linear(layer_dim, 4))
        super(Model, self).__init__(dev_str)

    def _forward(self, x, feat=None):
        embedding = ivy_vision.sinusoid_positional_encoding(x, self._embedding_length)
        x = ivy.relu(self._fc_layers[0](embedding))
        for i in range(1, self._num_layers-1):
            x = ivy.relu(self._fc_layers[i](x))
            if i % 4 == 0 and i > 0:
                x = ivy.concatenate([x, embedding], -1)
        x = self._fc_layers[-1](x)
        rgb = ivy.sigmoid(x[..., 0:3])
        sigma_a = ivy.nn.relu(x[..., -1])
        return rgb, sigma_a


class NerfDemo:

    def __init__(self, num_iters, compile_flag, interactive, f):

        # ivy
        f = choose_random_framework() if f is None else f
        ivy.set_framework(f)
        ivy.seed(0)

        # Load input images and poses
        this_dir = os.path.dirname(os.path.realpath(__file__))
        data = np.load(os.path.join(this_dir, 'nerf_data/tiny_nerf_data.npz'))
        images = ivy.array(data['images'], 'float32')
        inv_ext_mats = ivy.array(data['poses'], 'float32')

        # intrinsics
        focal_lengths = ivy.array(np.tile(np.reshape(data['focal'], (1, 1)), [100, 2]), 'float32')
        self._img_dims = images.shape[1:3]
        pp_offsets = ivy.tile(ivy.array([[dim/2 - 0.5 for dim in self._img_dims]]), [100, 1])

        # train data
        self._images = images[:100, ..., :3]
        self._intrinsics = ivy_vision.focal_lengths_and_pp_offsets_to_intrinsics_object(
            focal_lengths, pp_offsets, self._img_dims)
        self._cam_geoms = ivy_vision.inv_ext_mat_and_intrinsics_to_cam_geometry_object(
            inv_ext_mats[:100, 0:3], self._intrinsics)

        # test data
        self._test_img = images[101]
        self._test_cam_geom = ivy_vision.inv_ext_mat_and_intrinsics_to_cam_geometry_object(
            inv_ext_mats[101, 0:3], self._intrinsics.slice(0))

        # train config
        if compile_flag:
            self._loss_fn = ivy.compile_fn(self._loss_fn)
        self._embed_length = 6
        self._lr = 5e-4
        self._num_samples = 64
        self._num_iters = num_iters

        # log config
        self._interactive = interactive
        self._log_freq = 1
        self._vis_freq = 25 if self._interactive else -1
        self._vis_log_dir = 'nerf_renderings'
        if os.path.exists(self._vis_log_dir):
            shutil.rmtree(self._vis_log_dir)
        os.makedirs(self._vis_log_dir)

        # model
        self._model = Model(4, 256, self._embed_length)

    # Private #
    # --------#

    def _get_rays(self, cam_geom):
        pix_coords = ivy_vision.create_uniform_pixel_coords_image(self._img_dims)
        rays_d = ivy_vision.ds_pixel_coords_to_world_ray_vectors(
            pix_coords, cam_geom.inv_full_mats_homo[..., 0:3, :], cam_geom.extrinsics.cam_centers)
        rays_o = ivy.expand_dims(ivy.expand_dims(cam_geom.extrinsics.cam_centers[..., 0], 0), 0)
        return rays_o, rays_d

    def _loss_fn(self, model, rays_o, rays_d, target, v=None):
        rgb, depth = ivy_vision.render_implicit_features_and_depth(
            model, rays_o, rays_d, near=ivy.ones(self._img_dims) * 2, far=ivy.ones(self._img_dims) * 6,
            samples_per_ray=self._num_samples, v=v)
        return ivy.reduce_mean((rgb - target) ** 2)[0]

    # Public #
    # -------#

    def train(self):
        optimizer = ivy.Adam(self._lr)

        for i in range(self._num_iters + 1):

            img_i = np.random.randint(self._images.shape[0])
            target = self._images[img_i]
            cam_geom = self._cam_geoms.slice(img_i)
            rays_o, rays_d = self._get_rays(cam_geom)

            loss, grads = ivy.execute_with_gradients(
                lambda v: self._loss_fn(self._model, rays_o, rays_d, target, v=v), self._model.v)
            self._model.v = optimizer.step(self._model.v, grads)

            if i % self._log_freq == 0 and self._log_freq != -1:
                print('step {}, loss {}'.format(i, ivy.to_numpy(loss).item()))

            if i % self._vis_freq == 0 and self._vis_freq != -1:

                # Render the holdout view for logging
                rays_o, rays_d = self._get_rays(self._test_cam_geom)
                rgb, depth = ivy_vision.render_implicit_features_and_depth(
                    self._model, rays_o, rays_d, near=ivy.ones(self._img_dims) * 2, far=ivy.ones(self._img_dims)*6,
                    samples_per_ray=self._num_samples)
                plt.imsave(os.path.join(self._vis_log_dir, 'img_{}.png'.format(str(i).zfill(3))), ivy.to_numpy(rgb))

        print('Completed Training')

    def save_video(self):
        if not self._interactive:
            return
        trans_t = lambda t: ivy.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ], 'float32')

        rot_phi = lambda phi: ivy.array([
            [1, 0, 0, 0],
            [0, ivy.cos(phi), -ivy.sin(phi), 0],
            [0, ivy.sin(phi), ivy.cos(phi), 0],
            [0, 0, 0, 1],
        ], 'float32')

        rot_theta = lambda th_: ivy.array([
            [ivy.cos(th_), 0, -ivy.sin(th_), 0],
            [0, 1, 0, 0],
            [ivy.sin(th_), 0, ivy.cos(th_), 0],
            [0, 0, 0, 1],
        ], 'float32')

        def pose_spherical(theta, phi, radius):
            c2w_ = trans_t(radius)
            c2w_ = rot_phi(phi / 180. * np.pi) @ c2w_
            c2w_ = rot_theta(theta / 180. * np.pi) @ c2w_
            c2w_ = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w_
            c2w_ = c2w_ @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            return c2w_

        frames = []
        for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
            c2w = pose_spherical(th, -30., 4.)
            # rot_mat1 = ivy_mech.rot_vec_to_rot_mat(ivy.array([0., np.pi, 0.]))
            # rot_mat2 = ivy_mech.rot_vec_to_rot_mat(ivy.array([0., 0., np.pi]))
            # rot_mat = ivy.matmul(rot_mat2, rot_mat1)
            # rot_mat_homo = ivy_mech.make_transformation_homogeneous(ivy.concatenate((rot_mat, ivy.zeros((3, 1))), -1))
            # c2w = ivy.matmul(c2w_orig, rot_mat_homo)
            cam_geom = ivy_vision.inv_ext_mat_and_intrinsics_to_cam_geometry_object(
                c2w[0:3], self._intrinsics.slice(0))
            rays_o, rays_d = self._get_rays(cam_geom)
            rgb, depth = ivy_vision.render_implicit_features_and_depth(
                self._model, rays_o, rays_d, near=ivy.ones(self._img_dims) * 2, far=ivy.ones(self._img_dims) * 6,
                samples_per_ray=self._num_samples)
            frames.append((255 * np.clip(rgb, 0, 1)).astype(np.uint8))
        import imageio
        vid_filename = 'nerf_video.mp4'
        imageio.mimwrite(vid_filename, frames, fps=30, quality=7)


def main(num_iters, compile_flag, interactive=True, f=None):
    nerf_demo = NerfDemo(num_iters, compile_flag, interactive, f)
    nerf_demo.train()
    nerf_demo.save_video()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of iterations to train for.')
    parser.add_argument('--compile', action='store_true',
                        help='Whether or not to compile the loss function.')
    parser.add_argument('--non_interactive', action='store_true',
                        help='whether to run the demo in non-interactive mode.')
    parser.add_argument('--framework', type=str, default=None,
                        help='which framework to use. Chooses a random framework if unspecified.')
    parsed_args = parser.parse_args()
    framework = None if parsed_args.framework is None else get_framework_from_str(parsed_args.framework)
    main(parsed_args.iterations, parsed_args.compile, not parsed_args.non_interactive, framework)
