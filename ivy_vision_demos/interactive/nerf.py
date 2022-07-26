# global
import os
import ivy
import math
import shutil
import argparse
import ivy_vision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm


class Model(ivy.Module):

    def __init__(self, num_layers, layer_dim, embedding_length, device=None):
        self._num_layers = num_layers
        self._layer_dim = layer_dim
        self._embedding_length = embedding_length
        embedding_size = 3 + 3 * 2 * embedding_length
        self._fc_layers = [ivy.Linear(embedding_size, layer_dim, device=device)]
        self._fc_layers += [ivy.Linear(layer_dim + (embedding_size if i % 4 == 0 and i > 0 else 0), layer_dim,
                                       device=device)
                            for i in range(num_layers-2)]
        self._fc_layers.append(ivy.Linear(layer_dim, 4, device=device))
        super(Model, self).__init__(device)

    def _forward(self, x, feat=None, timestamps=None):
        embedding = ivy.fourier_encode(x/math.pi, max_freq=2*math.exp(math.log(2) * (self._embedding_length-1)),
                                       num_bands=self._embedding_length)
        embedding = ivy.reshape(embedding, list(x.shape[:-1]) + [-1])
        x = ivy.relu(self._fc_layers[0](embedding))
        for i in range(1, self._num_layers-1):
            x = ivy.relu(self._fc_layers[i](x))
            if i % 4 == 0 and i > 0:
                x = ivy.concat([x, embedding], -1)
        x = self._fc_layers[-1](x)
        rgb = ivy.sigmoid(x[..., 0:3])
        sigma_a = ivy.relu(x[..., -1])
        return rgb, sigma_a


class NerfDemo:

    def __init__(self, num_iters, samples_per_ray, num_layers, layer_dim, with_tensor_splitting, compile_flag,
                 interactive, device, f):

        # ivy
        f = choose_random_framework() if f is None else f
        ivy.set_framework(f)
        if device:
            ivy.set_default_device(device)
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
            inv_ext_mats[101, 0:3], self._intrinsics[0])

        # train config
        self._embed_length = 6
        self._lr = 5e-4
        self._samples_per_ray = samples_per_ray
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
        self._model = Model(num_layers, layer_dim, self._embed_length)

        # tensor splitting
        self._with_tensor_splitting = with_tensor_splitting
        if with_tensor_splitting:
            self._dev_manager = ivy.DevManager(devices=[ivy.default_device()])

        # compile
        if compile_flag:
            rays_o, rays_d = self._get_rays(self._cam_geoms[0])
            target = self._images[0]
            self._loss_fn = ivy.compile_fn(self._loss_fn, False,
                                           example_inputs=[self._model, rays_o, rays_d, target, self._model.v])

    # Private #
    # --------#

    def _get_rays(self, cam_geom):
        pix_coords = ivy_vision.create_uniform_pixel_coords_image(self._img_dims)
        rays_d = ivy_vision.pixel_coords_to_world_ray_vectors(
            cam_geom.inv_full_mats_homo[..., 0:3, :], pix_coords, cam_geom.extrinsics.cam_centers)
        rays_o = cam_geom.extrinsics.cam_centers[..., 0]
        return rays_o, rays_d

    def _render_implicit_features_and_depth_with_splitting(
            self, network_fn, rays_o, rays_d, near, far, samples_per_ray, timestamps=None, render_depth=True,
            render_feats=True, render_variance=False, inter_feat_fn=None, with_grads=True, v=None):

        if not self._with_tensor_splitting:
            return ivy_vision.render_implicit_features_and_depth(
                network_fn, rays_o, rays_d, near, far, samples_per_ray, timestamps, render_depth, render_feats,
                render_variance, inter_feat_fn, with_grads, v)

        # shapes
        batch_shape = list(rays_o.shape[:-1])
        flat_batch_size = int(np.prod(batch_shape))
        num_batch_dims = len(batch_shape)
        ray_batch_shape = list(rays_d.shape[num_batch_dims:-1])
        flat_ray_batch_size = int(np.prod(ray_batch_shape))

        # memory on device
        memory_on_dev = ivy.cache_fn(ivy.total_mem_on_dev)(ivy.device(rays_o))

        # chunk size
        max_chunk_size = int(round(memory_on_dev * 10000 / (flat_batch_size * samples_per_ray)))

        # flatten
        rays_d = ivy.reshape(rays_d, batch_shape + [flat_ray_batch_size, 3])
        near = ivy.reshape(near, batch_shape + [flat_ray_batch_size])
        far = ivy.reshape(far, batch_shape + [flat_ray_batch_size])

        # run split call
        rets = ivy.split_func_call(lambda rd, n, f: ivy_vision.render_implicit_features_and_depth(
            network_fn, rays_o, rd, n, f, samples_per_ray, timestamps, render_depth, render_feats, render_variance,
            inter_feat_fn, with_grads, v), [rays_d, near, far], 'concat', max_chunk_size=max_chunk_size,
                                   input_axes=[-2, -1, -1], output_axes=-2)

        # return un-flattened
        return [ivy.reshape(ret, batch_shape + ray_batch_shape + [-1]) for ret in rets]

    def _loss_fn(self, model, rays_o, rays_d, target, v=None):
        rgb, depth = self._render_implicit_features_and_depth_with_splitting(
            model, rays_o, rays_d, near=ivy.ones(self._img_dims) * 2,
            far=ivy.ones(self._img_dims) * 6, samples_per_ray=self._samples_per_ray, v=v)
        return ivy.reduce_mean((rgb - target) ** 2)[0]

    # Public #
    # -------#

    def train(self):
        optimizer = ivy.Adam(self._lr)

        if self._with_tensor_splitting and not self._dev_manager.tuned:
            print('tuning tensor splitting for device allocation, the first few iterations may be slow...')

        for i in range(self._num_iters + 1):

            img_i = np.random.randint(self._images.shape[0])
            target = self._images[img_i]
            cam_geom = self._cam_geoms[img_i]
            rays_o, rays_d = self._get_rays(cam_geom)

            loss, grads = ivy.execute_with_gradients(
                lambda v: self._loss_fn(self._model, rays_o, rays_d, target, v=v), self._model.v)
            self._model.v = optimizer.step(self._model.v, grads)

            if i % self._log_freq == 0 and self._log_freq != -1:
                print('step {}, loss {}'.format(i, ivy.to_numpy(loss).item()))

            if i % self._vis_freq == 0 and self._vis_freq != -1:

                # Render the holdout view for logging
                rays_o, rays_d = self._get_rays(self._test_cam_geom)
                rgb, depth = self._render_implicit_features_and_depth_with_splitting(
                    self._model, rays_o, rays_d, near=ivy.ones(self._img_dims) * 2,
                    far=ivy.ones(self._img_dims)*6, samples_per_ray=self._samples_per_ray)
                plt.imsave(os.path.join(self._vis_log_dir, 'img_{}.png'.format(str(i).zfill(3))), ivy.to_numpy(rgb))

            # tune the tensor splitting for keeping within device memory
            if self._with_tensor_splitting and not self._dev_manager.tuned:
                print('tuning step...')
                self._dev_manager.ds_tune_step()
                if self._dev_manager.tuned:
                    print('device splitting tuning complete!')

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
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ], 'float32')

        rot_theta = lambda th_: ivy.array([
            [np.cos(th_), 0, -np.sin(th_), 0],
            [0, 1, 0, 0],
            [np.sin(th_), 0, np.cos(th_), 0],
            [0, 0, 0, 1],
        ], 'float32')

        def pose_spherical(theta, phi, radius):
            c2w_ = trans_t(radius)
            c2w_ = rot_phi(phi / 180. * np.pi) @ c2w_
            c2w_ = rot_theta(theta / 180. * np.pi) @ c2w_
            c2w_ = ivy.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], 'float32') @ c2w_
            c2w_ = c2w_ @ ivy.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], 'float32')
            return c2w_

        frames = []
        for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
            c2w = ivy.to_dev(pose_spherical(th, -30., 4.))
            cam_geom = ivy_vision.inv_ext_mat_and_intrinsics_to_cam_geometry_object(
                c2w[0:3], self._intrinsics[0])
            rays_o, rays_d = self._get_rays(cam_geom)
            rgb, depth = self._render_implicit_features_and_depth_with_splitting(
                self._model, rays_o, rays_d, near=ivy.ones(self._img_dims) * 2,
                far=ivy.ones(self._img_dims) * 6, samples_per_ray=self._samples_per_ray)
            frames.append((255 * np.clip(ivy.to_numpy(rgb), 0, 1)).astype(np.uint8))
        import imageio
        vid_filename = 'nerf_video.mp4'
        imageio.mimwrite(vid_filename, frames, fps=30, quality=7)


def main(num_iters, samples_per_ray, num_layers, layer_dim, with_tensor_splitting, compile_flag, interactive=True,
         device=None, f=None, fw=None):

    nerf_demo = NerfDemo(num_iters, samples_per_ray, num_layers, layer_dim, with_tensor_splitting, compile_flag,
                         interactive, device, f, fw)
    nerf_demo.train()
    if interactive:
        nerf_demo.save_video()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of iterations to train for.')
    parser.add_argument('--samples_per_ray', type=int, default=64,
                        help='The number of samples to make for each ray.')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='The number of layers in the implicit network.')
    parser.add_argument('--layer_dim', type=int, default=256,
                        help='The dimension of each layer in the implicit network.')
    parser.add_argument('--no_tensor_splitting', action='store_true',
                        help='Turn off device splitting. In this case, all full sized tensors are sent to the device.')
    parser.add_argument('--compile', action='store_true',
                        help='Whether or not to compile the loss function.')
    parser.add_argument('--non_interactive', action='store_true',
                        help='whether to run the demo in non-interactive mode.')
    parser.add_argument('--device', type=str, default=None,
                        help='The device to run the demo with, as a string. Either gpu:idx or cpu.')
    parser.add_argument('--backend', type=str, default=None,
                        help='which backend to use. Chooses a random backend if unspecified.')
    parsed_args = parser.parse_args()
    fw = parsed_args.backend
    f = None if fw is None else ivy.get_backend(fw)
    main(parsed_args.iterations, parsed_args.samples_per_ray, parsed_args.num_layers, parsed_args.layer_dim,
         not parsed_args.no_tensor_splitting, parsed_args.compile, not parsed_args.non_interactive, parsed_args.device,
         f, fw)
