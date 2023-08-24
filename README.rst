.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/logo.png?raw=true#gh-light-mode-only
   :width: 100%
   :class: only-light

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/logo_dark.png?raw=true#gh-dark-mode-only
   :width: 100%
   :class: only-dark

.. raw:: html

    <br/>
    <a href="https://pypi.org/project/ivy-vision/0.0.1.post0/">
        <img class="dark-light" style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://badge.fury.io/py/ivy-vision.svg">
    </a>
    <a href="https://github.com/unifyai/vision/actions?query=workflow%3Adocs">
        <img class="dark-light" style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://github.com/unifyai/vision/actions/workflows/docs.yml/badge.svg">
    </a>
    <a href="https://github.com/unifyai/vision/actions?query=workflow%3Anightly-tests">
        <img class="dark-light" style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://github.com/unifyai/vision/actions/workflows/nightly-tests.yml/badge.svg">
    </a>
    <a href="https://discord.gg/G4aR9Q7DTN">
        <img class="dark-light" style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/discord/799879767196958751?color=blue&label=%20&logo=discord&logoColor=white">
    </a>
    <br clear="all" />

**3D Vision functions with end-to-end support for machine learning developers, written in Ivy.**

.. raw:: html

    <div style="display: block;" align="center">
        <img class="dark-light" width="6%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
        <a href="https://jax.readthedocs.io">
            <img class="dark-light" width="13%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/jax_logo.png">
        </a>
        <img class="dark-light" width="12%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
        <a href="https://www.tensorflow.org">
            <img class="dark-light" width="13%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/tensorflow_logo.png">
        </a>
        <img class="dark-light" width="12%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
        <a href="https://pytorch.org">
            <img class="dark-light" width="13%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/pytorch_logo.png">
        </a>
        <img class="dark-light" width="12%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
        <a href="https://numpy.org">
            <img class="dark-light" width="13%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/numpy_logo.png">
        </a>
        <img class="dark-light" width="6%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
    </div>

Contents
--------

* `Overview`_
* `Run Through`_
* `Interactive Demos`_
* `Get Involved`_

Overview
--------

.. _docs: https://unify.ai/docs/vision/

**What is Ivy Vision?**

Ivy vision focuses predominantly on 3D vision, with functions for camera geometry, image projections,
co-ordinate frame transformations, forward warping, inverse warping, optical flow, depth triangulation, voxel grids,
point clouds, signed distance functions, and others.  Check out the docs_ for more info!

The library is built on top of the Ivy machine learning framework.
This means all functions simultaneously support:
Jax, Tensorflow, PyTorch, MXNet, and Numpy.

**Ivy Libraries**

There are a host of derived libraries written in Ivy, in the areas of mechanics, 3D vision, robotics, gym environments,
neural memory, pre-trained models + implementations, and builder tools with trainers, data loaders and more. Click on the icons below to learn more!

.. raw:: html

    <div style="display: block;">
        <a href="https://github.com/unifyai/mech">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_mech_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_mech.png">
            </picture>
        </a>
        <a href="https://github.com/unifyai/vision">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_vision_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_vision.png">
            </picture>
        </a>
        <a href="https://github.com/unifyai/robot">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_robot_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_robot.png">
            </picture>
        </a>
        <a href="https://github.com/unifyai/gym">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_gym_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_gym.png">
            </picture>
        </a>

        <br clear="all" />

        <a href="https://pypi.org/project/ivy-mech/0.0.1.post0/">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-mech.svg">
        </a>
        <a href="https://pypi.org/project/ivy-vision/0.0.1.post0/">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-vision.svg">
        </a>
        <a href="https://pypi.org/project/ivy-robot/0.0.1.post0/">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-robot.svg">
        </a>
        <a href="https://pypi.org/project/ivy-gym">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;"width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-gym.svg">
        </a>

        <br clear="all" />

        <a href="https://github.com/unifyai/mech/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;"src="https://github.com/unifyai/mech/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/unifyai/vision/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/vision/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/unifyai/robot/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/robot/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/unifyai/gym/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/gym/actions/workflows/nightly-tests.yml/badge.svg">
        </a>

        <br clear="all" />

        <a href="https://github.com/unifyai/memory">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_memory_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_memory.png">
            </picture>
        </a>
        <a href="https://github.com/unifyai/builder">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_builder_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_builder.png">
            </picture>
        </a>
        <a href="https://github.com/unifyai/models">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_models_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_models.png">
            </picture>
        </a>
        <a href="https://github.com/unifyai/ecosystem">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_ecosystem_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_ecosystem.png">
            </picture>
        </a>

        <br clear="all" />

        <a href="https://pypi.org/project/ivy-memory/0.0.1.post0/">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-memory.svg">
        </a>
        <a href="https://pypi.org/project/ivy-builder/0.0.1.post0/">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-builder.svg">
        </a>
        <a href="https://pypi.org/project/ivy-models">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-models.svg">
        </a>
        <a href="https://github.com/unifyai/ecosystem/actions?query=workflow%3Adocs">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/ecosystem/actions/workflows/docs.yml/badge.svg">
        </a>

        <br clear="all" />

        <a href="https://github.com/unifyai/memory/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/memory/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/unifyai/builder/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/builder/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/unifyai/models/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/models/actions/workflows/nightly-tests.yml/badge.svg">
        </a>

        <br clear="all" />

    </div>
    <br clear="all" />

**Quick Start**

Ivy vision can be installed like so: ``pip install ivy-vision==0.0.1.post0``

.. _demos: https://github.com/unifyai/vision/tree/main/ivy_vision_demos
.. _interactive: https://github.com/unifyai/vision/tree/main/ivy_vision_demos/interactive

To quickly see the different aspects of the library, we suggest you check out the demos_!
we suggest you start by running the script ``run_through.py``,
and read the "Run Through" section below which explains this script.

For more interactive demos, we suggest you run either
``coords_to_voxel_grid.py`` or ``render_image.py`` in the interactive_ demos folder.

Run Through
-----------

We run through some of the different parts of the library via a simple ongoing example script.
The full script is available in the demos_ folder, as file ``run_through.py``.
First, we select a random backend framework to use for the examples, from the options
``ivy.jax``, ``ivy.tensorflow``, ``ivy.torch``, ``ivy.mxnet`` or ``ivy.numpy``,
and use this to set the ivy backend framework.

.. code-block:: python

    import  ivy
    ivy.set_backend(ivy.choose_random_backend())

**Camera Geometry**

To get to grips with some of the basics, we next show how to construct ivy containers which represent camera geometry.
The camera intrinsic matrix, extrinsic matrix, full matrix, and all of their inverses are central to most of the
functions in this library.

All of these matrices are contained within the Ivy camera geometry class.

.. code-block:: python

    # intrinsics

    # common intrinsic params
    img_dims = [512, 512]
    pp_offsets = ivy.array([dim / 2 - 0.5 for dim in img_dims], 'float32')
    cam_persp_angles = ivy.array([60 * np.pi / 180] * 2, 'float32')

    # ivy cam intrinsics container
    intrinsics = ivy_vision.persp_angles_and_pp_offsets_to_intrinsics_object(
        cam_persp_angles, pp_offsets, img_dims)

    # extrinsics

    # 3 x 4
    cam1_inv_ext_mat = ivy.array(np.load(data_dir + '/cam1_inv_ext_mat.npy'), 'float32')
    cam2_inv_ext_mat = ivy.array(np.load(data_dir + '/cam2_inv_ext_mat.npy'), 'float32')

    # full geometry

    # ivy cam geometry container
    cam1_geom = ivy_vision.inv_ext_mat_and_intrinsics_to_cam_geometry_object(
        cam1_inv_ext_mat, intrinsics)
    cam2_geom = ivy_vision.inv_ext_mat_and_intrinsics_to_cam_geometry_object(
        cam2_inv_ext_mat, intrinsics)
    cam_geoms = [cam1_geom, cam2_geom]

The geometries used in this quick start demo are based upon the scene presented below.

.. image:: https://github.com/unifyai/vision/blob/main/docs/images/scene.png?raw=true
   :width: 100%

The code sample below demonstrates all of the attributes contained within the Ivy camera geometry class.

.. code-block:: python

    for cam_geom in cam_geoms:

        assert cam_geom.intrinsics.focal_lengths.shape == (2,)
        assert cam_geom.intrinsics.persp_angles.shape == (2,)
        assert cam_geom.intrinsics.pp_offsets.shape == (2,)
        assert cam_geom.intrinsics.calib_mats.shape == (3, 3)
        assert cam_geom.intrinsics.inv_calib_mats.shape == (3, 3)

        assert cam_geom.extrinsics.cam_centers.shape == (3, 1)
        assert cam_geom.extrinsics.Rs.shape == (3, 3)
        assert cam_geom.extrinsics.inv_Rs.shape == (3, 3)
        assert cam_geom.extrinsics.ext_mats_homo.shape == (4, 4)
        assert cam_geom.extrinsics.inv_ext_mats_homo.shape == (4, 4)

        assert cam_geom.full_mats_homo.shape == (4, 4)
        assert cam_geom.inv_full_mats_homo.shape == (4, 4)

**Load Images**

We next load the color and depth images corresponding to the two camera frames.
We also construct the depth-scaled homogeneous pixel co-ordinates for each image,
which is a central representation for the ivy_vision functions.
This representation simplifies projections between frames.

.. code-block:: python

    # load images

    # h x w x 3
    color1 = ivy.array(cv2.imread(data_dir + '/rgb1.png').astype(np.float32) / 255)
    color2 = ivy.array(cv2.imread(data_dir + '/rgb2.png').astype(np.float32) / 255)

    # h x w x 1
    depth1 = ivy.array(np.reshape(np.frombuffer(cv2.imread(
        data_dir + '/depth1.png', -1).tobytes(), np.float32), img_dims + [1]))
    depth2 = ivy.array(np.reshape(np.frombuffer(cv2.imread(
        data_dir + '/depth2.png', -1).tobytes(), np.float32), img_dims + [1]))

    # depth scaled pixel coords

    # h x w x 3
    u_pix_coords = ivy_vision.create_uniform_pixel_coords_image(img_dims)
    ds_pixel_coords1 = u_pix_coords * depth1
    ds_pixel_coords2 = u_pix_coords * depth2

The rgb and depth images are presented below.

.. image:: https://github.com/unifyai/vision/blob/main/docs/images/rgb_and_depth.png?raw=true
   :width: 100%

**Optical Flow and Depth Triangulation**

Now that we have two cameras, their geometries, and their images fully defined,
we can start to apply some of the more interesting vision functions.
We start with some optical flow and depth triangulation functions.

.. code-block:: python

    # required mat formats
    cam1to2_full_mat_homo = ivy.matmul(cam2_geom.full_mats_homo, cam1_geom.inv_full_mats_homo)
    cam1to2_full_mat = cam1to2_full_mat_homo[..., 0:3, :]
    full_mats_homo = ivy.concat((ivy.expand_dims(cam1_geom.full_mats_homo, axis=0),
                                      ivy.expand_dims(cam2_geom.full_mats_homo, axis=0)), axis=0)
    full_mats = full_mats_homo[..., 0:3, :]

    # flow
    flow1to2 = ivy_vision.flow_from_depth_and_cam_mats(ds_pixel_coords1, cam1to2_full_mat)

    # depth again
    depth1_from_flow = ivy_vision.depth_from_flow_and_cam_mats(flow1to2, full_mats)

Visualizations of these images are given below.

.. image:: https://github.com/unifyai/vision/blob/main/docs/images/flow_and_depth.png?raw=true
   :width: 100%

**Inverse and Forward Warping**

Most of the vision functions, including the flow and depth functions above,
make use of image projections,
whereby an image of depth-scaled homogeneous pixel-coordinates is transformed into
cartesian co-ordinates relative to the acquiring camera, the world, another camera,
or transformed directly to pixel co-ordinates in another camera frame.
These projections also allow warping of the color values from one camera to another.

For inverse warping, we assume depth to be known for the target frame.
We can then determine the pixel projections into the source frame,
and bilinearly interpolate these color values at the pixel projections,
to infer the color image in the target frame.

Treating frame 1 as our target frame,
we can use the previously calculated optical flow from frame 1 to 2, in order
to inverse warp the color data from frame 2 to frame 1, as shown below.


.. code-block:: python

    # inverse warp rendering
    warp = u_pix_coords[..., 0:2] + flow1to2
    color2_warp_to_f1 = ivy_vision.image.bilinear_resample(color2, warp)

    # projected depth scaled pixel coords 2
    ds_pixel_coords1_wrt_f2 = ivy_vision.ds_pixel_to_ds_pixel_coords(ds_pixel_coords1, cam1to2_full_mat)

    # projected depth 2
    depth1_wrt_f2 = ds_pixel_coords1_wrt_f2[..., -1:]

    # inverse warp depth
    depth2_warp_to_f1 = ivy_vision.image.bilinear_resample(depth2, warp)

    # depth validity
    depth_validity = ivy.abs(depth1_wrt_f2 - depth2_warp_to_f1) < 0.01

    # inverse warp rendering with mask
    color2_warp_to_f1_masked = ivy.where(depth_validity, color2_warp_to_f1, ivy.zeros_like(color2_warp_to_f1))

Again, visualizations of these images are given below.
The images represent intermediate steps for the inverse warping of color from frame 2 to frame 1,
which is shown in the bottom right corner.

.. image:: https://github.com/unifyai/vision/blob/main/docs/images/inverse_warped.png?raw=true
   :width: 100%

For forward warping, we instead assume depth to be known in the source frame.
A common approach is to construct a mesh, and then perform rasterization of the mesh.

The ivy method ``ivy_vision.render_pixel_coords`` instead takes a simpler approach,
by determining the pixel projections into the target frame,
quantizing these to integer pixel co-ordinates,
and scattering the corresponding color values directly into these integer pixel co-ordinates.

This process in general leads to holes and duplicates in the resultant image,
but when compared to inverse warping,
it has the beneft that the target frame does not need to correspond to a real camera with known depth.
Only the target camera geometry is required, which can be for any hypothetical camera.

We now consider the case of forward warping the color data from camera frame 2 to camera frame 1,
and again render the new color image in target frame 1.

.. code-block:: python

    # forward warp rendering
    ds_pixel_coords1_proj = ivy_vision.ds_pixel_to_ds_pixel_coords(
        ds_pixel_coords2, ivy.inv(cam1to2_full_mat_homo)[..., 0:3, :])
    depth1_proj = ds_pixel_coords1_proj[..., -1:]
    ds_pixel_coords1_proj = ds_pixel_coords1_proj[..., 0:2] / depth1_proj
    features_to_render = ivy.concat((depth1_proj, color2), axis=-1)

    # without depth buffer
    f1_forward_warp_no_db, _, _ = ivy_vision.quantize_to_image(
        ivy.reshape(ds_pixel_coords1_proj, (-1, 2)), img_dims, ivy.reshape(features_to_render, (-1, 4)),
        ivy.zeros_like(features_to_render), with_db=False)

    # with depth buffer
    f1_forward_warp_w_db, _, _ = ivy_vision.quantize_to_image(
        ivy.reshape(ds_pixel_coords1_proj, (-1, 2)), img_dims, ivy.reshape(features_to_render, (-1, 4)),
        ivy.zeros_like(features_to_render), with_db=False if ivy.get_framework() == 'mxnet' else True)

Again, visualizations of these images are given below.
The images show the forward warping of both depth and color from frame 2 to frame 1,
which are shown with and without depth buffers in the right-hand and central columns respectively.

.. image:: https://github.com/unifyai/vision/blob/main/docs/images/forward_warped.png?raw=true
   :width: 100%

Interactive Demos
-----------------

In addition to the examples above, we provide two further demo scripts,
which are more visual and interactive, and are each built around a particular function.

Rather than presenting the code here, we show visualizations of the demos.
The scripts for these demos can be found in the interactive_ demos folder.

**Neural Rendering**

The first demo uses method ``ivy_vision.render_implicit_features_and_depth``
to train a Neural Radiance Field (NeRF) model to encode a lego digger. The NeRF model can then be queried at new camera
poses to render new images from poses unseen during training.

.. raw:: html

    <p align="center">
        <img width="50%" style="display: block;" src='https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/ivy_vision/nerf_demo.gif?raw=true'>
    </p>

**Co-ordinates to Voxel Grid**

The second demo captures depth and color images from a set of cameras,
converts the depth to world-centric co-ordinartes,
and uses the method ``ivy_vision.coords_to_voxel_grid`` to
voxelize the depth and color values into a grid, as shown below:

.. raw:: html

    <p align="center">
        <img width="75%" style="display: block;" src='https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/ivy_vision/voxel_grid_demo.gif?raw=true'>
    </p>

**Point Rendering**

The final demo again captures depth and color images from a set of cameras,
but this time uses the method ``ivy_vision.quantize_to_image`` to
dynamically forward warp and point render the images into a new target frame, as shown below.
The acquiring cameras all remain static, while the target frame for point rendering moves freely.

.. raw:: html

    <p align="center">
        <img width="75%" style="display: block;" src='https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/ivy_vision/point_render_demo.gif?raw=true'>
    </p>

Get Involved
------------

We hope the functions in this library are useful to a wide range of machine learning developers.
However, there are many more areas of 3D vision which could be covered by this library.

If there are any particular vision functions you feel are missing,
and your needs are not met by the functions currently on offer,
then we are very happy to accept pull requests!

We look forward to working with the community on expanding and improving the Ivy vision library.

Citation
--------

::

    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
