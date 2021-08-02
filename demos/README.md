# Ivy Vision Demos

We provide a simple set of interactive demos for the Ivy Vision library.
Running these demos is quick and simple.

## Install

First, clone this repo:

```bash
git clone https://github.com/ivy-dl/vision.git ~/ivy_vision
```

The interactive demos optionally make use of the simulator
[CoppeliaSim](https://www.coppeliarobotics.com/),
and the python wrapper [PyRep](https://github.com/stepjam/PyRep).

If these are not installed, the demos will all still run, but will display pre-rendered images from the simultator.

### Local

For a local installation, first install the dependencies:

```bash
cd ~/ivy_vision
python3 -m pip install -r requirements.txt
cd ~/ivy_vision/demos
python3 -m pip install -r requirements.txt
```

To run interactive demos inside a simulator, CoppeliaSim and PyRep should then be installed following the installation [intructions](https://github.com/stepjam/PyRep#install).

### Docker

For a docker installation, first ensure [docker](https://docs.docker.com/get-docker/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) are installed.

Then simply pull the ivy vision image:

```bash
docker pull ivydl/ivy-vision:latest
```

## Demos

All demos can be run by executing the python scripts directly.
If a demo script is run without command line arguments, then a random backend framework will be selected from those installed.
Alternatively, the `--framework` argument can be used to manually specify a framework from the options
`jax`, `tensorflow`, `torch`, `mxnd` or `numpy`.

The examples below assume a docker installation, but the demo scripts can also
be run with python directly for local installations.

### Run Through

For a basic run through the library:

```bash
cd ~/ivy_vision/demos
./run_demo.sh run_through
```

This script, and the various parts of the library, are further discussed in the [Run Through](https://github.com/ivy-dl/vision#run-through) section of the main README.
We advise following along with this section for maximum effect. The demo script should also be opened locally,
and breakpoints added to step in at intermediate points to further explore.

To run the script using a specific backend, tensorflow for example, then run like so:

```bash
./run_demo.sh run_through --framework tensorflow
```

### Neural Rendering

The first demo uses methods ``ivy_vision.sinusoid_positional_encoding``, and ``ivy_vision.render_implicit_features_and_depth``
to train a Neural Radiance Field (NeRF) model to encode a lego digger. The NeRF model can then be queried at new camera
poses to render new images from poses unseen during training.

```bash
cd ~/ivy_vision/demos
./run_demo.sh interactive.nerf
```

At the end of the demo, a video nerf_video.mp4 is created of renderings from unseen camera poses around a sphere.

<p align="center">
    <img width="50%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_vision/nerf_demo.gif?raw=true'>
</p>

### Co-ordinates to Voxel Grid

In this demo, a goup of 6 projective cameras are dragged around the scene,
and a voxel grid reconstruction is dynamically generated from the 6 depth maps,
using the method ivy_vision.coords_to_voxel_grid.

```bash
cd ~/ivy_vision/demos
./run_demo.sh interactive.coords_to_voxel_grid
```

Example output from the simulator, and Open3D renderings, are given below:

<p align="center">
    <img width="75%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_vision/voxel_grid_demo.gif?raw=true'>
</p>

### Point Rendering

In this demo, a goup of 3 projective cameras which capture color and depth are fixed in place,
and a target camera frame is dragged around the scene. Point renderings are then dynamically generated in the target frame,
using method ivy_vision.quantize_to_image, both with and without the use of depth buffer.

```bash
cd ~/ivy_vision/demos
./run_demo.sh interactive.render_image
```
Example output from the simulator, and the forward warp point renderings, are given below:

<p align="center">
    <img width="75%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_vision/point_render_demo.gif?raw=true'>
</p>

## Get Involved

If you have any issues running any of the demos, would like to request further demos, or would like to implement your own, then get it touch.
Feature requests, pull requests, and [tweets](https://twitter.com/ivythread) all welcome!