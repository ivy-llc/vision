"""
Collection of tests for ivy vision demos
"""

# global
import pytest


def test_demo_run_through(device, fw):
    from ivy_vision_demos.run_through import main

    main(False, fw=fw)


@pytest.mark.parametrize("with_sim", [False])
def test_demo_coords_to_voxel_grid(with_sim, device, fw):
    from ivy_vision_demos.interactive.coords_to_voxel_grid import main

    main(False, with_sim, fw=fw)


@pytest.mark.parametrize("with_sim", [False])
def test_demo_render_image(with_sim, device, fw):
    from ivy_vision_demos.interactive.render_image import main

    main(False, with_sim, fw=fw)


@pytest.mark.parametrize("with_sim", [False])
def test_demo_nerf(with_sim, device, fw):
    from ivy_vision_demos.interactive.nerf import main

    if fw in ["numpy", "mxnet"]:
        # NumPy does not support gradients
        # these particular demos are only implemented in eager mode,
        # without compilation
        # MXNet does not support splitting along an axis
        # with a remainder after division.
        pytest.skip()
    main(1, 2, 1, 1, False, with_sim, fw=fw)
    # cwd = os.getcwd()
    # os.remove(os.path.join(cwd, "nerf_video.mp4"))
    # shutil.rmtree(os.path.join(cwd, "nerf_renderings"))
