"""
Collection of tests for ivy vision demos
"""

# global
import os
import pytest
import shutil
import ivy_tests.test_ivy.helpers as helpers


def test_demo_run_through(dev_str, f, call):
    from ivy_vision_demos.run_through import main
    if call in [helpers.tf_graph_call]:
        # these particular demos are only implemented in eager mode, without compilation
        pytest.skip()
    main(False, f=f)


@pytest.mark.parametrize(
    "with_sim", [False])
def test_demo_coords_to_voxel_grid(with_sim, dev_str, f, call):
    from ivy_vision_demos.interactive.coords_to_voxel_grid import main
    if call in [helpers.tf_graph_call]:
        # these particular demos are only implemented in eager mode, without compilation
        pytest.skip()
    main(False, with_sim, f=f)


@pytest.mark.parametrize(
    "with_sim", [False])
def test_demo_render_image(with_sim, dev_str, f, call):
    from ivy_vision_demos.interactive.render_image import main
    if call in [helpers.tf_graph_call]:
        # these particular demos are only implemented in eager mode, without compilation
        pytest.skip()
    main(False, with_sim, f=f)


@pytest.mark.parametrize(
    "with_sim", [False])
def test_demo_nerf(with_sim, dev_str, f, call):
    from ivy_vision_demos.interactive.nerf import main
    if call in [helpers.np_call, helpers.tf_graph_call, helpers.mx_call]:
        # NumPy does not support gradients
        # these particular demos are only implemented in eager mode, without compilation
        # MXNet does not support splitting along an axis with a remainder after division.
        pytest.skip()
    main(1, 2, 1, 1, False, with_sim, f=f)
    cwd = os.getcwd()
    os.remove(os.path.join(cwd, 'nerf_video.mp4'))
    shutil. rmtree(os.path.join(cwd, 'nerf_renderings'))
