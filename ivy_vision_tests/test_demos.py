"""
Collection of tests for ivy vision demos
"""

# global
import pytest
import ivy_tests.helpers as helpers


def test_demo_run_through(dev_str, f, call):
    from demos.run_through import main
    if call in [helpers.tf_graph_call]:
        # these particular demos are only implemented in eager mode, without compilation
        pytest.skip()
    main(False, f=f)


def test_demo_coords_to_voxel_grid(dev_str, f, call):
    from demos.interactive.coords_to_voxel_grid import main
    if call in [helpers.tf_graph_call]:
        # these particular demos are only implemented in eager mode, without compilation
        pytest.skip()
    main(False, False, f=f)


def test_demo_render_image(dev_str, f, call):
    from demos.interactive.render_image import main
    if call in [helpers.tf_graph_call]:
        # these particular demos are only implemented in eager mode, without compilation
        pytest.skip()
    main(False, False, f=f)


'''
def test_demo_nerf(dev_str, f, call):
    from demos.interactive.nerf import main
    if call in [helpers.np_call, helpers.tf_graph_call, helpers.mx_call]:
        # NumPy does not support gradients
        # these particular demos are only implemented in eager mode, without compilation
        # MXNet does not support splitting along an axis with a remainder after division.
        pytest.skip()
    main(1, False, False, f=f)
'''