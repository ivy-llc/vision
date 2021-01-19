"""
Collection of tests for ivy vision demos
"""

# global
import ivy_vision_tests.helpers as helpers


def test_demo_run_through():
    from demos.run_through import main
    for lib, call in helpers.calls:
        if call in [helpers.tf_graph_call, helpers.mx_graph_call]:
            # these particular demos are only implemented in eager mode, without compilation
            continue
        main(False, f=lib)


def test_demo_coords_to_voxel_grid():
    from demos.interactive.coords_to_voxel_grid import main
    for lib, call in helpers.calls:
        if call in [helpers.tf_graph_call, helpers.mx_graph_call]:
            # these particular demos are only implemented in eager mode, without compilation
            continue
        main(False, False, f=lib)


def test_demo_render_image():
    from demos.interactive.render_image import main
    for lib, call in helpers.calls:
        if call in [helpers.tf_graph_call, helpers.mx_graph_call]:
            # these particular demos are only implemented in eager mode, without compilation
            continue
        main(False, False, f=lib)
