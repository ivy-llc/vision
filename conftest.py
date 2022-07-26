# global
import pytest
from typing import Dict

# local
import ivy
from ivy_tests.test_ivy import helpers


FW_STRS = ['numpy', 'jax', 'tensorflow', 'torch', 'mxnet']


TEST_BACKENDS: Dict[str, callable] = {'numpy': lambda: helpers.get_ivy_numpy(),
                                        'jax': lambda: helpers.get_ivy_jax(),
                                        'tensorflow': lambda: helpers.get_ivy_tensorflow(),
                                        'torch': lambda: helpers.get_ivy_torch(),
                                        'mxnet': lambda: helpers.get_ivy_mxnet()}
TEST_CALL_METHODS: Dict[str, callable] = {'numpy': helpers.np_call,
                                          'jax': helpers.jnp_call,
                                          'tensorflow': helpers.tf_call,
                                          'torch': helpers.torch_call,
                                          'mxnet': helpers.mx_call}


@pytest.fixture(autouse=True)
def run_around_tests(device, f, wrapped_mode, compile_graph, call, fw):
    if wrapped_mode and call is helpers.tf_graph_call:
        # ToDo: add support for wrapped_mode and tensorflow compilation
        pytest.skip()
    if wrapped_mode and call is helpers.jnp_call:
        # ToDo: add support for wrapped_mode with jax, presumably some errenously wrapped jax methods
        pytest.skip()
    if 'gpu' in device and call is helpers.np_call:
        # Numpy does not support GPU
        pytest.skip()
    ivy.clear_backend_stack()
    with f.use:
        # f.set_wrapped_mode(wrapped_mode)
        ivy.set_default_device(device)
        yield


def pytest_generate_tests(metafunc):

    # device
    raw_value = metafunc.config.getoption('--device')
    if raw_value == 'all':
        devices = ['cpu', 'gpu:0', 'tpu:0']
    else:
        devices = raw_value.split(',')

    # backend
    raw_value = metafunc.config.getoption('--backend')
    if raw_value == 'all':
        backend_strs = TEST_BACKEND.keys()
    else:
        backend_strs = raw_value.split(',')

    # wrapped_mode
    raw_value = metafunc.config.getoption('--wrapped_mode')
    if raw_value == 'both':
        wrapped_modes = [True, False]
    elif raw_value == 'true':
        wrapped_modes = [True]
    else:
        wrapped_modes = [False]

    # compile_graph
    raw_value = metafunc.config.getoption('--compile_graph')
    if raw_value == 'both':
        compile_modes = [True, False]
    elif raw_value == 'true':
        compile_modes = [True]
    else:
        compile_modes = [False]

    # create test configs
    configs = list()
    for backend_str in backend_strs:
        for device in devices:
            for wrapped_mode in wrapped_modes:
                for compile_graph in compile_modes:
                    configs.append(
                        (device, TEST_BACKENDS[backend_str](), wrapped_mode, compile_graph, TEST_CALL_METHODS[backend_str], backend_str))
    metafunc.parametrize('device,f,wrapped_mode,compile_graph,call,fw', configs)


def pytest_addoption(parser):
    parser.addoption('--device', action="store", default="cpu")
    parser.addoption('--backend', action="store", default="numpy,jax,tensorflow,torch")
    parser.addoption('--wrapped_mode', action="store", default="false")
    parser.addoption('--compile_graph', action="store", default="true")
