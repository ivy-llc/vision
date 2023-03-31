# global
import pytest
from typing import Dict

# local
import ivy
import ivy.compiler.compiler as ic
from ivy_tests.test_ivy import helpers


FW_STRS = ['numpy', 'jax', 'tensorflow', 'torch']


TEST_BACKENDS: Dict[str, callable] = {
    "numpy": lambda: helpers.globals._get_ivy_numpy(),
    "jax": lambda: helpers.globals._get_ivy_jax(),
    "tensorflow": lambda: helpers.globals._get_ivy_tensorflow(),
    "torch": lambda: helpers.globals._get_ivy_torch(),
}

@pytest.fixture(autouse=True)
def run_around_tests(dev_str, f, compile_graph, implicit, fw):
    if 'gpu' in dev_str and fw == 'numpy':
        # Numpy does not support GPU
        pytest.skip()
    ivy.unset_backend()
    with f.use:
        with ivy.DefaultDevice(dev_str):
            yield


def pytest_generate_tests(metafunc):

    # device
    raw_value = metafunc.config.getoption('--device')
    if raw_value == 'all':
        devices = ['cpu', 'gpu:0', 'tpu:0']
    else:
        devices = raw_value.split(',')

    # framework
    raw_value = metafunc.config.getoption('--backend')
    if raw_value == 'all':
        backend_strs = TEST_BACKENDS.keys()
    else:
        backend_strs = raw_value.split(',')

    # compile_graph
    raw_value = metafunc.config.getoption('--compile_graph')
    if raw_value == 'both':
        compile_modes = [True, False]
    elif raw_value == 'true':
        compile_modes = [True]
    else:
        compile_modes = [False]

    # with_implicit
    raw_value = metafunc.config.getoption('--with_implicit')
    if raw_value == "true":
        implicit_modes = [True, False]
    else:
        implicit_modes = [False]

    # create test configs
    configs = list()
    for backend_str in backend_strs:
        for device in devices:
            for compile_graph in compile_modes:
                for implicit in implicit_modes:
                    configs.append(
                        (device, TEST_BACKENDS[backend_str](), compile_graph, implicit, backend_str))
    metafunc.parametrize('dev_str,f,compile_graph,implicit,fw', configs)


def pytest_addoption(parser):
    parser.addoption('--device', action="store", default="cpu")
    parser.addoption('--backend', action="store", default="numpy,jax,tensorflow,torch")
    parser.addoption('--compile_graph', action="store", default="true")
    parser.addoption('--with_implicit', action="store", default="false")
