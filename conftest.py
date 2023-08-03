# global
import pytest

# local
import ivy
import jax

jax.config.update("jax_enable_x64", True)


FW_STRS = ["numpy", "jax", "tensorflow", "torch"]


@pytest.fixture(autouse=True)
def run_around_tests(device, compile_graph, fw):
    if "gpu" in device and fw == "numpy":
        # Numpy does not support GPU
        pytest.skip()
    with ivy.utils.backend.ContextManager(fw):
        with ivy.DefaultDevice(device):
            yield


def pytest_generate_tests(metafunc):
    # device
    raw_value = metafunc.config.getoption("--device")
    if raw_value == "all":
        devices = ["cpu", "gpu:0", "tpu:0"]
    else:
        devices = raw_value.split(",")

    # framework
    raw_value = metafunc.config.getoption("--backend")
    if raw_value == "all":
        backend_strs = FW_STRS
    else:
        backend_strs = raw_value.split(",")

    # compile_graph
    raw_value = metafunc.config.getoption("--compile_graph")
    if raw_value == "both":
        compile_modes = [True, False]
    elif raw_value == "true":
        compile_modes = [True]
    else:
        compile_modes = [False]

    # create test configs
    configs = list()
    for backend_str in backend_strs:
        for device in devices:
            for compile_graph in compile_modes:
                configs.append((device, compile_graph, backend_str))
    metafunc.parametrize("device,compile_graph,fw", configs)


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cpu")
    parser.addoption("--backend", action="store", default="numpy,jax,tensorflow,torch")
    parser.addoption("--compile_graph", action="store", default="true")
