# global
import numpy as np
import ivy_tests.helpers as helpers

# local
from ivy_vision import implicit as ivy_imp
from ivy_vision_tests.data import TestData


class ImplicitTestData(TestData):

    def __init__(self):
        super().__init__()
        self.start_vals = np.array([0, 1, 2], np.float32)
        self.end_vals = np.array([10, 21, 7], np.float32)


td = ImplicitTestData()


def test_stratified_sample(dev_str, call):
    if call is not helpers.mx_call:
        return
    num = 10
    res = call(ivy_imp.stratified_sample, td.start_vals, td.end_vals, num)
    assert res.shape == (3, num)
    for i in range(3):
        for j in range(num - 1):
            assert res[i][j] < res[i][j+1]
