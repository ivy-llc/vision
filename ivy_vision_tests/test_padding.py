# global
import numpy as np

# local
from ivy_vision_tests.data import TestData
from ivy_vision import padding as ivy_pad


class PaddingTestData(TestData):

    def __init__(self):
        super().__init__()

        self.omni_image = np.array([[[[0.], [1.], [2.], [3.]],
                                     [[4.], [5.], [6.], [7.]],
                                     [[8.], [9.], [10.], [11.]],
                                     [[12.], [13.], [14.], [15.]]]])

        self.padded_omni_image = np.array([[[[1.], [2.], [3.], [0.], [1.], [2.]],
                                            [[3.], [0.], [1.], [2.], [3.], [0.]],
                                            [[7.], [4.], [5.], [6.], [7.], [4.]],
                                            [[11.], [8.], [9.], [10.], [11.], [8.]],
                                            [[15.], [12.], [13.], [14.], [15.], [12.]],
                                            [[13.], [14.], [15.], [12.], [13.], [14.]]]])


td = PaddingTestData()


def test_pad_omni_image(device, call):
    assert np.allclose(call(ivy_pad.pad_omni_image, td.omni_image, 1), td.padded_omni_image, atol=1e-3)
