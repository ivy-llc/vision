# global
import ivy
import numpy as np

# local
from ivy_vision_tests.data import TestData
from ivy_vision import padding as ivy_pad


class PaddingTestData(TestData):
    def __init__(self):
        super().__init__()

        self.omni_image = np.array(
            [
                [
                    [[0.0], [1.0], [2.0], [3.0]],
                    [[4.0], [5.0], [6.0], [7.0]],
                    [[8.0], [9.0], [10.0], [11.0]],
                    [[12.0], [13.0], [14.0], [15.0]],
                ]
            ]
        )

        self.padded_omni_image = np.array(
            [
                [
                    [[1.0], [2.0], [3.0], [0.0], [1.0], [2.0]],
                    [[3.0], [0.0], [1.0], [2.0], [3.0], [0.0]],
                    [[7.0], [4.0], [5.0], [6.0], [7.0], [4.0]],
                    [[11.0], [8.0], [9.0], [10.0], [11.0], [8.0]],
                    [[15.0], [12.0], [13.0], [14.0], [15.0], [12.0]],
                    [[13.0], [14.0], [15.0], [12.0], [13.0], [14.0]],
                ]
            ]
        )


td = PaddingTestData()


def test_pad_omni_image(device, fw):
    ivy.set_backend(fw)
    assert np.allclose(
        ivy_pad.pad_omni_image(ivy.array(td.omni_image), 1),
        td.padded_omni_image,
        atol=1e-3,
    )
    ivy.previous_backend()
