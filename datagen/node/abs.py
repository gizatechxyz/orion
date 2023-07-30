import numpy as np
from datagen.node import RunAll
from ..helpers import make_node, Tensor, Dtype

class Abs(RunAll):
    @staticmethod
    def abs_i32_1D():
        input_0 = np.random.randint(-127, 3, (5)).astype(np.int32)
        output_0 = abs(input_0)

        input_0 = Tensor(Dtype.I32, input_0.shape, input_0.flatten())
        output_0 = Tensor(Dtype.I32, output_0.shape, output_0.flatten())
        make_node([input_0], [output_0], "abs_i32_1d")

    @staticmethod
    def abs_i32_2D():
        input_0 = np.random.randint(-127, 3, (2, 2)).astype(np.int32)
        output_0 = abs(input_0)

        input_0 = Tensor(Dtype.I32, input_0.shape, input_0.flatten())
        output_0 = Tensor(Dtype.I32, output_0.shape, output_0.flatten())
        make_node([input_0], [output_0], "abs_i32_2d")