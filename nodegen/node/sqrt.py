import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl


class Sqrt(RunAll):

    @staticmethod
    def sqrt_fp8x23():
        x = np.random.uniform(0, 6, (2, 2)).astype(np.float64)
        y = np.sqrt(x)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(
            y.flatten(), FixedImpl.FP8x23))

        name = "sqrt_fp8x23"
        make_node([x], [y], name)
        make_test([x], y, "input_0.sqrt()", name)

    @staticmethod
    def sqrt_fp16x16():
        x = np.random.uniform(0, 6, (2, 2)).astype(np.float64)
        y = np.sqrt(x)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "sqrt_fp16x16"
        make_node([x], [y], name)
        make_test([x], y, "input_0.sqrt()", name)
