import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


class Sin(RunAll):
    
    @staticmethod
    def sin_fp8x23():
        x = np.random.uniform(-3, 7, (2, 2)).astype(np.float64)
        y = np.sin(x)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(
            y.flatten(), FixedImpl.FP8x23))

        name = "sin_fp8x23"
        make_test([x], y, "input_0.sin()", name)

    @staticmethod
    def sin_fp16x16():
        x = np.random.uniform(-3, 7, (2, 2)).astype(np.float64)
        y = np.sin(x)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "sin_fp16x16"
        make_test([x], y, "input_0.sin()", name)
