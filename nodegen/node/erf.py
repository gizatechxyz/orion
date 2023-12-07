import numpy as np
from math import erf
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


class Erf(RunAll):

    @staticmethod
    def erf_fp8x23():
        x = np.asarray([0.12, -1.66, 3.4, 4.8, 2.7]).astype(np.float64).reshape(1,5)
        y = np.asarray([erf(value) for value in x[0]]).astype(np.float64).reshape(1,5)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(
            y.flatten(), FixedImpl.FP8x23))

        name = "erf_fp8x23"
        make_test([x], y, "input_0.erf()", name)
        

    @staticmethod
    def erf_fp16x16():
        x = np.asarray([0.12, -1.66, 3.4, 4.8, 2.7]).astype(np.float64).reshape(1,5)
        y = np.asarray([erf(value) for value in x[0]]).astype(np.float64).reshape(1,5)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "erf_fp16x16"
        make_test([x], y, "input_0.erf()", name)
