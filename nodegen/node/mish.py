import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait


class Mish(RunAll):

    @staticmethod
    def fp8x23():
        x = np.random.uniform(-4, 4, (2, 3)).astype(np.float64)
        y = x * np.tanh(np.log1p(np.exp(x)))

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(
            y.flatten(), FixedImpl.FP8x23))

        name = "mish_fp8x23"
        make_test([x], y, "NNTrait::mish(@input_0)",
                    name, Trait.NN)

    @staticmethod
    def fp16x16():
        x = np.random.uniform(-3, 3, (3, 2, 2, 3)).astype(np.float16)
        y = x * np.tanh(np.log1p(np.exp(x)))

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "mish_fp16x16"
        make_test([x], y, "NNTrait::mish(@input_0)",
                    name, Trait.NN)
        