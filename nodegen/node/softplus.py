import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log(np.exp(x) + 1)


class Softplus(RunAll):

    @staticmethod
    def softplus_fp():
        def fp8x23():
            x = np.random.uniform(-3, 3, (2, 2)).astype(np.float64)
            y = softplus(x)

            x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                x.flatten(), FixedImpl.FP8x23))
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23))

            name = "softplus_fp8x23"
            make_test([x], y, "NNTrait::softplus(@input_0)",
                      name, Trait.NN)

        def fp16x16():
            x = np.random.uniform(-3, 3, (2, 2)).astype(np.float64)
            y = softplus(x)

            x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                x.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16))

            name = "softplus_fp16x16"
            make_test([x], y, "NNTrait::softplus(@input_0)",
                      name, Trait.NN)

        fp8x23()
        fp16x16()
