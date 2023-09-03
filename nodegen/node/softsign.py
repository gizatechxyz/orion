import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl, Trait


def softsign(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.abs(x))


class Softsign(RunAll):

    @staticmethod
    def softsign_fp():
        def fp8x23():
            x = np.random.uniform(-5, 7, (2, 2)).astype(np.float64)
            y = softsign(x)

            x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                x.flatten(), FixedImpl.FP8x23))
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23))

            name = "softsign_fp8x23"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softsign(@input_0)",
                      name, Trait.NN)

        def fp16x16():
            x = np.random.uniform(-5, 7, (2, 2)).astype(np.float64)
            y = softsign(x)

            x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                x.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16))

            name = "softsign_fp16x16"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softsign(@input_0)",
                      name, Trait.NN)

        fp8x23()
        fp16x16()
