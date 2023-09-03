import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl, Trait


def logsoftmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    return (x - x_max) - np.log(s)


class Logsoftmax(RunAll):

    def logsoftmax_fp8x23():
        def axis_0():
            x = np.random.uniform(-3, 3, (2, 2)).astype(np.float64)
            y = logsoftmax(x, 0)

            x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                x.flatten(), FixedImpl.FP8x23))
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23))

            name = "logsoftmax_fp8x23_axis_0"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::logsoftmax(@input_0, 0)",
                      name, Trait.NN)

        def axis_1():
            x = np.random.uniform(-3, 3, (2, 2)).astype(np.float64)
            y = logsoftmax(x, 1)

            x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                x.flatten(), FixedImpl.FP8x23))
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23))

            name = "logsoftmax_fp8x23_axis_1"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::logsoftmax(@input_0, 1)",
                      name, Trait.NN)

        axis_0()
        axis_1()

    def logsoftmax_fp16x16():
        def axis_0():
            x = np.random.uniform(-3, 3, (2, 2)).astype(np.float64)
            y = logsoftmax(x, 0)

            x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                x.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16))

            name = "logsoftmax_fp16x16_axis_0"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::logsoftmax(@input_0, 0)",
                      name, Trait.NN)

        def axis_1():
            x = np.random.uniform(-3, 3, (2, 2)).astype(np.float64)
            y = logsoftmax(x, 1)

            x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                x.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16))

            name = "logsoftmax_fp16x16_axis_1"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::logsoftmax(@input_0, 1)",
                      name, Trait.NN)

        axis_0()
        axis_1()
