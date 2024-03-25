import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    return tmp / s


class Softmax(RunAll):
        
    @staticmethod
    def axis_0():
        x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
        y = softmax(x, axis=0)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "softmax_axis_0"
        make_test([x], y, "NNTrait::softmax(@input_0, Option::Some(0))",
                  name, Trait.NN)
    
    @staticmethod
    def axis_1():
        x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
        y = softmax(x, axis=1)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "softmax_axis_1"
        make_test([x], y, "NNTrait::softmax(@input_0, Option::Some(1))",
                  name, Trait.NN)
    
    @staticmethod
    def axis_2():
        x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
        y = softmax(x, axis=2)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "softmax_axis_2"
        make_test([x], y, "NNTrait::softmax(@input_0, Option::Some(2))",
                  name, Trait.NN)
    
    @staticmethod
    def axis_minus_1():
        x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
        y = softmax(x, axis=-1)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "softmax_axis_minus_1"
        make_test([x], y, "NNTrait::softmax(@input_0, Option::None)",
                  name, Trait.NN)
