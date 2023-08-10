import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    return tmp / s

class Softmax(RunAll):

    @staticmethod
    def softmax_i32():
        def fp8x23():
            x = np.random.randint(-3, 3, (2, 2)).astype(np.int32)
            y = softmax(x, 0)

            x = Tensor(Dtype.I32, x.shape, x.flatten(), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)

            name = "softmax_i32_fp8x23"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softmax(@input_0, 0)",
                      name, Trait.NN)

        def fp16x16():
            x = np.random.randint(-3, 3, (2, 2)).astype(np.int32)
            y = softmax(x, 1)

            x = Tensor(Dtype.I32, x.shape, x.flatten(), FixedImpl.FP16x16)
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)

            name = "softmax_i32_fp16x16"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softmax(@input_0, 1)",
                      name, Trait.NN)

        fp8x23()
        fp16x16()
    
    @staticmethod
    def softmax_i8():
        def fp8x23():
            x = np.random.randint(-3, 3, (2, 2)).astype(np.int8)
            y = softmax(x, 1)

            x = Tensor(Dtype.I8, x.shape, x.flatten(), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)

            name = "softmax_i8_fp8x23"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softmax(@input_0, 1)",
                      name, Trait.NN)

        def fp16x16():
            x = np.random.randint(-3, 3, (2, 2)).astype(np.int8)
            y = softmax(x, 0)

            x = Tensor(Dtype.I8, x.shape, x.flatten(), FixedImpl.FP16x16)
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)

            name = "softmax_i8_fp16x16"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softmax(@input_0, 0)",
                      name, Trait.NN)

        fp8x23()
        fp16x16()

    @staticmethod
    def softmax_u32():
        def fp8x23():
            x = np.random.randint(0, 3, (2, 2)).astype(np.int32)
            y = softmax(x, 1)

            x = Tensor(Dtype.U32, x.shape, x.flatten(), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)

            name = "softmax_u32_fp8x23"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softmax(@input_0, 1)",
                      name, Trait.NN)

        def fp16x16():
            x = np.random.randint(0, 3, (2, 2)).astype(np.int32)
            y = softmax(x, 0)

            x = Tensor(Dtype.U32, x.shape, x.flatten(), FixedImpl.FP16x16)
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)

            name = "softmax_u32_fp16x16"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softmax(@input_0, 0)",
                      name, Trait.NN)

        fp8x23()
        fp16x16()
