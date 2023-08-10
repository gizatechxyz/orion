import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl, Trait


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log(np.exp(x) + 1)


class Softplus(RunAll):

    @staticmethod
    def softplus_i32():
        def fp8x23():
            x = np.random.randint(-3, 3, (2, 2)).astype(np.int32)
            y = softplus(x)

            x = Tensor(Dtype.I32, x.shape, x.flatten(), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)

            name = "softplus_i32_fp8x23"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softplus(@input_0)",
                      name, Trait.NN)

        def fp16x16():
            x = np.random.randint(-3, 3, (2, 2)).astype(np.int32)
            y = softplus(x)

            x = Tensor(Dtype.I32, x.shape, x.flatten(), FixedImpl.FP16x16)
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)

            name = "softplus_i32_fp16x16"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softplus(@input_0)",
                      name, Trait.NN)

        fp8x23()
        fp16x16()

    @staticmethod
    def softplus_i8():
        def fp8x23():
            x = np.random.randint(-3, 3, (2, 2)).astype(np.int8)
            y = softplus(x)

            x = Tensor(Dtype.I8, x.shape, x.flatten(), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)

            name = "softplus_i8_fp8x23"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softplus(@input_0)",
                      name, Trait.NN)

        def fp16x16():
            x = np.random.randint(-3, 3, (2, 2)).astype(np.int8)
            y = softplus(x)

            x = Tensor(Dtype.I8, x.shape, x.flatten(), FixedImpl.FP16x16)
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)

            name = "softplus_i8_fp16x16"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softplus(@input_0)",
                      name, Trait.NN)

        fp8x23()
        fp16x16()

    @staticmethod
    def softplus_u32():
        def fp8x23():
            x = np.random.randint(0, 6, (2, 2)).astype(np.int32)
            y = softplus(x)

            x = Tensor(Dtype.U32, x.shape, x.flatten(), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)

            name = "softplus_u32_fp8x23"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softplus(@input_0)",
                      name, Trait.NN)

        def fp16x16():
            x = np.random.randint(0, 6, (2, 2)).astype(np.int32)
            y = softplus(x)

            x = Tensor(Dtype.U32, x.shape, x.flatten(), FixedImpl.FP16x16)
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)

            name = "softplus_u32_fp16x16"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softplus(@input_0)",
                      name, Trait.NN)

        fp8x23()
        fp16x16()
