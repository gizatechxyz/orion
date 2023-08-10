import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl, Trait


def softsign(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.abs(x))


class Softsign(RunAll):

    @staticmethod
    def softsign_i32():
        def fp8x23():
            x = np.random.randint(-5, 9, (2, 2)).astype(np.int32)
            y = softsign(x)

            x = Tensor(Dtype.I32, x.shape, x.flatten(), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)

            name = "softsign_i32_fp8x23"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softsign(@input_0)",
                      name, Trait.NN)

        def fp16x16():
            x = np.random.randint(-5, 9, (2, 2)).astype(np.int32)
            y = softsign(x)

            x = Tensor(Dtype.I32, x.shape, x.flatten(), FixedImpl.FP16x16)
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)

            name = "softsign_i32_fp16x16"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softsign(@input_0)",
                      name, Trait.NN)

        fp8x23()
        fp16x16()

    @staticmethod
    def softsign_i8():
        def fp8x23():
            x = np.random.randint(-5, 9, (2, 2)).astype(np.int8)
            y = softsign(x)

            x = Tensor(Dtype.I8, x.shape, x.flatten(), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)

            name = "softsign_i8_fp8x23"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softsign(@input_0)",
                      name, Trait.NN)

        def fp16x16():
            x = np.random.randint(-5, 9, (2, 2)).astype(np.int8)
            y = softsign(x)

            x = Tensor(Dtype.I8, x.shape, x.flatten(), FixedImpl.FP16x16)
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)

            name = "softsign_i8_fp16x16"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softsign(@input_0)",
                      name, Trait.NN)

        fp8x23()
        fp16x16()

    @staticmethod
    def softsign_u32():
        def fp8x23():
            x = np.random.randint(0, 9, (2, 2)).astype(np.int32)
            y = softsign(x)

            x = Tensor(Dtype.U32, x.shape, x.flatten(), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)

            name = "softsign_u32_fp8x23"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softsign(@input_0)",
                      name, Trait.NN)

        def fp16x16():
            x = np.random.randint(0, 9, (2, 2)).astype(np.int32)
            y = softsign(x)

            x = Tensor(Dtype.U32, x.shape, x.flatten(), FixedImpl.FP16x16)
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)

            name = "softsign_u32_fp16x16"
            make_node([x], [y], name)
            make_test([x], y, "NNTrait::softsign(@input_0)",
                      name, Trait.NN)

        fp8x23()
        fp16x16()
