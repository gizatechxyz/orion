import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl


class Cosh(RunAll):

    @staticmethod
    def cosh_i32():

        def fp8x23():
            x = np.random.randint(-3, 3, (2, 2)).astype(np.int32)
            y = np.cosh(x)

            x = Tensor(Dtype.I32, x.shape, x.flatten(), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)

            name = "cosh_i32_fp8x23"
            make_node([x], [y], name)
            make_test([x], y, "x.cosh()", name)

        def fp16x16():
            x = np.random.randint(-3, 3, (2, 2)).astype(np.int32)
            y = np.cosh(x)

            x = Tensor(Dtype.I32, x.shape, x.flatten(), FixedImpl.FP16x16)
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)

            name = "cosh_i32_fp16x16"
            make_node([x], [y], name)
            make_test([x], y, "x.cosh()", name)

        fp8x23()
        fp16x16()

    @staticmethod
    def cosh_i8():

        def fp8x23():
            x = np.random.randint(-3, 3, (2, 2)).astype(np.int8)
            y = np.cosh(x)

            x = Tensor(Dtype.I8, x.shape, x.flatten(), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)

            name = "cosh_i8_fp8x23"
            make_node([x], [y], name)
            make_test([x], y, "x.cosh()", name)

        def fp16x16():
            x = np.random.randint(-3, 3, (2, 2)).astype(np.int32)
            y = np.cosh(x)

            x = Tensor(Dtype.I8, x.shape, x.flatten(), FixedImpl.FP16x16)
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)

            name = "cosh_i8_fp16x16"
            make_node([x], [y], name)
            make_test([x], y, "x.cosh()", name)

        fp8x23()
        fp16x16()

    @staticmethod
    def cosh_u32():

        def fp8x23():
            x = np.random.randint(0, 6, (2, 2)).astype(np.uint32)
            y = np.cosh(x)

            x = Tensor(Dtype.U32, x.shape, x.flatten(), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)

            name = "cosh_u32_fp8x23"
            make_node([x], [y], name)
            make_test([x], y, "x.cosh()", name)

        def fp16x16():
            x = np.random.randint(0, 6, (2, 2)).astype(np.uint32)
            y = np.cosh(x)

            x = Tensor(Dtype.U32, x.shape, x.flatten(), FixedImpl.FP16x16)
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)

            name = "cosh_u32_fp16x16"
            make_node([x], [y], name)
            make_test([x], y, "x.cosh()", name)

        fp8x23()
        fp16x16()

    @staticmethod
    def cosh_fp8x23():
        x = np.random.uniform(-3, 3, (2, 2)).astype(np.float64)
        y = np.cosh(x)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(
            y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)

        name = "cosh_fp8x23"
        make_node([x], [y], name)
        make_test([x], y, "x.cosh()", name)

    @staticmethod
    def cosh_fp16x16():
        x = np.random.uniform(-3, 3, (2, 2)).astype(np.float64)
        y = np.cosh(x)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)

        name = "cosh_fp16x16"
        make_node([x], [y], name)
        make_test([x], y, "x.cosh()", name)
