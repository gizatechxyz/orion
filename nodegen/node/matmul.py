import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl


class Matmul(RunAll):

    @staticmethod
    def matmul_u32():
        def matmul_1D():
            a = np.random.randint(0, 255, (3)).astype(np.uint32)
            b = np.random.randint(0, 255, (3)).astype(np.uint32)
            y = np.matmul(a, b).reshape((1))

            a = Tensor(Dtype.U32, a.shape, a.flatten())
            b = Tensor(Dtype.U32, b.shape, b.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "matmul_u32_1d"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        def matmul_2x2():
            a = np.random.randint(0, 255, (2, 2)).astype(np.uint32)
            b = np.random.randint(0, 255, (2, 2)).astype(np.uint32)
            y = np.matmul(a, b)

            a = Tensor(Dtype.U32, a.shape, a.flatten())
            b = Tensor(Dtype.U32, b.shape, b.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "matmul_u32_2x2"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        def matmul_2x1():
            a = np.random.randint(0, 255, (2, 1)).astype(np.uint32)
            b = np.random.randint(0, 255, (1, 2)).astype(np.uint32)
            y = np.matmul(a, b)

            a = Tensor(Dtype.U32, a.shape, a.flatten())
            b = Tensor(Dtype.U32, b.shape, b.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "matmul_u32_2x1"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        def matmul_1x2():
            a = np.random.randint(0, 255, (1, 2)).astype(np.uint32)
            b = np.random.randint(0, 255, (2, 1)).astype(np.uint32)
            y = np.matmul(a, b)

            a = Tensor(Dtype.U32, a.shape, a.flatten())
            b = Tensor(Dtype.U32, b.shape, b.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "matmul_u32_1x2"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        matmul_1D()
        matmul_2x2()
        matmul_2x1()
        matmul_1x2()

    @staticmethod
    def matmul_i32():
        def matmul_1D():
            a = np.random.randint(-127, 127, (3)).astype(np.int32)
            b = np.random.randint(-127, 127, (3)).astype(np.int32)
            y = np.matmul(a, b).reshape((1))

            a = Tensor(Dtype.I32, a.shape, a.flatten())
            b = Tensor(Dtype.I32, b.shape, b.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "matmul_i32_1d"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        def matmul_2x2():
            a = np.random.randint(-127, 127, (2, 2)).astype(np.int32)
            b = np.random.randint(-127, 127, (2, 2)).astype(np.int32)
            y = np.matmul(a, b)

            a = Tensor(Dtype.I32, a.shape, a.flatten())
            b = Tensor(Dtype.I32, b.shape, b.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "matmul_i32_2x2"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        def matmul_2x1():
            a = np.random.randint(-127, 127, (2, 1)).astype(np.int32)
            b = np.random.randint(-127, 127, (1, 2)).astype(np.int32)
            y = np.matmul(a, b)

            a = Tensor(Dtype.I32, a.shape, a.flatten())
            b = Tensor(Dtype.I32, b.shape, b.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "matmul_i32_2x1"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        def matmul_1x2():
            a = np.random.randint(-127, 127, (1, 2)).astype(np.int32)
            b = np.random.randint(-127, 127, (2, 1)).astype(np.int32)
            y = np.matmul(a, b)

            a = Tensor(Dtype.I32, a.shape, a.flatten())
            b = Tensor(Dtype.I32, b.shape, b.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "matmul_i32_1x2"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        matmul_1D()
        matmul_2x2()
        matmul_2x1()
        matmul_1x2()

    @staticmethod
    def matmul_i8():
        def matmul_1D():
            a = np.random.randint(-4, 5, (3)).astype(np.int8)
            b = np.random.randint(-4, 5, (3)).astype(np.int8)
            y = np.matmul(a, b).reshape((1))

            a = Tensor(Dtype.I8, a.shape, a.flatten())
            b = Tensor(Dtype.I8, b.shape, b.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "matmul_i8_1d"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        def matmul_2x2():
            a = np.random.randint(-4, 5, (2, 2)).astype(np.int8)
            b = np.random.randint(-4, 5, (2, 2)).astype(np.int8)
            y = np.matmul(a, b)

            a = Tensor(Dtype.I8, a.shape, a.flatten())
            b = Tensor(Dtype.I8, b.shape, b.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "matmul_i8_2x2"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        def matmul_2x1():
            a = np.random.randint(-4, 5, (2, 1)).astype(np.int8)
            b = np.random.randint(-4, 5, (1, 2)).astype(np.int8)
            y = np.matmul(a, b)

            a = Tensor(Dtype.I8, a.shape, a.flatten())
            b = Tensor(Dtype.I8, b.shape, b.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "matmul_i8_2x1"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        def matmul_1x2():
            a = np.random.randint(-4, 5, (1, 2)).astype(np.int8)
            b = np.random.randint(-4, 5, (2, 1)).astype(np.int8)
            y = np.matmul(a, b)

            a = Tensor(Dtype.I8, a.shape, a.flatten())
            b = Tensor(Dtype.I8, b.shape, b.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "matmul_i8_1x2"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        matmul_1D()
        matmul_2x2()
        matmul_2x1()
        matmul_1x2()

    @staticmethod
    def matmul_fp8x23():
        def matmul_1D():
            a = np.random.randint(-3, 4, (3)).astype(np.int64)
            b = np.random.randint(-3, 4, (3)).astype(np.int64)
            y = np.matmul(a, b).reshape((1))

            a = Tensor(Dtype.FP8x23, a.shape, to_fp(
                a.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)
            b = Tensor(Dtype.FP8x23, b.shape, to_fp(
                b.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)

            name = "matmul_fp8x23_1d"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        def matmul_2x2():
            a = np.random.randint(-3, 4, (2, 2)).astype(np.int64)
            b = np.random.randint(-3, 4, (2, 2)).astype(np.int64)
            y = np.matmul(a, b)

            a = Tensor(Dtype.FP8x23, a.shape, to_fp(
                a.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)
            b = Tensor(Dtype.FP8x23, b.shape, to_fp(
                b.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)

            name = "matmul_fp8x23_2x2"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        def matmul_2x1():
            a = np.random.randint(-3, 4, (2, 1)).astype(np.int64)
            b = np.random.randint(-3, 4, (1, 2)).astype(np.int64)
            y = np.matmul(a, b)

            a = Tensor(Dtype.FP8x23, a.shape, to_fp(
                a.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)
            b = Tensor(Dtype.FP8x23, b.shape, to_fp(
                b.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)

            name = "matmul_fp8x23_2x1"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        def matmul_1x2():
            a = np.random.randint(-3, 4, (1, 2)).astype(np.int64)
            b = np.random.randint(-3, 4, (2, 1)).astype(np.int64)
            y = np.matmul(a, b)

            a = Tensor(Dtype.FP8x23, a.shape, to_fp(
                a.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)
            b = Tensor(Dtype.FP8x23, b.shape, to_fp(
                b.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)

            name = "matmul_fp8x23_1x2"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        matmul_1D()
        matmul_2x2()
        matmul_2x1()
        matmul_1x2()

    @staticmethod
    def matmul_fp16x16():
        def matmul_1D():
            a = np.random.randint(-3, 4, (3)).astype(np.int64)
            b = np.random.randint(-3, 4, (3)).astype(np.int64)
            y = np.matmul(a, b).reshape((1))

            a = Tensor(Dtype.FP16x16, a.shape, to_fp(
                a.flatten(), FixedImpl.FP16x16))
            b = Tensor(Dtype.FP16x16, b.shape, to_fp(
                b.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16))

            name = "matmul_fp16x16_1d"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        def matmul_2x2():
            a = np.random.randint(-3, 4, (2, 2)).astype(np.int64)
            b = np.random.randint(-3, 4, (2, 2)).astype(np.int64)
            y = np.matmul(a, b)

            a = Tensor(Dtype.FP16x16, a.shape, to_fp(
                a.flatten(), FixedImpl.FP16x16))
            b = Tensor(Dtype.FP16x16, b.shape, to_fp(
                b.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16))

            name = "matmul_fp16x16_2x2"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        def matmul_2x1():
            a = np.random.randint(-3, 4, (2, 1)).astype(np.int64)
            b = np.random.randint(-3, 4, (1, 2)).astype(np.int64)
            y = np.matmul(a, b)

            a = Tensor(Dtype.FP16x16, a.shape, to_fp(
                a.flatten(), FixedImpl.FP16x16))
            b = Tensor(Dtype.FP16x16, b.shape, to_fp(
                b.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16))

            name = "matmul_fp16x16_2x1"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        def matmul_1x2():
            a = np.random.randint(-3, 4, (1, 2)).astype(np.int64)
            b = np.random.randint(-3, 4, (2, 1)).astype(np.int64)
            y = np.matmul(a, b)

            a = Tensor(Dtype.FP16x16, a.shape, to_fp(
                a.flatten(), FixedImpl.FP16x16))
            b = Tensor(Dtype.FP16x16, b.shape, to_fp(
                b.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16))

            name = "matmul_fp16x16_1x2"
            make_node([a, b], [y], name)
            make_test(
                [a, b], y, "input_0.matmul(@input_1)", name)

        matmul_1D()
        matmul_2x2()
        matmul_2x1()
        matmul_1x2()
