import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl


class Cumsum(RunAll):
    @staticmethod
    def cumsum_u32():
        def cumsum_1D():
            def default():
                x = np.array([1, 2, 3, 4, 5]).astype(np.uint32)
                y = np.array([1, 3, 6, 10, 15]).astype(np.uint32)

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "cumsum_u32_1d_default"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::None(()), Option::None(()))", name)

            def exclusive():
                x = np.array([1, 2, 3, 4, 5]).astype(np.uint32)
                y = np.array([0, 1, 3, 6, 10]).astype(np.uint32)

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "cumsum_u32_1d_exclusive"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::Some(true), Option::Some(false))", name)

            def reverse():
                x = np.array([1, 2, 3, 4, 5]).astype(np.uint32)
                y = np.array([15, 14, 12, 9, 5]).astype(np.uint32)

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "cumsum_u32_1d_reverse"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::Some(false), Option::Some(true))", name)

            def reverse_exclusive():
                x = np.array([1, 2, 3, 4, 5]).astype(np.uint32)
                y = np.array([14, 12, 9, 5, 0]).astype(np.uint32)

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "cumsum_u32_1d_reverse_exclusive"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::Some(true), Option::Some(true))", name)

            default()
            exclusive()
            reverse()
            reverse_exclusive()
        cumsum_1D()

        def cumsum_2D():
            def axis_0():
                x = np.array([1, 2, 3, 4, 5, 6]).astype(
                    np.uint32).reshape((2, 3))
                y = np.array([1, 2, 3, 5, 7, 9]).astype(
                    np.uint32).reshape((2, 3))

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "cumsum_u32_2d_axis_0"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::None(()), Option::None(()))", name)

            def axis_1():
                x = np.array([1, 2, 3, 4, 5, 6]).astype(
                    np.uint32).reshape((2, 3))
                y = np.array([1, 3, 6, 4, 9, 15]).astype(
                    np.uint32).reshape((2, 3))

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "cumsum_u32_2d_axis_1"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(1, Option::None(()), Option::None(()))", name)

            axis_0()
            axis_1()
        cumsum_2D()

    @staticmethod
    def cumsum_i32():
        def cumsum_1D():
            def default():
                x = np.array([1, 2, 3, 4, 5]).astype(np.int32)
                y = np.array([1, 3, 6, 10, 15]).astype(np.int32)

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.I32, y.shape, y.flatten())

                name = "cumsum_i32_1d_default"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::None(()), Option::None(()))", name)

            def exclusive():
                x = np.array([1, 2, 3, 4, 5]).astype(np.int32)
                y = np.array([0, 1, 3, 6, 10]).astype(np.int32)

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.I32, y.shape, y.flatten())

                name = "cumsum_i32_1d_exclusive"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::Some(true), Option::Some(false))", name)

            def reverse():
                x = np.array([1, 2, 3, 4, 5]).astype(np.int32)
                y = np.array([15, 14, 12, 9, 5]).astype(np.int32)

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.I32, y.shape, y.flatten())

                name = "cumsum_i32_1d_reverse"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::Some(false), Option::Some(true))", name)

            def reverse_exclusive():
                x = np.array([1, 2, 3, 4, 5]).astype(np.int32)
                y = np.array([14, 12, 9, 5, 0]).astype(np.int32)

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.I32, y.shape, y.flatten())

                name = "cumsum_i32_1d_reverse_exclusive"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::Some(true), Option::Some(true))", name)

            default()
            exclusive()
            reverse()
            reverse_exclusive()
        cumsum_1D()

        def cumsum_2D():
            def axis_0():
                x = np.array([1, 2, 3, 4, 5, 6]).astype(
                    np.int32).reshape((2, 3))
                y = np.array([1, 2, 3, 5, 7, 9]).astype(
                    np.int32).reshape((2, 3))

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.I32, y.shape, y.flatten())

                name = "cumsum_i32_2d_axis_0"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::None(()), Option::None(()))", name)

            def axis_1():
                x = np.array([1, 2, 3, 4, 5, 6]).astype(
                    np.int32).reshape((2, 3))
                y = np.array([1, 3, 6, 4, 9, 15]).astype(
                    np.int32).reshape((2, 3))

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.I32, y.shape, y.flatten())

                name = "cumsum_i32_2d_axis_1"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(1, Option::None(()), Option::None(()))", name)

            axis_0()
            axis_1()
        cumsum_2D()

    @staticmethod
    def cumsum_i8():
        def cumsum_1D():
            def default():
                x = np.array([1, 2, 3, 4, 5]).astype(np.int8)
                y = np.array([1, 3, 6, 10, 15]).astype(np.int8)

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.I8, y.shape, y.flatten())

                name = "cumsum_i8_1d_default"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::None(()), Option::None(()))", name)

            def exclusive():
                x = np.array([1, 2, 3, 4, 5]).astype(np.int8)
                y = np.array([0, 1, 3, 6, 10]).astype(np.int8)

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.I8, y.shape, y.flatten())

                name = "cumsum_i8_1d_exclusive"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::Some(true), Option::Some(false))", name)

            def reverse():
                x = np.array([1, 2, 3, 4, 5]).astype(np.int8)
                y = np.array([15, 14, 12, 9, 5]).astype(np.int8)

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.I8, y.shape, y.flatten())

                name = "cumsum_i8_1d_reverse"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::Some(false), Option::Some(true))", name)

            def reverse_exclusive():
                x = np.array([1, 2, 3, 4, 5]).astype(np.int8)
                y = np.array([14, 12, 9, 5, 0]).astype(np.int8)

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.I8, y.shape, y.flatten())

                name = "cumsum_i8_1d_reverse_exclusive"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::Some(true), Option::Some(true))", name)

            default()
            exclusive()
            reverse()
            reverse_exclusive()
        cumsum_1D()

        def cumsum_2D():
            def axis_0():
                x = np.array([1, 2, 3, 4, 5, 6]).astype(
                    np.int8).reshape((2, 3))
                y = np.array([1, 2, 3, 5, 7, 9]).astype(
                    np.int8).reshape((2, 3))

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.I8, y.shape, y.flatten())

                name = "cumsum_i8_2d_axis_0"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::None(()), Option::None(()))", name)

            def axis_1():
                x = np.array([1, 2, 3, 4, 5, 6]).astype(
                    np.int8).reshape((2, 3))
                y = np.array([1, 3, 6, 4, 9, 15]).astype(
                    np.int8).reshape((2, 3))

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.I8, y.shape, y.flatten())

                name = "cumsum_i8_2d_axis_1"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(1, Option::None(()), Option::None(()))", name)

            axis_0()
            axis_1()
        cumsum_2D()

    @staticmethod
    def cumsum_fp8x23():
        def cumsum_1D():
            def default():
                x = np.array([1, 2, 3, 4, 5]).astype(np.int64)
                y = np.array([1, 3, 6, 10, 15]).astype(np.int64)

                x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                    x.flatten(), FixedImpl.FP8x23))
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                    y.flatten(), FixedImpl.FP8x23))

                name = "cumsum_fp8x23_1d_default"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::None(()), Option::None(()))", name)

            def exclusive():
                x = np.array([1, 2, 3, 4, 5]).astype(np.int64)
                y = np.array([0, 1, 3, 6, 10]).astype(np.int64)

                x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                    x.flatten(), FixedImpl.FP8x23))
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                    y.flatten(), FixedImpl.FP8x23))

                name = "cumsum_fp8x23_1d_exclusive"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::Some(true), Option::Some(false))", name)

            def reverse():
                x = np.array([1, 2, 3, 4, 5]).astype(np.int64)
                y = np.array([15, 14, 12, 9, 5]).astype(np.int64)

                x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                    x.flatten(), FixedImpl.FP8x23))
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                    y.flatten(), FixedImpl.FP8x23))

                name = "cumsum_fp8x23_1d_reverse"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::Some(false), Option::Some(true))", name)

            def reverse_exclusive():
                x = np.array([1, 2, 3, 4, 5]).astype(np.int64)
                y = np.array([14, 12, 9, 5, 0]).astype(np.int64)

                x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                    x.flatten(), FixedImpl.FP8x23))
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                    y.flatten(), FixedImpl.FP8x23))

                name = "cumsum_fp8x23_1d_reverse_exclusive"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::Some(true), Option::Some(true))", name)

            default()
            exclusive()
            reverse()
            reverse_exclusive()
        cumsum_1D()

        def cumsum_2D():
            def axis_0():
                x = np.array([1, 2, 3, 4, 5, 6]).astype(
                    np.int64).reshape((2, 3))
                y = np.array([1, 2, 3, 5, 7, 9]).astype(
                    np.int64).reshape((2, 3))

                x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                    x.flatten(), FixedImpl.FP8x23))
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                    y.flatten(), FixedImpl.FP8x23))

                name = "cumsum_fp8x23_2d_axis_0"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::None(()), Option::None(()))", name)

            def axis_1():
                x = np.array([1, 2, 3, 4, 5, 6]).astype(
                    np.int64).reshape((2, 3))
                y = np.array([1, 3, 6, 4, 9, 15]).astype(
                    np.int64).reshape((2, 3))

                x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                    x.flatten(), FixedImpl.FP8x23))
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                    y.flatten(), FixedImpl.FP8x23))

                name = "cumsum_fp8x23_2d_axis_1"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(1, Option::None(()), Option::None(()))", name)

            axis_0()
            axis_1()
        cumsum_2D()

    @staticmethod
    def cumsum_fp16x16():
        def cumsum_1D():
            def default():
                x = np.array([1, 2, 3, 4, 5]).astype(np.int64)
                y = np.array([1, 3, 6, 10, 15]).astype(np.int64)

                x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                    x.flatten(), FixedImpl.FP16x16))
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                    y.flatten(), FixedImpl.FP16x16))

                name = "cumsum_fp16x16_1d_default"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::None(()), Option::None(()))", name)

            def exclusive():
                x = np.array([1, 2, 3, 4, 5]).astype(np.int64)
                y = np.array([0, 1, 3, 6, 10]).astype(np.int64)

                x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                    x.flatten(), FixedImpl.FP16x16))
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                    y.flatten(), FixedImpl.FP16x16))

                name = "cumsum_fp16x16_1d_exclusive"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::Some(true), Option::Some(false))", name)

            def reverse():
                x = np.array([1, 2, 3, 4, 5]).astype(np.int64)
                y = np.array([15, 14, 12, 9, 5]).astype(np.int64)

                x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                    x.flatten(), FixedImpl.FP16x16))
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                    y.flatten(), FixedImpl.FP16x16))

                name = "cumsum_fp16x16_1d_reverse"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::Some(false), Option::Some(true))", name)

            def reverse_exclusive():
                x = np.array([1, 2, 3, 4, 5]).astype(np.int64)
                y = np.array([14, 12, 9, 5, 0]).astype(np.int64)

                x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                    x.flatten(), FixedImpl.FP16x16))
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                    y.flatten(), FixedImpl.FP16x16))

                name = "cumsum_fp16x16_1d_reverse_exclusive"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::Some(true), Option::Some(true))", name)

            default()
            exclusive()
            reverse()
            reverse_exclusive()
        cumsum_1D()

        def cumsum_2D():
            def axis_0():
                x = np.array([1, 2, 3, 4, 5, 6]).astype(
                    np.int64).reshape((2, 3))
                y = np.array([1, 2, 3, 5, 7, 9]).astype(
                    np.int64).reshape((2, 3))

                x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                    x.flatten(), FixedImpl.FP16x16))
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                    y.flatten(), FixedImpl.FP16x16))

                name = "cumsum_fp16x16_2d_axis_0"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(0, Option::None(()), Option::None(()))", name)

            def axis_1():
                x = np.array([1, 2, 3, 4, 5, 6]).astype(
                    np.int64).reshape((2, 3))
                y = np.array([1, 3, 6, 4, 9, 15]).astype(
                    np.int64).reshape((2, 3))

                x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                    x.flatten(), FixedImpl.FP16x16))
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                    y.flatten(), FixedImpl.FP16x16))

                name = "cumsum_fp16x16_2d_axis_1"
                make_node([x], [y], name)
                make_test(
                    [x], y, "x.cumsum(1, Option::None(()), Option::None(()))", name)

            axis_0()
            axis_1()
        cumsum_2D()
