import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


class Reduce_sum_single_axis(RunAll):
    @staticmethod
    def reduce_sum_single_axis_u32():
        def reduce_sum_single_axis_1D():
            x = np.array([0, 1, 2,]).astype(np.uint32)
            y = np.array([3]).astype(np.uint32)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "reduce_sum_single_axis_u32_1D"
            make_test(
                [x], y, "input_0.reduce_sum_single_axis(0, false)", name)

        def reduce_sum_single_axis_2D():
            def default():
                x = np.array([0, 1, 2, 3]).astype(np.uint32).reshape(2, 2)
                y = np.array([2, 4]).astype(np.uint32)

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "reduce_sum_single_axis_u32_2D_default"
                make_test(
                    [x], y, "input_0.reduce_sum_single_axis(0, false)", name)

            def keepdims():
                x = np.array([0, 1, 2, 3]).astype(np.uint32).reshape(2, 2)
                y = np.array([2, 4]).astype(np.uint32).reshape(1, 2)

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "reduce_sum_single_axis_u32_2D_keepdims"
                make_test(
                    [x], y, "input_0.reduce_sum_single_axis(0, true)", name)

            def axis_1():
                x = np.array([0, 1, 2, 3]).astype(np.uint32).reshape(2, 2)
                y = np.array([1, 5]).astype(np.uint32)

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "reduce_sum_single_axis_u32_2D_axis_1"
                make_test(
                    [x], y, "input_0.reduce_sum_single_axis(1, false)", name)

            default()
            keepdims()
            axis_1()
        reduce_sum_single_axis_1D()
        reduce_sum_single_axis_2D()

    @staticmethod
    def reduce_sum_single_axis_i32():
        def reduce_sum_single_axis_1D():
            x = np.array([0, 1, 2,]).astype(np.int32)
            y = np.array([3]).astype(np.int32)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "reduce_sum_single_axis_i32_1D"
            make_test(
                [x], y, "input_0.reduce_sum_single_axis(0, false)", name)

        def reduce_sum_single_axis_2D():
            def default():
                x = np.array([0, 1, 2, 3]).astype(np.int32).reshape(2, 2)
                y = np.array([2, 4]).astype(np.int32)

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.I32, y.shape, y.flatten())

                name = "reduce_sum_single_axis_i32_2D_default"
                make_test(
                    [x], y, "input_0.reduce_sum_single_axis(0, false)", name)

            def keepdims():
                x = np.array([0, 1, 2, 3]).astype(np.int32).reshape(2, 2)
                y = np.array([2, 4]).astype(np.int32).reshape(1, 2)

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.I32, y.shape, y.flatten())

                name = "reduce_sum_single_axis_i32_2D_keepdims"
                make_test(
                    [x], y, "input_0.reduce_sum_single_axis(0, true)", name)

            def axis_1():
                x = np.array([0, 1, 2, 3]).astype(np.int32).reshape(2, 2)
                y = np.array([1, 5]).astype(np.int32)

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.I32, y.shape, y.flatten())

                name = "reduce_sum_single_axis_i32_2D_axis_1"
                make_test(
                    [x], y, "input_0.reduce_sum_single_axis(1, false)", name)

            default()
            keepdims()
            axis_1()
        reduce_sum_single_axis_1D()
        reduce_sum_single_axis_2D()

    @staticmethod
    def reduce_sum_single_axis_i8():
        def reduce_sum_single_axis_1D():
            x = np.array([0, 1, 2,]).astype(np.int8)
            y = np.array([3]).astype(np.int8)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "reduce_sum_single_axis_i8_1D"
            make_test(
                [x], y, "input_0.reduce_sum_single_axis(0, false)", name)

        def reduce_sum_single_axis_2D():
            def default():
                x = np.array([0, 1, 2, 3]).astype(np.int8).reshape(2, 2)
                y = np.array([2, 4]).astype(np.int8)

                x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
                y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

                name = "reduce_sum_single_axis_i8_2D_default"
                make_test(
                    [x], y, "input_0.reduce_sum_single_axis(0, false)", name)

            def keepdims():
                x = np.array([0, 1, 2, 3]).astype(np.int8).reshape(2, 2)
                y = np.array([2, 4]).astype(np.int8).reshape(1, 2)

                x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
                y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

                name = "reduce_sum_single_axis_i8_2D_keepdims"
                make_test(
                    [x], y, "input_0.reduce_sum_single_axis(0, true)", name)

            def axis_1():
                x = np.array([0, 1, 2, 3]).astype(np.int8).reshape(2, 2)
                y = np.array([1, 5]).astype(np.int8)

                x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
                y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

                name = "reduce_sum_single_axis_i8_2D_axis_1"
                make_test(
                    [x], y, "input_0.reduce_sum_single_axis(1, false)", name)

            default()
            keepdims()
            axis_1()
        reduce_sum_single_axis_1D()
        reduce_sum_single_axis_2D()

    @staticmethod
    def reduce_sum_single_axis_fp8x23():
        def reduce_sum_single_axis_1D():
            x = np.array([0, 1, 2,]).astype(np.int64)
            y = np.array([3]).astype(np.int64)

            x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                x.flatten(), FixedImpl.FP8x23))
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23))

            name = "reduce_sum_single_axis_fp8x23_1D"
            make_test(
                [x], y, "input_0.reduce_sum_single_axis(0, false)", name)

        def reduce_sum_single_axis_2D():
            def default():
                x = np.array([0, 1, 2, 3]).astype(np.int64).reshape(2, 2)
                y = np.array([2, 4]).astype(np.int64)

                x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                    x.flatten(), FixedImpl.FP8x23))
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                    y.flatten(), FixedImpl.FP8x23))

                name = "reduce_sum_single_axis_fp8x23_2D_default"
                make_test(
                    [x], y, "input_0.reduce_sum_single_axis(0, false)", name)

            def keepdims():
                x = np.array([0, 1, 2, 3]).astype(np.int64).reshape(2, 2)
                y = np.array([2, 4]).astype(np.int64).reshape(1, 2)

                x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                    x.flatten(), FixedImpl.FP8x23))
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                    y.flatten(), FixedImpl.FP8x23))

                name = "reduce_sum_single_axis_fp8x23_2D_keepdims"
                make_test(
                    [x], y, "input_0.reduce_sum_single_axis(0, true)", name)

            def axis_1():
                x = np.array([0, 1, 2, 3]).astype(np.int64).reshape(2, 2)
                y = np.array([1, 5]).astype(np.int64)

                x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                    x.flatten(), FixedImpl.FP8x23))
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                    y.flatten(), FixedImpl.FP8x23))

                name = "reduce_sum_single_axis_fp8x23_2D_axis_1"
                make_test(
                    [x], y, "input_0.reduce_sum_single_axis(1, false)", name)

            default()
            keepdims()
            axis_1()
            
        reduce_sum_single_axis_1D()
        reduce_sum_single_axis_2D()

    @staticmethod
    def reduce_sum_single_axis_fp16x16():
        def reduce_sum_single_axis_1D():
            x = np.array([0, 1, 2,]).astype(np.int64)
            y = np.array([3]).astype(np.int64)

            x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                x.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16))

            name = "reduce_sum_single_axis_fp16x16_1D"
            make_test(
                [x], y, "input_0.reduce_sum_single_axis(0, false)", name)

        def reduce_sum_single_axis_2D():
            def default():
                x = np.array([0, 1, 2, 3]).astype(np.int64).reshape(2, 2)
                y = np.array([2, 4]).astype(np.int64)

                x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                    x.flatten(), FixedImpl.FP16x16))
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                    y.flatten(), FixedImpl.FP16x16))

                name = "reduce_sum_single_axis_fp16x16_2D_default"
                make_test(
                    [x], y, "input_0.reduce_sum_single_axis(0, false)", name)

            def keepdims():
                x = np.array([0, 1, 2, 3]).astype(np.int64).reshape(2, 2)
                y = np.array([2, 4]).astype(np.int64).reshape(1, 2)

                x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                    x.flatten(), FixedImpl.FP16x16))
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                    y.flatten(), FixedImpl.FP16x16))

                name = "reduce_sum_single_axis_fp16x16_2D_keepdims"
                make_test(
                    [x], y, "input_0.reduce_sum_single_axis(0, true)", name)

            def axis_1():
                x = np.array([0, 1, 2, 3]).astype(np.int64).reshape(2, 2)
                y = np.array([1, 5]).astype(np.int64)

                x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                    x.flatten(), FixedImpl.FP16x16))
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                    y.flatten(), FixedImpl.FP16x16))

                name = "reduce_sum_single_axis_fp16x16_2D_axis_1"
                make_test(
                    [x], y, "input_0.reduce_sum_single_axis(1, false)", name)

            default()
            keepdims()
            axis_1()

        reduce_sum_single_axis_1D()
        reduce_sum_single_axis_2D()
