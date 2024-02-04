import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl
import numpy as np


class Reduce_l1(RunAll):
    @staticmethod
    def reduce_l1_fp8x23():
        def reduce_l1_export_do_not_keepdims():
            shape = [3, 2, 2]
            axes = np.array([2], dtype=np.int64)
            keepdims = False
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.int64)
            y = np.sum(a=np.abs(x), axis=tuple(axes), keepdims=False).astype(np.int64)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "reduce_l1_fp8x23_export_do_not_keepdims"
            
            make_test(
                [x], y, "input_0.reduce_l1(2, false)", name)
            
        def reduce_l1_export_keepdims():
            shape = [3, 2, 2]
            axes = np.array([2], dtype=np.int64)
            keepdims = True
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.int64)
            y = np.sum(a=np.abs(x), axis=tuple(axes), keepdims=True).astype(np.int64)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "reduce_l1_fp8x23_export_keepdims"
            
            make_test(
                [x], y, "input_0.reduce_l1(2, true)", name)
            
        def reduce_l1_axis_0():
            shape = [3, 3, 3]
            axes = np.array([0], dtype=np.int64)
            keepdims = True
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.int64)
            y = np.sum(a=np.abs(x), axis=tuple(axes), keepdims=True).astype(np.int64)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "reduce_l1_fp8x23_export_negative_axes_keepdims"
            
            make_test(
                [x], y, "input_0.reduce_l1(0, true)", name)

        
        reduce_l1_export_do_not_keepdims()
        reduce_l1_export_keepdims()
        reduce_l1_axis_0()

    @staticmethod
    def reduce_l1_fp16x16():
        def reduce_l1_export_do_not_keepdims():
            shape = [3, 2, 2]
            axes = np.array([2], dtype=np.int64)
            keepdims = False
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.int64)
            y = np.sum(a=np.abs(x), axis=tuple(axes), keepdims=False).astype(np.int64)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "reduce_l1_fp16x16_export_do_not_keepdims"
            
            make_test(
                [x], y, "input_0.reduce_l1(2, false)", name)
            
        def reduce_l1_export_keepdims():
            shape = [3, 2, 2]
            axes = np.array([2], dtype=np.int64)
            keepdims = True
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.int64)
            y = np.sum(a=np.abs(x), axis=tuple(axes), keepdims=True).astype(np.int64)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "reduce_l1_fp16x16_export_keepdims"
            
            make_test(
                [x], y, "input_0.reduce_l1(2, true)", name)
            
        def reduce_l1_axis_0():
            shape = [3, 3, 3]
            axes = np.array([0], dtype=np.int64)
            keepdims = True
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.int64)
            y = np.sum(a=np.abs(x), axis=tuple(axes), keepdims=True).astype(np.int64)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "reduce_l1_fp16x16_export_negative_axes_keepdims"
            
            make_test(
                [x], y, "input_0.reduce_l1(0, true)", name)

        
        reduce_l1_export_do_not_keepdims()
        reduce_l1_export_keepdims()
        reduce_l1_axis_0()

    @staticmethod
    def reduce_l1_i8():
        def reduce_l1_export_do_not_keepdims():
            shape = [3, 2, 2]
            axes = np.array([2], dtype=np.int8)
            keepdims = False
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.int8)
            y = np.sum(a=np.abs(x), axis=tuple(axes), keepdims=False).astype(np.int8)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "reduce_l1_i8_export_do_not_keepdims"
            
            make_test(
                [x], y, "input_0.reduce_l1(2, false)", name)
            
        def reduce_l1_export_keepdims():
            shape = [3, 2, 2]
            axes = np.array([2], dtype=np.int8)
            keepdims = True
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.int8)
            y = np.sum(a=np.abs(x), axis=tuple(axes), keepdims=True).astype(np.int8)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "reduce_l1_i8_export_keepdims"
            
            make_test(
                [x], y, "input_0.reduce_l1(2, true)", name)
            
        def reduce_l1_axis_0():
            shape = [3, 3, 3]
            axes = np.array([0], dtype=np.int8)
            keepdims = True
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.int8)
            y = np.sum(a=np.abs(x), axis=tuple(axes), keepdims=True).astype(np.int8)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "reduce_l1_i8_export_negative_axes_keepdims"
            
            make_test(
                [x], y, "input_0.reduce_l1(0, true)", name)

        
        reduce_l1_export_do_not_keepdims()
        reduce_l1_export_keepdims()
        reduce_l1_axis_0()

    @staticmethod
    def reduce_l1_i32():
        def reduce_l1_export_do_not_keepdims():
            shape = [3, 2, 2]
            axes = np.array([2], dtype=np.int32)
            keepdims = False
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.int32)
            y = np.sum(a=np.abs(x), axis=tuple(axes), keepdims=False).astype(np.int32)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "reduce_l1_i32_export_do_not_keepdims"
            
            make_test(
                [x], y, "input_0.reduce_l1(2, false)", name)
            
        def reduce_l1_export_keepdims():
            shape = [3, 2, 2]
            axes = np.array([2], dtype=np.int32)
            keepdims = True
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.int32)
            y = np.sum(a=np.abs(x), axis=tuple(axes), keepdims=True).astype(np.int32)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "reduce_l1_i32_export_keepdims"
            
            make_test(
                [x], y, "input_0.reduce_l1(2, true)", name)
            
        def reduce_l1_axis_0():
            shape = [3, 3, 3]
            axes = np.array([0], dtype=np.int32)
            keepdims = True
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.int32)
            y = np.sum(a=np.abs(x), axis=tuple(axes), keepdims=True).astype(np.int32)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "reduce_l1_i32_export_negative_axes_keepdims"
            
            make_test(
                [x], y, "input_0.reduce_l1(0, true)", name)

        
        reduce_l1_export_do_not_keepdims()
        reduce_l1_export_keepdims()
        reduce_l1_axis_0()

    @staticmethod
    def reduce_l1_u32():
        def reduce_l1_export_do_not_keepdims():
            shape = [3, 2, 2]
            axes = np.array([2], dtype=np.uint32)
            keepdims = False
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.uint32)
            y = np.sum(a=np.abs(x), axis=tuple(axes), keepdims=False).astype(np.uint32)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "reduce_l1_u32_export_do_not_keepdims"
            
            make_test(
                [x], y, "input_0.reduce_l1(2, false)", name)
            
        def reduce_l1_export_keepdims():
            shape = [3, 2, 2]
            axes = np.array([2], dtype=np.uint32)
            keepdims = True
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.uint32)
            y = np.sum(a=np.abs(x), axis=tuple(axes), keepdims=True).astype(np.uint32)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "reduce_l1_u32_export_keepdims"
            
            make_test(
                [x], y, "input_0.reduce_l1(2, true)", name)
            
        def reduce_l1_axis_0():
            shape = [3, 3, 3]
            axes = np.array([0], dtype=np.uint32)
            keepdims = True
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.uint32)
            y = np.sum(a=np.abs(x), axis=tuple(axes), keepdims=True).astype(np.uint32)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "reduce_l1_u32_export_negative_axes_keepdims"
            
            make_test(
                [x], y, "input_0.reduce_l1(0, true)", name)

        
        reduce_l1_export_do_not_keepdims()
        reduce_l1_export_keepdims()
        reduce_l1_axis_0()