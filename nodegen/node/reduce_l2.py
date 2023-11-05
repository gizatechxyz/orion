import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl
import numpy as np


class Reduce_l2(RunAll):
    @staticmethod
    def reduce_l2_fp8x23():
        def reduce_l2_export_do_not_keepdims():
            shape = [3, 2, 2]
            axes = np.array([2], dtype=np.int64)
            keepdims = False
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.int64)
            y = np.sqrt(np.sum(a=np.abs(x), axis=tuple(axes), keepdims=False)).astype(np.int64)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "reduce_l2_fp8x23_export_do_not_keepdims"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.reduce_l2(2, false)", name)
            
        def reduce_l2_export_keepdims():
            shape = [3, 2, 2]
            axes = np.array([2], dtype=np.int64)
            keepdims = True
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.int64)
            y = np.sqrt(np.sum(a=np.abs(x), axis=tuple(axes), keepdims=True)).astype(np.int64)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "reduce_l2_fp8x23_export_keepdims"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.reduce_l2(2, true)", name)
            
        def reduce_l2_axis_0():
            shape = [3, 3, 3]
            axes = np.array([0], dtype=np.int64)
            keepdims = True
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.int64)
            y = np.sqrt(np.sum(a=np.abs(x), axis=tuple(axes), keepdims=True)).astype(np.int64)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "reduce_l2_fp8x23_export_negative_axes_keepdims"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.reduce_l2(0, true)", name)

        
        reduce_l2_export_do_not_keepdims()
        reduce_l2_export_keepdims()
        reduce_l2_axis_0()

    @staticmethod
    def reduce_l2_fp16x16():
        def reduce_l2_export_do_not_keepdims():
            shape = [3, 2, 2]
            axes = np.array([2], dtype=np.int64)
            keepdims = False
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.int64)
            y = np.sqrt(np.sum(a=np.abs(x), axis=tuple(axes), keepdims=False)).astype(np.int64)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "reduce_l2_fp16x16_export_do_not_keepdims"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.reduce_l2(2, false)", name)
            
        def reduce_l2_export_keepdims():
            shape = [3, 2, 2]
            axes = np.array([2], dtype=np.int64)
            keepdims = True
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.int64)
            y = np.sqrt(np.sum(a=np.abs(x), axis=tuple(axes), keepdims=True)).astype(np.int64)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "reduce_l2_fp16x16_export_keepdims"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.reduce_l2(2, true)", name)
            
        def reduce_l2_axis_0():
            shape = [3, 3, 3]
            axes = np.array([0], dtype=np.int64)
            keepdims = True
            x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape).astype(np.int64)
            y = np.sqrt(np.sum(a=np.abs(x), axis=tuple(axes), keepdims=True)).astype(np.int64)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "reduce_l2_fp16x16_export_negative_axes_keepdims"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.reduce_l2(0, true)", name)

        
        reduce_l2_export_do_not_keepdims()
        reduce_l2_export_keepdims()
        reduce_l2_axis_0()