import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, Tensor, Dtype, FixedImpl, to_fp

class Reduce_log_sum_exp(RunAll):
    @staticmethod
    def reduce_log_sum_exp_fp32x32():
        def reduce_log_sum_exp_export_do_not_keepdims():
            shape = [3, 2, 2]
            axes = np.array([2], dtype=np.int64)
            keepdims = False
            x = np.reshape(np.arange(1, np.prod(shape) + 1), shape)
            y = np.log(np.sum(np.exp(x), axis=tuple(axes), keepdims=False)).astype(np.float64)

            x = Tensor(Dtype.FP32x32, x.shape, to_fp(
            x.flatten(), FixedImpl.FP32x32))
            y = Tensor(Dtype.FP32x32, y.shape, to_fp(
            y.flatten(), FixedImpl.FP32x32))

            name = "reduce_log_sum_exp_fp32x32_export_do_not_keepdims"
            make_test(
                [x], y, "input_0.reduce_log_sum_exp(2, false)", name)
            
        def reduce_log_sum_exp_export_keepdims():
            shape = [3, 2, 2]
            axes = np.array([2], dtype=np.int64)
            keepdims = True
            x = np.reshape(np.arange(1, np.prod(shape) + 1), shape)
            y = np.log(np.sum(np.exp(x), axis=tuple(axes), keepdims=True)).astype(np.float64)

            x = Tensor(Dtype.FP32x32, x.shape, to_fp(
            x.flatten(), FixedImpl.FP32x32))
            y = Tensor(Dtype.FP32x32, y.shape, to_fp(
            y.flatten(), FixedImpl.FP32x32))

            name = "reduce_log_sum_exp_fp32x32_export_keepdims"
            make_test(
                [x], y, "input_0.reduce_log_sum_exp(2, true)", name)
            
        def reduce_log_sum_exp_axis_0():
            shape = [3, 2, 2]
            axes = np.array([0], dtype=np.int64)
            keepdims = True
            x = np.reshape(np.arange(1, np.prod(shape) + 1), shape)
            y = np.log(np.sum(np.exp(x), axis=tuple(axes), keepdims=True)).astype(np.float64)

            x = Tensor(Dtype.FP32x32, x.shape, to_fp(
            x.flatten(), FixedImpl.FP32x32))
            y = Tensor(Dtype.FP32x32, y.shape, to_fp(
            y.flatten(), FixedImpl.FP32x32))

            name = "reduce_log_sum_exp_fp32x32_export_negative_axes_keepdims"
            make_test(
                [x], y, "input_0.reduce_log_sum_exp(0, true)", name)

        
        reduce_log_sum_exp_export_do_not_keepdims()
        reduce_log_sum_exp_export_keepdims()
        reduce_log_sum_exp_axis_0()

  

