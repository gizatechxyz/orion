import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl

class Round(RunAll):
   
    @staticmethod
    def round_fp8x23():
        x = np.array([0.1, 0.5, 0.9, 1.2, 1.5, 1.8, 2.3, 2.5, 2.7, -1.1, -1.5, -1.9, -2.2, -2.5, -2.8]).astype(np.float64)
        y = np.array([0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, -1.0, -2.0, -2.0, -2.0, -3.0, -3.0]).astype(np.float64)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
        
        name = "round_fp8x23"
        make_node([x], [y], name)
        make_test([x], y, "input_0.round()", name)
     
    @staticmethod
    def round_fp16x16():
        x = np.array([0.1, 0.5, 0.9, 1.2, 1.5, 1.8, 2.3, 2.5, 2.7, -1.1, -1.5, -1.9, -2.2, -2.5, -2.8]).astype(np.float64)
        y = np.array([0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, -1.0, -2.0, -2.0, -2.0, -3.0, -3.0]).astype(np.float64)        

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "round_fp16x16"
        make_node([x], [y], name)
        make_test([x], y, "input_0.round()", name)