import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl

class Ceil(RunAll):
   
    @staticmethod
    def ceil_fp8x23():
        x = np.random.uniform(-1, 1, (2, 2)).astype(np.float64)
        y = np.ceil(x)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
        
        name = "ceil_fp8x23"
        make_test([x], y, "input_0.ceil()", name)
     
    @staticmethod
    def ceil_fp16x16():
        x = np.random.uniform(-1, 1, (2, 2)).astype(np.float64)
        y = np.ceil(x)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "ceil_fp16x16"
        make_test([x], y, "input_0.ceil()", name)
