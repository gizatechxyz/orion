import numpy as np
from datagen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl

class Asin(RunAll):
   
    @staticmethod
    def asin_fp8x23():
        x = np.random.uniform(-1, 1, (2, 2)).astype(np.float64)
        y = np.arcsin(x)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)
        
        name = "asin_fp8x23"
        make_node([x], [y], name)
        make_test([x], y, "x.asin()", name)
     
    @staticmethod
    def asin_fp16x16():
        x = np.random.uniform(-1, 1, (2, 2)).astype(np.float64)
        y = np.arcsin(x)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)

        name = "asin_fp16x16"
        make_node([x], [y], name)
        make_test([x], y, "x.asin()", name)
     





 

