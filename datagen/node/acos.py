import numpy as np
from datagen.node import RunAll
from ..helpers import make_node, to_fp, Tensor, Dtype, FixedImpl

class Acos(RunAll):
   
    @staticmethod
    def acos_fp8x23():
        x = np.random.uniform(-1, 1, (2, 2)).astype(np.float64)
        y = np.arccos(x)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)
        make_node([x], [y], "acos_fp8x23")

     
    @staticmethod
    def acos_fp16x16():
        x = np.random.uniform(-1, 1, (2, 2)).astype(np.float64)
        y = np.arccos(x)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)
        make_node([x], [y], "acos_fp16x16")

     





 

