import numpy as np
from datagen.node import RunAll
from ..helpers import make_node, to_fp, Tensor, Dtype, FixedImpl


class Abs(RunAll):
    @staticmethod
    def abs_i32():
        x = np.random.randint(-127, 127, (2, 2)).astype(np.int32)
        y = abs(x)

        x = Tensor(Dtype.I32, x.shape, x.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())
        make_node([x], [y], "abs_i32")

    @staticmethod
    def abs_i8():
        x = np.random.randint(-127, 127, (2, 2)).astype(np.int8)
        y = abs(x)

        x = Tensor(Dtype.I8, x.shape, x.flatten())
        y = Tensor(Dtype.I8, y.shape, y.flatten())
        make_node([x], [y], "abs_i8")
    
    @staticmethod
    def abs_fp8x23():
        x = to_fp(np.random.randint(-127, 127, (2, 2)).astype(np.int64), FixedImpl.FP8x23)
        y = abs(x)

        x = Tensor(Dtype.FP8x23, x.shape, x.flatten(), FixedImpl.FP8x23)
        y = Tensor(Dtype.FP8x23, y.shape, y.flatten(), FixedImpl.FP8x23)
        make_node([x], [y], "abs_fp8x23")

    @staticmethod
    def abs_fp16x16():
        x = to_fp(np.random.randint(-127, 127, (2, 2)).astype(np.int64), FixedImpl.FP16x16)
        y = abs(x)

        x = Tensor(Dtype.FP16x16, x.shape, x.flatten(), FixedImpl.FP16x16)
        y = Tensor(Dtype.FP16x16, y.shape, y.flatten(), FixedImpl.FP16x16)
        make_node([x], [y], "abs_fp16x16")
