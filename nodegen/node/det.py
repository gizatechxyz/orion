import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait, get_data_statement


class Det(RunAll):
     
    @staticmethod
    # We test here with fp8x23 implementation.
    def fp8x23():
        x = np.random.randint(-2, 2, (2, 4, 4)).astype(np.float64)
        y = np.linalg.det(x)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape,  to_fp(
            y.flatten(), FixedImpl.FP8x23))

        name = "det_fp8x23"
        make_test([x], y, f"input_0.det()", name)
     
    @staticmethod
    # We test here with fp16x16 implementation.
    def fp16x16():
        x = np.random.randint(-3, 3, (1, 2, 2)).astype(np.float64)
        y = np.linalg.det(x)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape,  to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "det_fp16x16"
        make_test([x], y, f"input_0.det()", name)
     
    @staticmethod
    # We test here with i8 implementation.
    def i8():
        x = np.random.randint(0, 6, (3, 1, 1)).astype(np.int8)
        y = np.linalg.det(x)

        x = Tensor(Dtype.I8, x.shape, x.flatten())
        y = Tensor(Dtype.I8, y.shape, y.flatten())

        name = "det_i8"
        make_test([x], y, f"input_0.det()", name)
    
    @staticmethod
    # We test here with i32 implementation.
    def i32():
        x = np.array([[[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]],
                   [[2, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]
                   ])
        y = np.linalg.det(x)

        x = Tensor(Dtype.I32, x.shape, x.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())

        name = "det_i32"
        make_test([x], y, f"input_0.det()", name)
     