import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl


class Neg(RunAll):
    @staticmethod
    def neg_i32():
        x = np.random.randint(-127, 127, (2, 2)).astype(np.int32)
        y = np.negative(x)

        x = Tensor(Dtype.I32, x.shape, x.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())

        name = "neg_i32"
        make_node([x], [y], name)
        make_test([x], y, "input_0.neg()", name)

    @staticmethod
    def neg_i8():
        x = np.random.randint(-127, 127, (2, 2)).astype(np.int8)
        y = np.negative(x)

        x = Tensor(Dtype.I8, x.shape, x.flatten())
        y = Tensor(Dtype.I8, y.shape, y.flatten())

        name = "neg_i8"
        make_node([x], [y], name)
        make_test([x], y, "input_0.neg()", name)

    @staticmethod
    def neg_fp8x23():
        x = to_fp(np.random.randint(-127, 127, (2, 2)
                                    ).astype(np.int64), FixedImpl.FP8x23)
        y = np.negative(x)

        x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
        y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

        name = "neg_fp8x23"
        make_node([x], [y], name)
        make_test([x], y, "input_0.neg()", name)

    @staticmethod
    def neg_fp16x16():
        x = to_fp(np.random.randint(-127, 127, (2, 2)
                                    ).astype(np.int64), FixedImpl.FP16x16)
        y = np.negative(x)

        x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
        y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

        name = "neg_fp16x16"
        make_node([x], [y], name)
        make_test([x], y, "input_0.neg()", name)
