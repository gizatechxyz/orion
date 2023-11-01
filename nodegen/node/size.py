import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl


class Size(RunAll):

    @staticmethod
    def size_fp8x23():
        x = np.array([[1, 2, 3],[4, 5, 6],]).astype(np.float32)
        y = np.array([x.size]).astype(np.float32)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(
            y.flatten(), FixedImpl.FP8x23))

        name = "size_fp8x23"
        make_node([x], [y], name)
        make_test([x], y, "input_0.size()", name)

    @staticmethod
    def size_fp16x16():
        x = np.array([[1, 2, 3],[4, 5, 6],]).astype(np.float32)
        y = np.array([x.size]).astype(np.float32)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "size_fp16x16"
        make_node([x], [y], name)
        make_test([x], y, "input_0.size()", name)

    @staticmethod
    def size_i8():
        x = np.array([[1, 2, 3],[4, 5, 6],]).astype(np.int8)
        y = np.array([x.size]).astype(np.int8)

        x = Tensor(Dtype.I8, x.shape, x.flatten())
        y = Tensor(Dtype.I8, y.shape, y.flatten())

        name = "size_i8"
        make_node([x], [y], name)
        make_test([x], y, "input_0.size()", name)

    @staticmethod
    def size_i32():
        x = np.array([[1, 2, 3],[4, 5, 6],]).astype(np.int32)
        y = np.array([x.size]).astype(np.int32)

        x = Tensor(Dtype.I32, x.shape, x.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())

        name = "size_i32"
        make_node([x], [y], name)
        make_test([x], y, "input_0.size()", name)

    @staticmethod
    def size_u32():
        x = np.array([[1, 2, 3],[4, 5, 6],]).astype(np.uint32)
        y = np.array([x.size]).astype(np.uint32)

        x = Tensor(Dtype.U32, x.shape, x.flatten())
        y = Tensor(Dtype.U32, y.shape, y.flatten())

        name = "size_u32"
        make_node([x], [y], name)
        make_test([x], y, "input_0.size()", name)