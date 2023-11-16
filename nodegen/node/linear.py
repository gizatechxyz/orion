import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait
from typing import Optional


def linear(
    i: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray] = None,

) -> np.ndarray:
    return np.dot(i, w.T) + b


class Linear(RunAll):

    @staticmethod
    def linear_i32():
        i = np.random.randint(-5, 9, (3)).astype(np.int32)
        w = np.random.randint(-5, 9, (2, 3)).astype(np.int32)
        b = np.random.randint(-5, 9, (2)).astype(np.int32)
        y = linear(i, w, b)

        i = Tensor(Dtype.I32, i.shape, i.flatten())
        w = Tensor(Dtype.I32, w.shape, w.flatten())
        b = Tensor(Dtype.I32, b.shape, b.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())

        name = "linear_i32"
        make_test([i, w, b], y, "NNTrait::linear(input_0, input_1, input_2)",
                  name, Trait.NN)

    @staticmethod
    def linear_i8():
        i = np.random.randint(-3, 3, (3)).astype(np.int8)
        w = np.random.randint(-3, 3, (2, 3)).astype(np.int8)
        b = np.random.randint(-3, 3, (2)).astype(np.int8)
        y = linear(i, w, b)

        i = Tensor(Dtype.I8, i.shape, i.flatten())
        w = Tensor(Dtype.I8, w.shape, w.flatten())
        b = Tensor(Dtype.I8, b.shape, b.flatten())
        y = Tensor(Dtype.I8, y.shape, y.flatten())

        name = "linear_i8"
        make_test([i, w, b], y, "NNTrait::linear(input_0, input_1, input_2)",
                  name, Trait.NN)

    @staticmethod
    def linear_u32():
        i = np.random.randint(0, 6, (3)).astype(np.uint32)
        w = np.random.randint(0, 6, (2, 3)).astype(np.uint32)
        b = np.random.randint(0, 6, (2)).astype(np.uint32)
        y = linear(i, w, b)

        i = Tensor(Dtype.U32, i.shape, i.flatten())
        w = Tensor(Dtype.U32, w.shape, w.flatten())
        b = Tensor(Dtype.U32, b.shape, b.flatten())
        y = Tensor(Dtype.U32, y.shape, y.flatten())

        name = "linear_u32"
        make_test([i, w, b], y, "NNTrait::linear(input_0, input_1, input_2)",
                  name, Trait.NN)

    @staticmethod
    def linear_fp8x23():
        i = np.random.uniform(-5, 7, (3)).astype(np.float64)
        w = np.random.uniform(-5, 7, (2, 3)).astype(np.float64)
        b = np.random.uniform(-5, 7, (2)).astype(np.float64)
        y = linear(i, w, b)

        i = Tensor(Dtype.FP8x23, i.shape, to_fp(
            i.flatten(), FixedImpl.FP8x23))
        w = Tensor(Dtype.FP8x23, w.shape, to_fp(
            w.flatten(), FixedImpl.FP8x23))
        b = Tensor(Dtype.FP8x23, b.shape, to_fp(
            b.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(
            y.flatten(), FixedImpl.FP8x23))

        name = "linear_fp8x23"
        make_test([i, w, b], y, "NNTrait::linear(input_0, input_1, input_2)",
                  name, Trait.NN)

    @staticmethod
    def linear_fp16x16():
        i = np.random.uniform(-5, 7, (3)).astype(np.float64)
        w = np.random.uniform(-5, 7, (2, 3)).astype(np.float64)
        b = np.random.uniform(-5, 7, (2)).astype(np.float64)
        y = linear(i, w, b)

        i = Tensor(Dtype.FP16x16, i.shape, to_fp(
            i.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(
            w.flatten(), FixedImpl.FP16x16))
        b = Tensor(Dtype.FP16x16, b.shape, to_fp(
            b.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "linear_fp16x16"
        make_test([i, w, b], y, "NNTrait::linear(input_0, input_1, input_2)",
                  name, Trait.NN)
