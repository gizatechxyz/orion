import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl, Trait
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
        make_node([i, w, b], [y], name)
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
        make_node([i, w, b], [y], name)
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
        make_node([i, w, b], [y], name)
        make_test([i, w, b], y, "NNTrait::linear(input_0, input_1, input_2)",
                  name, Trait.NN)
