from typing import Optional

import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl, Trait


def gemm_reference_implementation(
    A: np.ndarray,
    B: np.ndarray,
    C: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    transA: int = 0,
    transB: int = 0,
) -> np.ndarray:
    A = A if transA == 0 else A.T
    B = B if transB == 0 else B.T
    C = C if C is not None else np.array(0)

    Y = alpha * np.dot(A, B) + beta * C
    return Y


class Gemm(RunAll):

    @staticmethod
    def gemm_default_zero_bias():
        a = np.random.ranf([3, 5]).astype(np.float32)
        b = np.random.ranf([5, 4]).astype(np.float32)
        c = np.zeros([1, 4]).astype(np.float32)
        y = gemm_reference_implementation(a, b, c)

        a = Tensor(Dtype.FP16x16, a.shape, to_fp(
            a.flatten(), FixedImpl.FP16x16))
        b = Tensor(Dtype.FP16x16, b.shape, to_fp(
            b.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "gemm_default_no_bias"
        make_node([a, b], [y], name)
        make_test(
            [a, b], y, "NNTrait::gemm(input_0, input_1, Option::None(()), Option::None(()), Option::None(()), false, false)", name, Trait.NN)
        
    @staticmethod
    def gemm_default_vector_bias():
        a = np.random.ranf([2, 7]).astype(np.float32)
        b = np.random.ranf([7, 4]).astype(np.float32)
        c = np.random.ranf([1, 4]).astype(np.float32)
        y = gemm_reference_implementation(a, b, c)

        a = Tensor(Dtype.FP16x16, a.shape, to_fp(
            a.flatten(), FixedImpl.FP16x16))
        b = Tensor(Dtype.FP16x16, b.shape, to_fp(
            b.flatten(), FixedImpl.FP16x16))
        c = Tensor(Dtype.FP16x16, c.shape, to_fp(
            c.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "gemm_default_vector_bias"
        make_node([a, b, c], [y], name)
        make_test(
            [a, b, c], y, "NNTrait::gemm(input_0, input_1, Option::Some(input_2), Option::None(()), Option::None(()), false, false)", name, Trait.NN)
        
    @staticmethod
    def gemm_default_matrix_bias():
        a = np.random.ranf([3, 6]).astype(np.float32)
        b = np.random.ranf([6, 4]).astype(np.float32)
        c = np.random.ranf([3, 4]).astype(np.float32)
        y = gemm_reference_implementation(a, b, c)

        a = Tensor(Dtype.FP16x16, a.shape, to_fp(
            a.flatten(), FixedImpl.FP16x16))
        b = Tensor(Dtype.FP16x16, b.shape, to_fp(
            b.flatten(), FixedImpl.FP16x16))
        c = Tensor(Dtype.FP16x16, c.shape, to_fp(
            c.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "gemm_default_matrix_bias"
        make_node([a, b, c], [y], name)
        make_test(
            [a, b, c], y, "NNTrait::gemm(input_0, input_1, Option::Some(input_2), Option::None(()), Option::None(()), false, false)", name, Trait.NN)

    @staticmethod
    def gemm_transposeA():
        a = np.random.ranf([6, 3]).astype(np.float32)
        b = np.random.ranf([6, 4]).astype(np.float32)
        c = np.zeros([1, 4]).astype(np.float32)
        y = gemm_reference_implementation(a, b, c, transA=1)

        a = Tensor(Dtype.FP16x16, a.shape, to_fp(
            a.flatten(), FixedImpl.FP16x16))
        b = Tensor(Dtype.FP16x16, b.shape, to_fp(
            b.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "gemm_transposeA"
        make_node([a, b], [y], name)
        make_test(
            [a, b], y, "NNTrait::gemm(input_0, input_1, Option::None(()), Option::None(()), Option::None(()), true, false)", name, Trait.NN)
