import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


class Pow(RunAll):
    @staticmethod
    def pow_fp8x23():
        def default():
            x = np.array([1, 2, 3]).astype(np.float64)
            y = np.array([1, 2, 3]).astype(np.float64)
            z = np.array(pow(x, y), dtype=np.float64)

            x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                x.flatten(), FixedImpl.FP8x23))
            y = Tensor(Dtype.FP8x23, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP8x23))
            z = Tensor(Dtype.FP8x23, z.shape, to_fp(
                z.flatten(), FixedImpl.FP8x23))

            name = "pow_fp8x23"
            make_test([x, y], z, "input_0.pow(@input_1)", name)

        def broadcast():
            x = np.array([1, 2, 3]).astype(np.float64)
            y = np.array([2]).astype(np.float64)
            z = np.array(pow(x, y), dtype=np.float64)

            x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                x.flatten(), FixedImpl.FP8x23))
            y = Tensor(Dtype.FP8x23, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP8x23))
            z = Tensor(Dtype.FP8x23, z.shape, to_fp(
                z.flatten(), FixedImpl.FP8x23))

            name = "pow_fp8x23_broadcast"
            make_test([x, y], z, "input_0.pow(@input_1)", name)

        default()
        broadcast()

    @staticmethod
    def and_fp16x16():
        def default():
            x = np.array([1, 2, 3]).astype(np.float64)
            y = np.array([1, 2, 3]).astype(np.float64)
            z = np.array(pow(x, y), dtype=np.float64)

            x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                x.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP16x16))
            z = Tensor(Dtype.FP16x16, z.shape, to_fp(
                z.flatten(), FixedImpl.FP16x16))

            name = "pow_fp16x16"
            make_test([x, y], z, "input_0.pow(@input_1)", name)

        def broadcast():
            x = np.array([1, 2, 3]).astype(np.float64)
            y = np.array([2]).astype(np.float64)
            z = np.array(pow(x, y), dtype=np.float64)

            x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                x.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP16x16))
            z = Tensor(Dtype.FP16x16, z.shape, to_fp(
                z.flatten(), FixedImpl.FP16x16))

            name = "pow_fp16x16_broadcast"
            make_test([x, y], z, "input_0.pow(@input_1)", name)

        default()
        broadcast()
