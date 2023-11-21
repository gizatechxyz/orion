import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


def shrink(input_array: np.ndarray, bias: float, lambd: float) -> np.ndarray:
    output_array = np.where(input_array > lambd, input_array - bias, 
                            np.where(input_array < -lambd, input_array + bias, 0))
    return output_array


class Shrink(RunAll):

    @staticmethod
    def shrink_fp8x23():
        def shrink_hard():
            x = np.random.uniform(-3, 3, (3, 3, 3)).astype(np.float64)
            bias = np.float64(0) # Default value
            lambd = np.float64(1)
            y = shrink(x, bias, lambd)

            x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                x.flatten(), FixedImpl.FP8x23))
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23))

            name = "shrink_hard_fp8x23"
            make_test([x], y, "TensorTrait::shrink(input_0, Option::None(()), Option::Some(FixedTrait::new(8388608, false)))", name)

        def shrink_soft():
            x = np.random.uniform(-3, 3, (3, 3, 3)).astype(np.float64)
            bias = np.float64(1)
            lambd = np.float64(1)
            y = shrink(x, bias, lambd)

            x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                x.flatten(), FixedImpl.FP8x23))
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23))

            name = "shrink_soft_fp8x23"
            make_test([x], y, "TensorTrait::shrink(input_0, Option::Some(FixedTrait::new(8388608, false)), Option::Some(FixedTrait::new(8388608, false)))", name)

        shrink_hard()
        shrink_soft()


    @staticmethod
    def shrink_fp16x16():
        def shrink_hard():
            x = np.random.uniform(-3, 3, (3, 3, 3)).astype(np.float64)
            bias = np.float64(0) # Default value
            lambd = np.float64(1)
            y = shrink(x, bias, lambd)

            x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                x.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16))

            name = "shrink_hard_fp16x16"
            make_test([x], y, "TensorTrait::shrink(input_0, Option::None(()), Option::Some(FixedTrait::new(65536, false)))", name)

        def shrink_soft():
            x = np.random.uniform(-3, 3, (3, 3, 3)).astype(np.float64)
            bias = np.float64(1)
            lambd = np.float64(1)
            y = shrink(x, bias, lambd)

            x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                x.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16))

            name = "shrink_soft_fp16x16"
            make_test([x], y, "TensorTrait::shrink(input_0, Option::Some(FixedTrait::new(65536, false)), Option::Some(FixedTrait::new(65536, false)))", name)

        shrink_hard()
        shrink_soft()