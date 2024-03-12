import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

def global_average_pool(x: np.ndarray) -> np.ndarray:
    axis = tuple(range(2, np.ndim(x)))
    y = np.average(x, axis=axis)
    for _ in axis:
        y = np.expand_dims(y, -1)
    return y  # type: ignore

class Global_average_pool(RunAll):

    @staticmethod
    def fp8x23_2D():
        x = np.random.uniform(-30, 30, (2, 4)).astype(np.float64)
        y = global_average_pool(x)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(
            y.flatten(), FixedImpl.FP8x23))

        name = "global_average_pool_fp8x23_2D"
        make_test([x], y, "NNTrait::global_average_pool(@input_0)",
                    name, Trait.NN)

    @staticmethod
    def fp16x16_2D():
        x = np.random.uniform(-30, 30, (3, 2)).astype(np.float16)
        y = global_average_pool(x)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "global_average_pool_fp16x16_2D"
        make_test([x], y, "NNTrait::global_average_pool(@input_0)",
                    name, Trait.NN)

    @staticmethod
    def fp8x23_3D():
        x = np.random.uniform(-30, 30, (2, 4, 2)).astype(np.float64)
        y = global_average_pool(x)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(
            y.flatten(), FixedImpl.FP8x23))

        name = "global_average_pool_fp8x23_3D"
        make_test([x], y, "NNTrait::global_average_pool(@input_0)",
                    name, Trait.NN)

    @staticmethod
    def fp16x16_3D():
        x = np.random.uniform(-30, 30, (3, 2, 2)).astype(np.float16)
        y = global_average_pool(x)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "global_average_pool_fp16x16_3D"
        make_test([x], y, "NNTrait::global_average_pool(@input_0)",
                    name, Trait.NN)

    @staticmethod
    def fp8x23_4D():
        x = np.random.uniform(-30, 30, (2, 4, 2, 2)).astype(np.float64)
        y = global_average_pool(x)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(
            y.flatten(), FixedImpl.FP8x23))

        name = "global_average_pool_fp8x23_4D"
        make_test([x], y, "NNTrait::global_average_pool(@input_0)",
                    name, Trait.NN)

    @staticmethod
    def fp16x16_4D():
        x = np.random.uniform(-30, 30, (3, 2, 2, 3)).astype(np.float16)
        y = global_average_pool(x)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "global_average_pool_fp16x16_4D"
        make_test([x], y, "NNTrait::global_average_pool(@input_0)",
                    name, Trait.NN)
        
    # @staticmethod
    # def fp32x32():
    #     x = np.random.uniform(-30, 30, (2, 4)).astype(np.float64)
    #     y = global_average_pool(x)

    #     x = Tensor(Dtype.FP32x32, x.shape, to_fp(
    #         x.flatten(), FixedImpl.FP32x32))
    #     y = Tensor(Dtype.FP32x32, y.shape, to_fp(
    #         y.flatten(), FixedImpl.FP32x32))

    #     name = "global_average_pool_fp32x32"
    #     make_test([x], y, "NNTrait::global_average_pool(@input_0)",
    #                 name, Trait.NN)
