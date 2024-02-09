import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

def space_to_depth(data: np.ndarray, blocksize: int = 2) -> np.ndarray:
    if len(data.shape) != 4:
        raise RuntimeError(f"Unexpected shape {data.shape!r}.")
    b, C, H, W = data.shape
    tmpshape = (
        b,
        C,
        H // blocksize,
        blocksize,
        W // blocksize,
        blocksize,
    )
    reshaped = np.reshape(data, tmpshape)
    transposed = np.transpose(reshaped, [0, 3, 5, 1, 2, 4])
    finalshape = (
        b,
        C * blocksize * blocksize,
        H // blocksize,
        W // blocksize,
    )
    y = np.reshape(transposed, finalshape).astype(data.dtype)
    return y

class Space_to_depth(RunAll):


    @staticmethod
    def fp8x23():
        x = np.random.uniform(-3, 3, (1, 2, 2, 4)).astype(np.float64)
        y = space_to_depth(x)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(
            y.flatten(), FixedImpl.FP8x23))

        name = "space_to_depth_fp8x23"
        make_test([x], y, "NNTrait::space_to_depth(@input_0, 2)",
                    name, Trait.NN)

    @staticmethod
    def fp16x16():
        x = np.random.uniform(-3, 3, (1, 2, 2, 4)).astype(np.float16)
        y = space_to_depth(x)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "space_to_depth_fp16x16"
        make_test([x], y, "NNTrait::space_to_depth(@input_0, 2)",
                    name, Trait.NN)
        
    # @staticmethod
    # def fp64x64():
    #     x = np.random.uniform(-3, 3, (1, 2, 2, 4)).astype(np.float64)
    #     y = space_to_depth(x)

    #     x = Tensor(Dtype.FP64x64, x.shape, to_fp(
    #         x.flatten(), FixedImpl.FP64x64))
    #     y = Tensor(Dtype.FP64x64, y.shape, to_fp(
    #         y.flatten(), FixedImpl.FP64x64))

    #     name = "space_to_depth_fp64x64"
    #     make_test([x], y, "NNTrait::space_to_depth(@input_0, 2)",
    #                 name, Trait.NN)

    @staticmethod
    def fpi8():
        x = np.random.randint(-3, 3, (1, 2, 2, 4)).astype(np.int8)
        y = space_to_depth(x)

        x = Tensor(Dtype.I8, x.shape, x.flatten())
        y = Tensor(Dtype.I8, y.shape, y.flatten())

        name = "space_to_depth_i8"
        make_test([x], y, "NNTrait::space_to_depth(@input_0, 2)",
                    name, Trait.NN)

    @staticmethod
    def fpi32():
        x = np.random.randint(-3, 3, (1, 2, 2, 4)).astype(np.int32)
        y = space_to_depth(x)

        x = Tensor(Dtype.I32, x.shape, x.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())

        name = "space_to_depth_i32"
        make_test([x], y, "NNTrait::space_to_depth(@input_0, 2)",
                    name, Trait.NN)


    @staticmethod
    def fpu32():
        x = np.random.randint(-3, 3, (1, 2, 2, 4)).astype(np.uint32)
        y = space_to_depth(x)

        x = Tensor(Dtype.U32, x.shape, x.flatten())
        y = Tensor(Dtype.U32, y.shape, y.flatten())

        name = "space_to_depth_u32"
        make_test([x], y, "NNTrait::space_to_depth(@input_0, 2)",
                    name, Trait.NN)
