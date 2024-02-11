import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

def depth_to_space(data: np.ndarray, blocksize: int = 2, mode = "DCR") -> np.ndarray:
    if len(data.shape) != 4:
        raise RuntimeError(f"Unexpected shape {data.shape!r}.")
    b, c, h, w = data.shape
    if mode == "DCR":
        tmpshape = (
            b,
            blocksize,
            blocksize,
            c // (blocksize * blocksize),
            h,
            w,
        )
        reshaped = data.reshape(tmpshape)
        transposed = np.transpose(reshaped, [0, 3, 4, 1, 5, 2])
    else:
        # assert mode == "CRD"
        tmpshape = (
            b,
            c // (blocksize * blocksize),
            blocksize,
            blocksize,
            h,
            w,
        )
        reshaped = data.reshape(tmpshape)
        transposed = np.transpose(reshaped, [0, 1, 4, 2, 5, 3])
    finalshape = (
        b,
        c // (blocksize * blocksize),
        h * blocksize,
        w * blocksize,
    )
    y = np.reshape(transposed, finalshape)
    return y

class Depth_to_space(RunAll):

    @staticmethod
    def fp8x23():
        x = np.random.uniform(-3, 3, (1, 4, 2, 2)).astype(np.float64)
        y = depth_to_space(x)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(
            y.flatten(), FixedImpl.FP8x23))

        name = "depth_to_space_fp8x23"
        make_test([x], y, "NNTrait::depth_to_space(@input_0, 2, 'DCR')",
                    name, Trait.NN)

    @staticmethod
    def fp16x16():
        x = np.random.uniform(-3, 3, (1, 4, 2, 2)).astype(np.float16)
        y = depth_to_space(x)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "depth_to_space_fp16x16"
        make_test([x], y, "NNTrait::depth_to_space(@input_0, 2, 'DCR')",
                    name, Trait.NN)
        
    # @staticmethod
    # def fp64x64():
    #     x = np.random.uniform(-3, 3, (1, 4, 2, 2)).astype(np.float64)
    #     y = depth_to_space(x)

    #     x = Tensor(Dtype.FP64x64, x.shape, to_fp(
    #         x.flatten(), FixedImpl.FP64x64))
    #     y = Tensor(Dtype.FP64x64, y.shape, to_fp(
    #         y.flatten(), FixedImpl.FP64x64))

    #     name = "depth_to_space_fp64x64"
    #     make_test([x], y, "NNTrait::depth_to_space(@input_0, 2, 'DCR')",
    #                 name, Trait.NN)

    @staticmethod
    def fpi8():
        x = np.random.randint(-3, 3, (1, 4, 2, 2)).astype(np.int8)
        y = depth_to_space(x)

        x = Tensor(Dtype.I8, x.shape, x.flatten())
        y = Tensor(Dtype.I8, y.shape, y.flatten())

        name = "depth_to_space_i8"
        make_test([x], y, "NNTrait::depth_to_space(@input_0, 2, 'DCR')",
                    name, Trait.NN)

    @staticmethod
    def fpi32():
        x = np.random.randint(-3, 3, (1, 4, 2, 2)).astype(np.int32)
        y = depth_to_space(x)

        x = Tensor(Dtype.I32, x.shape, x.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())

        name = "depth_to_space_i32"
        make_test([x], y, "NNTrait::depth_to_space(@input_0, 2, 'CRD')",
                    name, Trait.NN)


    @staticmethod
    def fpu32():
        x = np.random.randint(-3, 3, (1, 4, 2, 2)).astype(np.uint32)
        y = depth_to_space(x)

        x = Tensor(Dtype.U32, x.shape, x.flatten())
        y = Tensor(Dtype.U32, y.shape, y.flatten())

        name = "depth_to_space_u32"
        make_test([x], y, "NNTrait::depth_to_space(@input_0, 2, 'CRD')",
                    name, Trait.NN)
