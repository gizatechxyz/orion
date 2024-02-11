import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

def random_uniform_like(x: np.ndarray, high: int=1,low: int=0,seed: int=25) ->np.ndarray:
    dtype = np.float64
    if seed is None or np.isnan(seed):  # type: ignore
        state = np.random.RandomState()
    else:
        state = np.random.RandomState(seed=int(seed))  # type: ignore
    res = state.rand(*x.shape).astype(dtype)
    res *= high - low  # type: ignore
    res += low  # type: ignore
    return (res.astype(dtype),)

def get_data_statement(data: np.ndarray, dtype: Dtype) -> list[str]:
    match dtype:
        case Dtype.FP8x23:
            return ["Option::Some(FP8x23 { "+f"mag: {abs(int(x))}, sign: {str(x < 0).lower()} "+"})" for x in data.flatten()]
        case Dtype.FP16x16:
            return ["Option::Some(FP16x16 { "+f"mag: {abs(int(x))}, sign: {str(x < 0).lower()} "+"})" for x in data.flatten()]
        case Dtype.U32:
            return [f"Option::Some({int(x)})" for x in data.flatten()]

class Random_uniform_like(RunAll):

    @staticmethod
    def fp8x23():
        x = np.random.uniform(1, 10, (1, 2, 2, 4)).astype(np.float64)
        y = random_uniform_like(x)

        args = [10, 1]
        args_str = get_data_statement(to_fp(np.array(args).flatten(), FixedImpl.FP8x23), Dtype.FP8x23)
        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y[0].shape, to_fp(
            y[0].flatten(), FixedImpl.FP8x23))

        name = "random_uniform_like_fp8x23"
        make_test(
            [x], # List of input tensors.
            y, # The expected output result.
            f"TensorTrait::random_uniform_like(@input_0, {','.join(args_str)}, Option::Some(354145))", # The code signature.
            name # The name of the generated folder.
        )

    @staticmethod
    def fp16x16():
        x = np.random.uniform(1, 10, (1, 2, 2, 4)).astype(np.float16)
        y = random_uniform_like(x)

        args = [10, 1]
        args_str = get_data_statement(to_fp(np.array(args).flatten(), FixedImpl.FP16x16), Dtype.FP16x16)


        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y[0].shape, to_fp(
            y[0].flatten(), FixedImpl.FP16x16))

        name = "random_uniform_like_fp16x16"
        make_test(
            [x], # List of input tensors.
            y, # The expected output result.
            f"TensorTrait::random_uniform_like(@input_0, {','.join(args_str)}, Option::Some(354145))", # The code signature.
            name # The name of the generated folder.
        )

    # @staticmethod
    # def fp64x64():
    #     x = np.random.uniform(-3, 3, (1, 2, 2, 4)).astype(np.float64)
    #     y = random_uniform_like(x)

    #     x = Tensor(Dtype.FP64x64, x.shape, to_fp(
    #         x.flatten(), FixedImpl.FP64x64))
    #     y = Tensor(Dtype.FP64x64, y[0].shape, to_fp(
    #         y[0].flatten(), FixedImpl.FP64x64))

    #     name = "random_uniform_like_fp64x64"
    #     make_test([x], y, "TensorTrait::random_uniform_like(@input_0, 5, 1, 10)",
    #                 name)

    # @staticmethod
    # def fpi8():
    #     x = np.random.randint(-3, 3, (1, 2, 2, 4)).astype(np.int8)
    #     y = random_uniform_like(x)

    #     x = Tensor(Dtype.I8, x.shape, x.flatten())
    #     y = Tensor(Dtype.I8, y[0].shape, y[0].flatten())

    #     name = "random_uniform_like_i8"
    #     make_test([x], y, "TensorTrait::random_uniform_like(@input_0, 5, 1, 10)",
    #                 name)

    # @staticmethod
    # def fpi32():
    #     x = np.random.randint(-3, 3, (1, 2, 2, 4)).astype(np.int32)
    #     y = random_uniform_like(x)

    #     x = Tensor(Dtype.I32, x.shape, x.flatten())
    #     y = Tensor(Dtype.I32, y[0].shape, y[0].flatten())

    #     name = "random_uniform_like_i32"
    #     make_test([x], y, "TensorTrait::random_uniform_like(@input_0, 5, 1, 10)",
    #                 name)


    # @staticmethod
    # def fpu32():
    #     x = np.random.randint(-3, 3, (1, 2, 2, 4)).astype(np.uint32)
    #     y = random_uniform_like(x)
    #     args = [5, 1, 10]
    #     args_str = get_data_statement(np.array(args).flatten(), Dtype.U32)


    #     x = Tensor(Dtype.U32, x.shape, x.flatten())
    #     y = Tensor(Dtype.U32, y[0].shape, y[0].flatten())

    #     name = "random_uniform_like_u32"
    #     make_test(
    #         [x], # List of input tensors.
    #         y, # The expected output result.
    #         f"TensorTrait::random_uniform_like(@input_0, {','.join(args_str)})", # The code signature.
    #         name # The name of the generated folder.
    #     )
