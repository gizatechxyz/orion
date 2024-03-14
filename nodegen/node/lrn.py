import numpy as np
from nodegen.node import RunAll
import math
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

def lrn(x, size, alpha=0.0001, beta=0.75, bias=1.0):  # type: ignore
    if len(x.shape) != 4:
        raise RuntimeError(
            f"LRN only applies on 4D tensors but shape is {x.shape!r}."
        )
    square_sum = np.zeros(x.shape).astype(x.dtype)
    minc = x.shape[1]
    c1 = int(math.floor((size - 1) / 2))
    c2 = int(math.ceil((size - 1) / 2)) + 1
    for c in range(x.shape[1]):
        begin = max(0, c - c1)
        end = min(minc, c + c2)
        square_sum[:, c, :, :] = np.sum(x[:, begin:end, :, :] ** 2, axis=1)
    y = x / ((bias + (alpha / size) * square_sum) ** beta)
    return y.astype(x.dtype)

def get_data_statement(data: np.ndarray, dtype: Dtype) -> list[str]:
    match dtype:
        case Dtype.FP8x23:
            return ["Option::Some(FP8x23 { "+f"mag: {abs(int(x))}, sign: {str(x < 0).lower()} "+"})" for x in data.flatten()]
        case Dtype.FP16x16:
            return ["Option::Some(FP16x16 { "+f"mag: {abs(int(x))}, sign: {str(x < 0).lower()} "+"})" for x in data.flatten()]

class Lrn(RunAll):
    @staticmethod
    def fp8x23():
        alpha = 0.0002
        beta = 0.5
        bias = 2.0
        args = [alpha, beta, bias]
        args_str = get_data_statement(to_fp(np.array(args).flatten(), FixedImpl.FP8x23), Dtype.FP8x23)
        nsize = 3
        x = np.random.uniform(-30, 30, (2, 4, 2, 2)).astype(np.float64)
        y = lrn(x, nsize, *args)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(
            y.flatten(), FixedImpl.FP8x23))

        name = "lrn_fp8x23"
        make_test([x], y, f"NNTrait::lrn(@input_0, {nsize}, {','.join(args_str)})",
                    name, Trait.NN)

    @staticmethod
    def fp16x16():
        alpha = 0.0002
        beta = 0.5
        bias = 2.0
        args = [alpha, beta, bias]
        args_str = get_data_statement(to_fp(np.array(args).flatten(), FixedImpl.FP16x16), Dtype.FP16x16)
        nsize = 3
        x = np.random.uniform(-30, 30, (3, 2, 2, 3)).astype(np.float16)
        y = lrn(x, nsize, *args)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "lrn_fp16x16"
        make_test([x], y, f"NNTrait::lrn(@input_0, {nsize}, {','.join(args_str)})",
                    name, Trait.NN)
        