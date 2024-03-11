import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait, get_data_statement

def eye_like(data, k=None) -> np.ndarray:  # type: ignore
    if data is None:
        _dtype = np.float32
    else:
        _dtype = data.dtype
    shape = data.shape
    if len(shape) == 1:
        sh = (shape[0], shape[0])
    elif len(shape) == 2:
        sh = shape
    else:
        raise RuntimeError(f"EyeLike only accept 1D or 2D tensors not {shape!r}.")
    return np.eye(*sh, k=k, dtype=_dtype)  # type: ignore

class Eye_like(RunAll):
     
    @staticmethod
    # We test here with fp8x23 implementation.
    def fp8x23():
        x = np.random.randint(-3, 3, (2, 2)).astype(np.float64)
        k = 0
        y = eye_like(x, k)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape,  to_fp(
            y.flatten(), FixedImpl.FP8x23))

        name = "eye_like_fp8x23"
        make_test([x], y, f"input_0.eye_like(Option::Some({k}))", name)
     
    @staticmethod
    # We test here with fp16x16 implementation.
    def fp16x16():
        x = np.random.randint(-3, 3, (4, 7)).astype(np.float64)
        k = 3
        y = eye_like(x, k)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape,  to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "eye_like_fp16x16"
        make_test([x], y, f"input_0.eye_like(Option::Some({k}))", name)
     
    @staticmethod
    # We test here with i8 implementation.
    def i8():
        x = np.random.randint(0, 6, (5, 3)).astype(np.int8)
        k = -2
        y = eye_like(x, k)

        x = Tensor(Dtype.I8, x.shape, x.flatten())
        y = Tensor(Dtype.I8, y.shape, y.flatten())

        name = "eye_like_i8"
        make_test([x], y, f"input_0.eye_like(Option::Some({k}))", name)
    
    @staticmethod
    # We test here with i32 implementation.
    def i32():
        x = np.random.randint(0, 6, (5, 8)).astype(np.int32)
        k = 9
        y = eye_like(x, k)

        x = Tensor(Dtype.I32, x.shape, x.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())

        name = "eye_like_i32"
        make_test([x], y, f"input_0.eye_like(Option::Some({k}))", name)
     
    @staticmethod
    # We test here with u32 implementation.
    def u32():
        x = np.random.randint(0, 6, (1, 2)).astype(np.uint32)
        k = -7
        y = eye_like(x, k)

        x = Tensor(Dtype.U32, x.shape, x.flatten())
        y = Tensor(Dtype.U32, y.shape, y.flatten())

        name = "eye_like_u32"
        make_test([x], y, f"input_0.eye_like(Option::Some({k}))", name)
        