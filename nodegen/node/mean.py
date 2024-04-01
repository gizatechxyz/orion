import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait, get_data_statement

def mean(*args) -> np.ndarray:  # type: ignore
    res = args[0].copy()
    for m in args[1:]:
        res += m
    return (res / len(args)).astype(args[0].dtype)

class Mean(RunAll):
     
    @staticmethod
    # We test here with fp8x23 implementation.
    def fp8x23():
        x = np.random.randint(-3, 3, (2, 2)).astype(np.float64)
        y = np.random.randint(-3, 3, (2, 2)).astype(np.float64)
        z = mean(x, y)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape,  to_fp(
            y.flatten(), FixedImpl.FP8x23))
        z = Tensor(Dtype.FP8x23, z.shape,  to_fp(
            z.flatten(), FixedImpl.FP8x23))

        name = "mean_fp8x23"
        make_test([x, y], z, "TensorTrait::mean(array![input_0, input_1].span())", name)
     
    @staticmethod
    # We test here with fp16x16 implementation.
    def fp16x16():
        x = np.random.randint(-3, 3, (2, 2)).astype(np.float64)
        y = np.random.randint(-3, 3, (2, 2)).astype(np.float64)
        z = mean(x, y)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape,  to_fp(
            y.flatten(), FixedImpl.FP16x16))
        z = Tensor(Dtype.FP16x16, z.shape,  to_fp(
            z.flatten(), FixedImpl.FP16x16))

        name = "mean_fp16x16"
        make_test([x, y], z, "TensorTrait::mean(array![input_0, input_1].span())", name)
     
    @staticmethod
    # We test here with i8 implementation.
    def i8():
        x = np.random.randint(0, 6, (2, 2)).astype(np.int8)
        y = np.random.randint(0, 6, (2, 2)).astype(np.int8)
        z = np.random.randint(0, 6, (2, 2)).astype(np.int8)
        m = mean(x, y, z)

        x = Tensor(Dtype.I8, x.shape, x.flatten())
        y = Tensor(Dtype.I8, y.shape, y.flatten())
        z = Tensor(Dtype.I8, z.shape, z.flatten())
        m = Tensor(Dtype.I8, m.shape, m.flatten())

        name = "mean_i8"
        make_test([x, y, z], m, "TensorTrait::mean(array![input_0, input_1, input_2].span())", name)
    
    @staticmethod
    # We test here with i32 implementation.
    def i32():
        x = np.random.randint(0, 6, (2, 2)).astype(np.int32)
        y = np.random.randint(0, 6, (2, 2)).astype(np.int32)
        z = np.random.randint(0, 6, (2, 2)).astype(np.int32)
        m = mean(x, y, z)

        x = Tensor(Dtype.I32, x.shape, x.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())
        z = Tensor(Dtype.I32, z.shape, z.flatten())
        m = Tensor(Dtype.I32, m.shape, m.flatten())

        name = "mean_i32"
        make_test([x, y, z], m, "TensorTrait::mean(array![input_0, input_1, input_2].span())", name)
     
    @staticmethod
    # We test here with u32 implementation.
    def u32():
        x = np.random.randint(0, 6, (2, 2)).astype(np.uint32)
        y = np.random.randint(0, 6, (2, 2)).astype(np.uint32)
        z = np.random.randint(0, 6, (2, 2)).astype(np.uint32)
        m = mean(x, y, z)

        x = Tensor(Dtype.U32, x.shape, x.flatten())
        y = Tensor(Dtype.U32, y.shape, y.flatten())
        z = Tensor(Dtype.U32, z.shape, z.flatten())
        m = Tensor(Dtype.U32, m.shape, m.flatten())

        name = "mean_u32"
        make_test([x, y, z], m, "TensorTrait::mean(array![input_0, input_1, input_2].span())", name)
        