import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait, get_data_statement


class Tile(RunAll):
     
    @staticmethod
    # We test here with fp8x23 implementation.
    def fp8x23():
        x = np.random.randint(-3, 3, (2, 2, 4, 5)).astype(np.float64)
        k = np.random.randint(0, 5, (4))
        y = np.tile(x, k)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape,  to_fp(
            y.flatten(), FixedImpl.FP8x23))

        name = "tile_fp8x23"
        make_test([x], y, f"input_0.tile(array!{k.tolist()}.span())", name)
     
    @staticmethod
    # We test here with fp16x16 implementation.
    def fp16x16():
        x = np.random.randint(-3, 3, (4, 7, 9)).astype(np.float64)
        k = np.random.randint(0, 5, (3))
        y = np.tile(x, k)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape,  to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "tile_fp16x16"
        make_test([x], y, f"input_0.tile(array!{k.tolist()}.span())", name)
     
    @staticmethod
    # We test here with i8 implementation.
    def i8():
        x = np.random.randint(0, 6, (5)).astype(np.int8)
        k = np.random.randint(0, 5, (1))
        y = np.tile(x, k)

        x = Tensor(Dtype.I8, x.shape, x.flatten())
        y = Tensor(Dtype.I8, y.shape, y.flatten())

        name = "tile_i8"
        make_test([x], y, f"input_0.tile(array!{k.tolist()}.span())", name)
    
    @staticmethod
    # We test here with i32 implementation.
    def i32():
        x = np.random.randint(0, 6, (5, 8)).astype(np.int32)
        k = np.random.randint(0, 5, (2))
        y = np.tile(x, k)

        x = Tensor(Dtype.I32, x.shape, x.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())

        name = "tile_i32"
        make_test([x], y, f"input_0.tile(array!{k.tolist()}.span())", name)
     
    @staticmethod
    # We test here with u32 implementation.
    def u32():
        x = np.random.randint(0, 6, (1, 2)).astype(np.uint32)
        k = np.random.randint(0, 5, (2))
        y = np.tile(x, k)

        x = Tensor(Dtype.U32, x.shape, x.flatten())
        y = Tensor(Dtype.U32, y.shape, y.flatten())

        name = "tile_u32"
        make_test([x], y, f"input_0.tile(array!{k.tolist()}.span())", name)
        